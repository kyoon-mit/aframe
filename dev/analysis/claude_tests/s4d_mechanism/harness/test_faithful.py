"""Check the harness against the pipeline's own code on identical random draws.

1. validation files: harness.valid_fnames vs BaseAframeDataset.train_val_split
2. background reads: Hdf5TimeSeriesDataset.sample_batch (indices recorded) vs harness.read_background
3. waveform slice: BaseAframeDataset.slice_waveforms on full 32 s rows vs harness.read_waveforms columns
4. injection and whitening: DenoiserOnlyAframeDataset.inject (real method, transforms from
   its own build_transforms; only the random draws are fixed) vs harness.assemble
Run from aframe/projects/train with its .venv python.
"""
import sys
import types
from pathlib import Path

import h5py
import numpy as np
import torch
from lightning.fabric.utilities.data import AttributeDict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

import train.data.supervised.supervised as sup  # noqa: E402
from ml4gw.dataloading import Hdf5TimeSeriesDataset  # noqa: E402
from ml4gw.utils.slicing import slice_kernels  # noqa: E402
from train.data.base import BaseAframeDataset  # noqa: E402
from train.data.supervised.denoiser import DenoiserOnlyAframeDataset  # noqa: E402
from train.transforms import ChirpMass, MassRatio  # noqa: E402

torch.set_num_threads(8)
HP = AttributeDict(
    background_dir=harness.BACKGROUND_DIR, ifos=harness.IFOS, sample_rate=harness.SAMPLE_RATE,
    kernel_length=harness.KERNEL_LENGTH, fduration=harness.FDURATION, psd_length=harness.PSD_LENGTH,
    fftlength=harness.FFTLENGTH, highpass=harness.HIGHPASS, lowpass=harness.LOWPASS,
    left_pad=harness.LEFT_PAD, right_pad=harness.RIGHT_PAD, min_valid_duration=harness.MIN_VALID_DURATION,
    waveform_prob=1.0, model_input_sample_rate=None, batch_size=32,
)


def pipeline_object():
    """A DenoiserOnlyAframeDataset with no __init__/setup: hparams and the
    transforms its build_transforms makes, nothing loaded."""
    obj = DenoiserOnlyAframeDataset.__new__(DenoiserOnlyAframeDataset)
    obj._hparams = AttributeDict(HP)
    obj.background_dir = harness.BACKGROUND_DIR
    obj._logger = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None)
    obj.waveform_sampler = types.SimpleNamespace(right_pad=float(h5py.File(harness.VAL_WAVEFORM_FILE, "r").attrs["right_pad"]))
    obj.build_transforms()
    return obj


def check_files(obj):
    train_f, valid_f = BaseAframeDataset.train_val_split(obj)
    ok = valid_f == harness.valid_fnames()
    print(f"[1] validation files equal: {ok}  {valid_f}")
    return ok


def check_background():
    files = harness.valid_fnames()
    ds = Hdf5TimeSeriesDataset(files, channels=harness.IFOS, kernel_size=harness.SAMPLE_SIZE,
                               batch_size=6, batches_per_epoch=1, coincident=False, num_files_per_batch=1)
    rec = {}
    orig_choice, orig_randint = ds.sample_fnames, np.random.randint

    def sample_fnames(size):
        f = orig_choice(size)
        rec["fnames"] = f
        return f

    def randint(*a, **k):
        r = orig_randint(*a, **k)
        rec.setdefault("idx", []).append(r)
        return r

    ds.sample_fnames = sample_fnames
    np.random.seed(5)
    np.random.randint = randint
    try:
        x = ds.sample_batch()
    finally:
        np.random.randint = orig_randint
    fn = rec["fnames"]                                   # (6, 2) one file per window and detector
    uniq, inv = np.unique(fn, return_inverse=True)
    inv = inv.reshape(-1)
    starts = np.zeros((6, 2), np.int64)
    for i, idx in enumerate(rec["idx"]):
        flat = np.where(inv == i)[0]
        starts[flat // 2, flat % 2] = idx
    same_file = bool((fn[:, 0] == fn[:, 1]).all())
    handles = {f: h5py.File(f, "r") for f in files}
    fidx = np.array([files.index(f) for f in fn[:, 0]])
    mine = harness.read_background([handles[f] for f in files], fidx, starts)
    diff = float((mine - x).abs().max())
    print(f"[2] background: one file per batch {len(set(fn.ravel())) == 1}, both detectors from the same file {same_file}, "
          f"H1/L1 starts differ in {int((starts[:, 0] != starts[:, 1]).sum())}/6 windows, max |harness - dataset| = {diff:g}")
    return diff == 0.0


def check_slice(obj):
    col0, col1, merger_col = harness._waveform_slice()
    rows = np.array([3, 77, 150001])
    with h5py.File(harness.WAVEFORM_FILE, "r") as wf:
        full = np.stack([np.stack([wf["waveforms/cross"][r], wf["waveforms/plus"][r]]) for r in rows])
        mine = harness.read_waveforms(wf, rows, col0, col1)
    piped = BaseAframeDataset.slice_waveforms(obj, torch.tensor(full)).numpy()
    ok = piped.shape == mine.shape and np.array_equal(piped, mine)
    print(f"[3] slice_waveforms columns {col0}:{col1} (merger at {merger_col} of the slice), pipeline shape {piped.shape}, equal {ok}")
    return ok


def check_inject(obj, n=8, seed=11):
    tr = harness._Transforms("cpu")
    files = harness.valid_fnames()
    handles = [h5py.File(f, "r") for f in files]
    wf = h5py.File(harness.WAVEFORM_FILE, "r")
    col0, col1, _ = harness._waveform_slice()
    d = harness._draws(n, seed, [len(h["H1"]) for h in handles], wf["waveforms/cross"].shape[0], (col1 - col0) - harness.KERNEL_SIZE)
    X = harness.read_background(handles, d["bg_file"], d["bg_start"])
    wave64 = harness.read_waveforms(wf, d["waveform_row"], col0, col1)
    params = {k: torch.tensor(wf[f"parameters/{k}"][:][d["waveform_row"]]) for k in ("mass_1", "mass_2")}
    t = torch.from_numpy

    # harness
    mine = harness.assemble(tr, X.clone(), torch.tensor(wave64).float(), t(d["dec"]), t(d["psi"]), t(d["phi"]),
                            t(d["snr"]), t(d["inverted"]), t(d["reversed"]), t(d["kernel_offset"]))

    # pipeline, with its random draws replaced by the same values
    class Fixed:
        def __init__(self, v):
            self.v = torch.as_tensor(v)

        def sample(self, shape):
            assert tuple(shape) == (len(self.v),)
            return self.v.clone()

    def inverter(Z):
        Z[t(d["inverted"])] *= -1
        return Z

    def reverser(Z):
        m = t(d["reversed"])
        Z[m] = Z[m].flip(-1)
        return Z

    obj.inverter, obj.reverser = inverter, reverser
    obj.dec, obj.psi, obj.phi = Fixed(d["dec"]), Fixed(d["psi"]), Fixed(d["phi"])
    obj.val_snr_sampler, obj.snr_sampler = Fixed(d["snr"]), None
    obj.trainer = types.SimpleNamespace(training=False, validating=True, sanity_checking=False)
    obj.waveforms_from_disk = True
    obj.signal_repeats = 1
    obj.swapper = obj.muter = None
    obj.swap_prob = obj.mute_prob = 0
    obj.param_transforms = [ChirpMass(), MassRatio()]
    orig_sk, orig_rp = sup.sample_kernels, torch.randperm
    sup.sample_kernels = lambda R, kernel_size, coincident=True: slice_kernels(R, t(d["kernel_offset"]), kernel_size)
    torch.randperm = lambda k, *a, **kw: torch.arange(k)
    try:
        Xp, Xcp, y, params_out = obj.inject(X=X.clone(), waveforms=torch.tensor(wave64), params=params)
    finally:
        sup.sample_kernels, torch.randperm = orig_sk, orig_rp
    dX = float((Xp - mine["X"]).abs().max())
    dC = float((Xcp - mine["X_clean"]).abs().max())
    mc_h = (lambda m1, m2: (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)(params["mass_1"].float(), params["mass_2"].float())
    dM = float((params_out["chirp_mass"] - mc_h).abs().max())
    print(f"[4] inject on {n} windows: dtype pipeline {Xp.dtype}/{Xcp.dtype}, harness {mine['X'].dtype}/{mine['X_clean'].dtype}; "
          f"max |dX| = {dX:g}, max |dX_clean| = {dC:g} (X_clean rms {float(Xcp.pow(2).mean().sqrt()):.4f}), max |d chirp_mass| = {dM:g}, y all 1: {bool((y == 1).all())}")
    for h in handles:
        h.close()
    wf.close()
    return dX == 0.0 and dC == 0.0 and dM == 0.0


if __name__ == "__main__":
    obj = pipeline_object()
    results = [check_files(obj), check_background(), check_slice(obj), check_inject(obj)]
    print("ALL EQUAL" if all(results) else f"MISMATCH {results}")
