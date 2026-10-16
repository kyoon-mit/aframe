"""Noisy windows as the harness makes them, with three options the harness fixes:

- ``bg``: "valid" (the two validation background files, as the harness) or "train" (every
  other O3a background file, the files the networks were trained on);
- ``snr_min``: lower end of the SNR power law (index -3, maximum 100); 4 is validation, 8 training;
- ``rows``: the waveform rows to draw from, without replacement.

Everything else (draw order, projection, rescaling, kernel offset, whitening) is the harness's
own code: ``harness.assemble``, ``read_background``, ``read_waveforms``, ``_waveform_slice``.
"""
import glob
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

HARNESS = Path(__file__).resolve().parents[1] / "harness"
sys.path.insert(0, str(HARNESS))
import harness  # noqa: E402


def bg_files(which):
    valid = harness.valid_fnames()
    if which == "valid":
        return valid
    every = sorted(str(Path(f)) for f in glob.glob(f"{harness.BACKGROUND_DIR}/background/*.hdf5"))
    train = [f for f in every if f not in set(valid)]
    if not train or len(train) + len(valid) != len(every):
        raise RuntimeError("background split failed")
    return train


def draws(n, seed, bg_sizes, rows, n_offsets, snr_min):
    """``harness._draws`` with rows from ``rows`` and SNR from [snr_min, 100]."""
    rng = np.random.default_rng(seed)
    sizes = np.array(bg_sizes, dtype=np.int64)
    d = {}
    d["bg_file"] = rng.choice(len(sizes), size=n, p=sizes / sizes.sum())
    max_idx = sizes[d["bg_file"]] - harness.SAMPLE_SIZE
    d["bg_start"] = rng.integers(0, max_idx[:, None], size=(n, len(harness.IFOS)))
    d["inverted"] = rng.random((n, len(harness.IFOS))) < 0.5
    d["reversed"] = rng.random((n, len(harness.IFOS))) < 0.5
    d["waveform_row"] = np.asarray(rows)[rng.choice(len(rows), size=n, replace=False)]
    d["kernel_offset"] = rng.integers(0, n_offsets, size=n)
    from ml4gw.distributions import Cosine, PowerLaw

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        d["dec"] = Cosine().sample((n,)).numpy()
        d["psi"] = torch.distributions.Uniform(0.0, 3.14159, validate_args=False).sample((n,)).numpy()
        d["phi"] = torch.distributions.Uniform(-3.14159, 3.14159, validate_args=False).sample((n,)).numpy()
        d["snr"] = PowerLaw(minimum=float(snr_min), maximum=harness.SNR_MAX,
                            index=harness.SNR_INDEX).sample((n,)).numpy()
    return d


def iter_windows(n, seed, bg="valid", snr_min=4.0, rows=None, batch_size=64, device="cpu"):
    """Yield ``(start, X, labels)``: X (b, 2, 8192) float32 noisy whitened windows; labels of all n."""
    tr = harness._Transforms(device)
    files = bg_files(bg)
    handles = [h5py.File(f, "r") for f in files]
    sizes = [len(h[harness.IFOS[0]]) for h in handles]
    wf = h5py.File(harness.WAVEFORM_FILE, "r")
    try:
        n_wf = wf["waveforms/cross"].shape[0]
        rows = np.arange(n_wf) if rows is None else np.asarray(rows)
        col0, col1, merger_col = harness._waveform_slice()
        d = draws(n, seed, sizes, rows, (col1 - col0) - harness.KERNEL_SIZE, snr_min)
        d["merger_sample"] = merger_col - d["kernel_offset"] - harness.FILTER_SIZE // 2
        params = {k: wf[f"parameters/{k}"][:] for k in wf["parameters"].keys()}
        r = d["waveform_row"]
        m1 = torch.tensor(params["mass_1"][r]).float()
        m2 = torch.tensor(params["mass_2"][r]).float()
        d["chirp_mass"] = ((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2).numpy()
        for k, v in params.items():
            d[f"file_{k}"] = v[r]
        d["bg_files"] = np.array(files)
        dev = tr.device

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        for i0 in range(0, n, batch_size):
            sl = slice(i0, min(n, i0 + batch_size))
            X = harness.read_background(handles, d["bg_file"][sl], d["bg_start"][sl]).to(dev)
            wave = torch.tensor(harness.read_waveforms(wf, r[sl], col0, col1)).to(dev).float()
            out = harness.assemble(tr, X, wave, t(d["dec"][sl]), t(d["psi"][sl]), t(d["phi"][sl]),
                                   t(d["snr"][sl]), t(d["inverted"][sl]), t(d["reversed"][sl]),
                                   t(d["kernel_offset"][sl]), want=("X",))
            yield sl.start, out["X"].float().cpu().numpy(), d
    finally:
        for h in handles:
            h.close()
        wf.close()
