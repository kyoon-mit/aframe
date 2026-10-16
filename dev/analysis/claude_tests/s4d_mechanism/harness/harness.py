"""Validation-like windows without the aframe data module, and loaders for trained networks.

Mirrors the validation path of ``DenoiserOnlyAframeDataset`` (configs in
aframe/dev/local/train/claude_tests/{oracle,long_conv}):

- background: validation files of ``BaseAframeDataset.train_val_split`` (O3a,
  background-1238412673-14008 and background-1238384700-27833), 25 s per
  detector at independent start times (``Hdf5TimeSeriesDataset``,
  coincident=False), file chosen in proportion to its length;
- PSD from the first 20 s (``PsdEstimator``: 2 s FFTs, median, fast); last 5 s
  kept, each detector inverted and reversed with probability 0.5;
- waveforms: rows of the training waveform file (the file the validation loop
  reads; its rows are i.i.d.), sliced as ``slice_waveforms``, projected with sky
  angles from the config distributions, rescaled to an SNR from the power law
  with index -3 on [4, 100], one 5 s kernel at a uniform offset (``sample_kernels``);
- whitened (``Whiten``: 1 s filter, 20 Hz highpass) to (2, 8192) windows; the
  geocentric merger falls at sample 7169-8192.

``X`` is the noisy-network input, ``X_clean`` the signal-only-network input
(same signal, same PSD). Every window carries a signal. Draws depend on
``(n, seed)``: a smaller ``n`` with the same seed gives different windows.

Saved sets in data/ (float16 windows; float32 copies as ``*_f32.npy``):
clean_X.npy (4000, 2, 8192) seed 1001; noisy_X.npy and noisy_Xclean.npy
(20000, 2, 8192) seed 2002; labels in ``<set>_labels.npz`` and
``<set>_{chirp_mass,snr,merger_sample}.npy``; network estimates in
``est_<set>__<run>__<tag>.npy``. Read slices with ``read_rows``.

Usage (from aframe/projects/train, with its .venv python):

    import sys; sys.path.insert(0, HARNESS_DIR)
    import harness
    labels, paths = harness.load_set("clean")
    X = harness.read_rows(paths["X"], 0, 1000)                    # (1000, 2, 8192) float32
    net = harness.load_net("sig_only_reg_resnet", "ck_47")        # copy in ckpt/
    harness.score(net(X), labels["chirp_mass"][:1000])            # within 1/2/5/10 %
    w = harness.noisy_windows(256, seed=7)                        # fresh windows + labels
"""

import glob
import importlib
import os
import types
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

torch.set_num_threads(min(8, torch.get_num_threads()))

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt"
CONFIG_DIRS = [
    Path("/home/kyoon/SSM-BNS/aframe/dev/local/train/claude_tests/oracle"),
    Path("/home/kyoon/SSM-BNS/aframe/dev/local/train/claude_tests/long_conv"),
]

# data settings shared by all seven configs (checked against each checkpoint's
# datamodule_hyper_parameters)
BACKGROUND_DIR = "/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz"
WAVEFORM_FILE = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
VAL_WAVEFORM_FILE = "/home/kyoon/SSM-BNS/DATA/aframe/val/end_o3_ratesandpops_bns_snr4.hdf5"
IFOS = ["H1", "L1"]
SAMPLE_RATE = 2048.0
KERNEL_LENGTH = 4.0
FDURATION = 1.0
PSD_LENGTH = 20.0
FFTLENGTH = 2.0
HIGHPASS = 20.0
LOWPASS = None
LEFT_PAD = 3.5
RIGHT_PAD = 0.0
MIN_VALID_DURATION = 15000.0
SNR_MIN, SNR_MAX, SNR_INDEX = 4.0, 100.0, -3
VALIDATION_BATCH = 32

# sizes, as BaseAframeDataset computes them
SAMPLE_SIZE = int((KERNEL_LENGTH + FDURATION + PSD_LENGTH) * SAMPLE_RATE)   # 51200
FILTER_SIZE = int(FDURATION * SAMPLE_RATE)                                  # 2048
KERNEL_SIZE = int(KERNEL_LENGTH * SAMPLE_RATE) + FILTER_SIZE                # 10240, unwhitened
WINDOW_SIZE = int(KERNEL_LENGTH * SAMPLE_RATE)                              # 8192, whitened
LEFT_PAD_SIZE = int(LEFT_PAD * SAMPLE_RATE) + FILTER_SIZE // 2              # 8192
RIGHT_PAD_SIZE = int(RIGHT_PAD * SAMPLE_RATE) + FILTER_SIZE // 2            # 1024


def valid_fnames(background_dir=BACKGROUND_DIR, min_valid_duration=MIN_VALID_DURATION):
    """Validation background files, as ``BaseAframeDataset.train_val_split``."""
    fnames = sorted(Path(f) for f in glob.glob(f"{background_dir}/background/*.hdf5"))
    durations = [int(f.stem.split("-")[-1]) for f in fnames]
    valid, total = [], 0
    while total < min_valid_duration:
        f, d = fnames.pop(-1), durations.pop(-1)
        total += d
        valid.append(str(f))
    return valid


def _waveform_slice():
    """Columns kept by ``slice_waveforms`` and the merger column within them."""
    with h5py.File(VAL_WAVEFORM_FILE, "r") as f:
        right_pad = float(f.attrs["right_pad"])
    with h5py.File(WAVEFORM_FILE, "r") as f:
        length = f["waveforms/cross"].shape[1]
        if float(f.attrs["right_pad"]) != right_pad:
            raise ValueError("training and validation waveform files differ in right_pad")
    signal_idx = length - int(right_pad * SAMPLE_RATE)
    start = signal_idx - (KERNEL_SIZE - RIGHT_PAD_SIZE)
    stop = signal_idx + (KERNEL_SIZE - LEFT_PAD_SIZE)
    if start < 0 or stop > length:
        raise ValueError("slice needs padding, not mirrored here")
    return start, stop, signal_idx - start


class _Transforms:
    """The pipeline's own modules, built as ``build_transforms`` builds them."""

    def __init__(self, device="cpu"):
        from ml4gw.transforms import Whiten
        from utils.preprocessing import PsdEstimator

        from train.augmentations import WaveformProjector

        self.device = torch.device(device)
        self.psd_estimator = PsdEstimator(
            KERNEL_LENGTH + FDURATION, SAMPLE_RATE, FFTLENGTH, window=None,
            fast=HIGHPASS is not None, average="median",
        ).to(self.device)
        self.whitener = Whiten(FDURATION, SAMPLE_RATE, HIGHPASS, LOWPASS).to(self.device)
        self.projector = WaveformProjector(IFOS, SAMPLE_RATE, HIGHPASS, LOWPASS).to(self.device)

    def psd_split(self, X):
        """``PsdEstimator.forward`` without its two-row special case."""
        size = self.psd_estimator.size
        background, X = torch.split(X, [X.size(-1) - size, size], dim=-1)
        return X, self.psd_estimator.spectral_density(background.double())


def _draws(n, seed, bg_sizes, n_waveforms, n_offsets):
    """All random choices for ``n`` windows, fixed by ``seed``."""
    rng = np.random.default_rng(seed)
    sizes = np.array(bg_sizes, dtype=np.int64)
    d = {}
    # file in proportion to its length, then an independent start per detector
    # (Hdf5TimeSeriesDataset, coincident=False: np.random.randint(size - kernel))
    d["bg_file"] = rng.choice(len(sizes), size=n, p=sizes / sizes.sum())
    max_idx = sizes[d["bg_file"]] - SAMPLE_SIZE
    d["bg_start"] = rng.integers(0, max_idx[:, None], size=(n, len(IFOS)))
    # SignalInverter(0.5), SignalReverser(0.5): per window and detector
    d["inverted"] = rng.random((n, len(IFOS))) < 0.5
    d["reversed"] = rng.random((n, len(IFOS))) < 0.5
    d["waveform_row"] = rng.choice(n_waveforms, size=n, replace=False)
    # sample_kernels: torch.randint(0, len(slice) - kernel_size)
    d["kernel_offset"] = rng.integers(0, n_offsets, size=n)
    from ml4gw.distributions import Cosine, PowerLaw

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        d["dec"] = Cosine().sample((n,)).numpy()
        d["psi"] = torch.distributions.Uniform(0.0, 3.14159, validate_args=False).sample((n,)).numpy()
        d["phi"] = torch.distributions.Uniform(-3.14159, 3.14159, validate_args=False).sample((n,)).numpy()
        d["snr"] = PowerLaw(minimum=SNR_MIN, maximum=SNR_MAX, index=SNR_INDEX).sample((n,)).numpy()
    return d


def windows(n, seed, batch_size=64, device="cpu", want=("X", "X_clean")):
    """Generate ``n`` validation-like windows in memory (use ``iter_windows``
    above a few thousand).

    Returns a dict with ``X`` (noisy, whitened) and/or ``X_clean`` (signal
    alone, whitened with the same PSD), each (n, 2, 8192) float32, and labels:
    ``chirp_mass`` (float32, from mass_1 and mass_2 as ``ChirpMass``), ``snr``
    (target network SNR), ``merger_sample`` (geocentric merger index in the
    whitened window, 7169-8192), sky angles, background file and starts,
    inversion and reversal flags, waveform row, kernel offset, and the file's
    parameters as ``file_<name>``.
    """
    out = {k: np.empty((n, len(IFOS), WINDOW_SIZE), np.float32) for k in want}
    labels = None
    for i, batch in iter_windows(n, seed, batch_size=batch_size, device=device, want=want):
        for k in want:
            out[k][i:i + len(batch[k])] = batch[k]
        labels = batch["labels"]
    out.update(labels)
    return out


def assemble(tr, X, wave, dec, psi, phi, snr, inverted, reversed_, offset, want=("X", "X_clean")):
    """One batch through the steps of ``SupervisedAframeDataset.inject`` and
    ``DenoisingTimeDomainSupervisedAframeDataset.inject`` with every row
    injected: X (b, 2, 51200) float32 raw background, wave (b, 2, 11264) float32
    sliced (cross, plus). Returns whitened (b, 2, 8192) tensors."""
    X, psds = tr.psd_split(X)
    X[inverted] *= -1
    X[reversed_] = X[reversed_].flip(-1)
    responses = tr.projector(dec, psi, phi, cross=wave[:, 0], plus=wave[:, 1])
    responses = tr.projector.rescaler(responses, psds, snr)
    idx = offset[:, None] + torch.arange(KERNEL_SIZE, device=X.device)[None]
    kernels = torch.gather(responses, 2, idx[:, None, :].expand(-1, responses.shape[1], -1))
    out = {}
    if "X" in want:
        Xn = X.clone()
        Xn += kernels
        out["X"] = tr.whitener(Xn, psds)
    if "X_clean" in want:
        clean = torch.zeros_like(X)
        clean[:] = kernels
        out["X_clean"] = tr.whitener(clean, psds)
    return out


def read_background(handles, files, starts):
    """(b, 2, 51200) float32 windows: per detector ``handles[file][ifo][start:start + 51200]``."""
    x = np.zeros((len(files), len(IFOS), SAMPLE_SIZE))
    for j, (fi, st) in enumerate(zip(files, starts)):
        for c, ifo in enumerate(IFOS):
            x[j, c] = handles[fi][ifo][st[c]:st[c] + SAMPLE_SIZE]
    return torch.Tensor(x)


def read_waveforms(wf, rows, col0, col1):
    """(b, 2, col1 - col0) float64 (cross, plus) for ``rows``, read in increasing order."""
    order = np.argsort(rows)
    wave = np.empty((len(rows), 2, col1 - col0))
    for c, pol in enumerate(("cross", "plus")):
        wave[order, c] = wf[f"waveforms/{pol}"][rows[order], col0:col1]
    return wave


def iter_windows(n, seed, batch_size=64, device="cpu", want=("X", "X_clean")):
    """Yield ``(start, batch)`` with ``batch[k]`` (b, 2, 8192) float32 numpy for
    ``k`` in ``want`` and ``batch["labels"]`` the labels of all ``n`` windows."""
    tr = _Transforms(device)
    bg_files = valid_fnames()
    bg_handles = [h5py.File(f, "r") for f in bg_files]
    bg_sizes = [len(h[IFOS[0]]) for h in bg_handles]
    wf = h5py.File(WAVEFORM_FILE, "r")
    try:
        n_wf = wf["waveforms/cross"].shape[0]
        col0, col1, merger_col = _waveform_slice()
        d = _draws(n, seed, bg_sizes, n_wf, (col1 - col0) - KERNEL_SIZE)
        d["merger_sample"] = merger_col - d["kernel_offset"] - FILTER_SIZE // 2
        file_params = {k: wf[f"parameters/{k}"][:] for k in wf["parameters"].keys()}
        rows = d["waveform_row"]
        # params go to float32 on device before ChirpMass, as in inject
        m1 = torch.tensor(file_params["mass_1"][rows]).float()
        m2 = torch.tensor(file_params["mass_2"][rows]).float()
        d["chirp_mass"] = ((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2).numpy()
        for k, v in file_params.items():
            d[f"file_{k}"] = v[rows]
        d["bg_files"] = np.array(bg_files)
        dev = tr.device

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a)).to(dev)

        for i0 in range(0, n, batch_size):
            sl = slice(i0, min(n, i0 + batch_size))
            X = read_background(bg_handles, d["bg_file"][sl], d["bg_start"][sl]).to(dev)
            wave = torch.tensor(read_waveforms(wf, rows[sl], col0, col1)).to(dev).float()
            out = assemble(tr, X, wave, t(d["dec"][sl]), t(d["psi"][sl]), t(d["phi"][sl]), t(d["snr"][sl]),
                           t(d["inverted"][sl]), t(d["reversed"][sl]), t(d["kernel_offset"][sl]), want)
            batch = {k: v.float().cpu().numpy() for k, v in out.items()}
            batch["labels"] = d
            yield sl.start, batch
    finally:
        for h in bg_handles:
            h.close()
        wf.close()


def clean_windows(n, seed, **kw):
    """Whitened signal alone, as the signal-only networks see it, plus labels."""
    w = windows(n, seed, want=("X_clean",), **kw)
    w["X"] = w.pop("X_clean")
    return w


def noisy_windows(n, seed, **kw):
    """Whitened background plus signal, as the noisy networks see it, plus
    labels; ``X_clean`` is the same signal alone (equal to ``clean_windows``
    with the same seed)."""
    return windows(n, seed, want=("X", "X_clean"), **kw)


DATA_DIR = HERE / "data"


def read_rows(path, start=0, stop=None):
    """Rows ``start:stop`` of a saved .npy window file, read without mmap
    (keeps resident memory at the size of the slice), as float32."""
    mm = np.load(path, mmap_mode="r")
    n = mm.shape[0]
    stop = n if stop is None else min(stop, n)
    row = int(np.prod(mm.shape[1:]))
    a = np.fromfile(path, dtype=mm.dtype, count=(stop - start) * row,
                    offset=mm.offset + start * row * mm.dtype.itemsize)
    return a.reshape((stop - start,) + mm.shape[1:]).astype(np.float32)


def load_set(name):
    """Labels of a saved set (``clean`` or ``noisy``) and the paths of its arrays."""
    labels = dict(np.load(DATA_DIR / f"{name}_labels.npz"))
    paths = {"X": DATA_DIR / ("clean_X.npy" if name == "clean" else "noisy_X.npy")}
    if name == "noisy":
        paths["X_clean"] = DATA_DIR / "noisy_Xclean.npy"
    return labels, paths


def score(est, true):
    """Fractions within 1, 2, 5, 10 percent relative error, as validation logs them."""
    est = np.asarray(est, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    err = np.abs(est - true) / np.abs(true)
    return {f"within_{t}pct": float((err < t / 100).mean()) for t in (1, 2, 5, 10)}


# ---------------------------------------------------------------- networks

def _config(name):
    for d in CONFIG_DIRS:
        p = d / f"{name}.yaml"
        if p.exists():
            return yaml.safe_load(open(p)), p
    raise FileNotFoundError(f"no config {name}.yaml in {CONFIG_DIRS}")


def _import(path):
    mod, cls = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod), cls)


def _find_ckpt(name, ckpt):
    """A path, a tag such as ``ck_69`` (copied as ckpt/<name>__ck_69.ckpt), or
    ``None`` for the only ``ck_*`` copy of ``name``."""
    if ckpt is not None and os.path.exists(str(ckpt)):
        return Path(ckpt)
    hits = sorted(CKPT_DIR.glob(f"{name}__{ckpt or 'ck_*'}.ckpt"))
    if len(hits) != 1:
        raise FileNotFoundError(f"need exactly one copied checkpoint for {name} ({ckpt}), found {hits}")
    return hits[0]


def _cache_s4d_kernel(kmod, L, chunk=16):
    """Evaluate ``ComplexS4DKernel.forward`` channel by channel and pin the
    result, so the (H, N/2, L) intermediate never exists in full."""
    from train.experimental.s4d_energy.arch import ComplexS4DKernel

    parts = []
    with torch.no_grad():
        H = kmod.log_dt.shape[0]
        for h0 in range(0, H, chunk):
            ns = types.SimpleNamespace(
                log_dt=kmod.log_dt[h0:h0 + chunk], C=kmod.C[h0:h0 + chunk],
                log_A_real=kmod.log_A_real[h0:h0 + chunk], A_imag=kmod.A_imag[h0:h0 + chunk],
            )
            parts.append(ComplexS4DKernel.forward(ns, L))
    cached = torch.cat(parts, 0)

    def forward(length):
        if length != cached.shape[-1]:
            raise ValueError(f"kernel cached for L={cached.shape[-1]}, asked {length}")
        return cached

    kmod.forward = forward
    return cached


class Net:
    """A trained network as a chirp-mass estimator.

    ``net(X)`` takes (n, 2, 8192) whitened windows (numpy or torch, any float
    dtype) and returns chirp-mass estimates (n,) float64, converted exactly as
    the task's validation step does: oracle regressors give
    ``mean * y_std + y_mean``; long_conv classifiers give ``exp`` of the
    parabola-refined logits peak (``grid_estimate``). ``net.raw(X)`` returns
    the network outputs; ``net.arch`` is the module itself.

    long_conv networks run in batches of at least 32 windows, shorter ones
    padded with copies of their own rows, as in training and validation. At 64
    or more transforms, cuFFT's inverse real FFT in the decimation of
    ``LongConvEnergyNet.peak_energy`` adds -Im(X[512 Hz]) / 4096 to every
    sample (one DC offset per window and detector); below that it does not.
    ``match_training_fft=False`` skips the padding (exact arithmetic for
    batches under 32).
    """

    def __init__(self, name, arch, task, y_mean, y_std, normalize_input, device, batch_size, ckpt,
                 match_training_fft=True):
        self.name, self.arch, self.task = name, arch, task
        self.match_training_fft = match_training_fft
        self.y_mean, self.y_std = y_mean, y_std
        self.normalize_input = normalize_input
        self.device, self.batch_size, self.ckpt = device, batch_size, ckpt

    @torch.no_grad()
    def raw(self, X, batch_size=None):
        bs = batch_size or self.batch_size
        if self.task == "long_conv" and self.match_training_fft:
            bs = max(bs, VALIDATION_BATCH)
        outs = []
        for i in range(0, len(X), bs):
            x = torch.as_tensor(np.asarray(X[i:i + bs]) if not torch.is_tensor(X) else X[i:i + bs])
            x = x.to(self.device, torch.float32)
            m = len(x)
            if self.task == "long_conv" and self.match_training_fft and m < VALIDATION_BATCH:
                # pad with copies of its own rows: same rows, same cuFFT path as a 32-row batch
                x = torch.cat([x] + [x] * (VALIDATION_BATCH // m))[:VALIDATION_BATCH]
            if self.normalize_input:
                x = x / x.std(dim=-1, keepdim=True).clamp(min=1e-8)
            outs.append(self.arch(x)[:m].float().cpu())
        return torch.cat(outs)

    @torch.no_grad()
    def __call__(self, X, batch_size=None):
        out = self.raw(X, batch_size)
        if self.task == "oracle":
            y_mean = torch.tensor(self.y_mean, dtype=torch.float32)
            y_std = torch.tensor(self.y_std, dtype=torch.float32)
            return (out[:, 0] * y_std + y_mean).double().numpy()
        log_mc, _ = self.arch.estimate(out.to(self.device))
        return log_mc.exp().double().cpu().numpy()


def load_net(name, ckpt=None, device=None, batch_size=None, gpu_gb=3.0, matmul_precision="high",
             match_training_fft=True):
    """Build ``name``'s architecture from its yaml and load a copied checkpoint
    (``ckpt``: a path, a tag such as ``ck_69`` or ``last``, or ``None`` when
    only one ``ck_*`` copy exists). ``matmul_precision`` is set globally, as
    train/cli.py does ("high": TF32 matmuls); ``None`` leaves it alone.
    ``gpu_gb`` caps this process's CUDA allocator."""
    if matmul_precision is not None:
        torch.set_float32_matmul_precision(matmul_precision)
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, gpu_gb * 2**30 / total), device)
    cfg, _ = _config(name)
    model_cfg = cfg["model"]
    arch_cfg = model_cfg["init_args"]["arch"]
    arch = _import(arch_cfg["class_path"])(**arch_cfg.get("init_args", {}))
    path = _find_ckpt(name, ckpt)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    arch_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
    missing, unexpected = arch.load_state_dict(arch_sd, strict=False)
    # buffers rebuilt at init and not saved (persistent=False) are the only allowed misses
    persistent = set(arch.state_dict().keys())
    missing = [k for k in missing if k in persistent]
    if missing or unexpected:
        raise RuntimeError(f"{name}: missing {missing}, unexpected {unexpected}")
    hp = ck.get("hyper_parameters", {})
    normalize_input = bool(hp.get("normalize_input", model_cfg["init_args"].get("normalize_input", False)))
    task_cls = model_cfg["class_path"]
    if task_cls.endswith("oracle.task.SignalOnlyRegression"):
        task = "oracle"
        y_mean, y_std = float(sd["y_mean"][0]), float(sd["y_std"][0])
    elif task_cls.endswith("long_conv.task.LongConvClassifier"):
        task = "long_conv"
        y_mean = y_std = None
    else:
        raise ValueError(f"unsupported task {task_cls}")
    arch = arch.to(device).eval()
    # pin S4D kernels of the long-convolution networks (2 GB intermediate at d_state 512)
    if hasattr(arch, "kernel_type") and arch.kernel_type == "s4d":
        _cache_s4d_kernel(arch.kernel, arch.length)
    if batch_size is None:
        batch_size = 8 if "FreeKernel" in arch_cfg["class_path"] else (VALIDATION_BATCH if task == "long_conv" else 64)
    net = Net(name, arch, task, y_mean, y_std, normalize_input, device, batch_size, str(path),
              match_training_fft=match_training_fft)
    net.epoch = ck.get("epoch")
    return net
