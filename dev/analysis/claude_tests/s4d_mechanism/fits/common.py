"""Kernel construction and least-squares match, copied from tone_check.py (same constants)."""

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402

WF_PATH = Path(
    "/home/kyoon/SSM-BNS/DATA/aframe/train/"
    "end_o3_ratesandpops_bns_uniform_chirp.hdf5"
)
PSD_PATH = Path(
    "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/"
    "train_windows_snr8_powerlaw.h5"
)
TONE_DIR = Path("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check")
OUT_DIR = Path(__file__).resolve().parent

FS = 2048.0
DT = 1.0 / FS
WIN = 8192
POST = 128
L = WIN - POST
TAPER = 512
F_LOW = 20.0
GRID_BAND = (20.0, 800.0)
N_PSD_ROWS = 2000
PSD_DF = 0.5
MCS = [1.0, 1.2, 1.5, 2.0]
NS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
P_FIT = [8, 16, 32, 64, 128, 256, 512, 1024]
P_PENCIL = L // 2


def load_psd():
    """Mean one-sided PSD over the first N_PSD_ROWS rows and both detectors."""
    with h5py.File(PSD_PATH, "r") as f:
        psds = f["psds"][:N_PSD_ROWS]
    S = psds.mean(axis=(0, 1))
    return np.arange(S.size) * PSD_DF, S


def select_templates():
    """One row per target Mc: Mc match, q in [0.85, 0.95], min a_1 + a_2."""
    with h5py.File(WF_PATH, "r") as f:
        par = {k: f["parameters"][k][:] for k in f["parameters"]}
    m1, m2 = par["mass_1"], par["mass_2"]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    out = {}
    for Mc in MCS:
        cand = np.flatnonzero(
            (np.abs(np.log(mc / Mc)) < 0.002) & (q >= 0.85) & (q <= 0.95)
        )
        if cand.size == 0:
            raise RuntimeError(f"no waveform row for Mc={Mc}")
        i = int(cand[np.argmin(par["a_1"][cand] + par["a_2"][cand])])
        info = {
            "row": i,
            "n_candidates": int(cand.size),
            "chirp_mass": float(mc[i]),
            "mass_ratio": float(q[i]),
        }
        info.update({k: float(v[i]) for k, v in par.items()})
        out[Mc] = info
    return out


def load_polarizations(rows):
    """Plus and cross for the given rows, keyed by row."""
    idx = sorted(rows)
    with h5py.File(WF_PATH, "r") as f:
        plus = f["waveforms/plus"][idx]
        cross = f["waveforms/cross"][idx]
    return {r: (plus[j], cross[j]) for j, r in enumerate(idx)}


def whitened_window(plus, cross, f_psd, S):
    """Whitened complex window with early-end half-Hann taper; merger at L-1."""
    h = plus - 1j * cross
    m = int(np.argmax(plus**2 + cross**2))
    start, stop = m - (L - 1), m + POST + 1
    if start < 0 or stop > h.size:
        raise RuntimeError(f"window [{start}, {stop}) out of range")
    seg = h[start:stop]
    af = np.abs(np.fft.fftfreq(WIN, DT))
    keep = af >= F_LOW
    H = np.fft.fft(seg)
    Hw = np.zeros_like(H)
    Hw[keep] = H[keep] / np.sqrt(np.interp(af[keep], f_psd, S))
    hw = np.fft.ifft(Hw)
    taper = np.ones(WIN)
    taper[:TAPER] = 0.5 * (1.0 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    return m, hw * taper


def kernel_from(hw):
    """K(l) = conj(h_w[m - l]), l = 0..L-1, unit L2 norm."""
    K = np.conj(hw[L - 1 :: -1])
    return K / np.linalg.norm(K)


def spectrum_stats(K):
    """DFT energy of K and its split by frequency sign and grid band."""
    E = np.abs(np.fft.fft(K)) ** 2
    f = np.fft.fftfreq(L, DT)
    tot = E.sum()
    pos = float(E[f > 0].sum() / tot)
    neg = float(E[f < 0].sum() / tot)
    sign = 1 if pos >= neg else -1
    sf = sign * f
    band = float(E[(sf >= GRID_BAND[0]) & (sf <= GRID_BAND[1])].sum() / tot)
    stats = {
        "frac_energy_pos_freq": pos,
        "frac_energy_neg_freq": neg,
        "grid_sign": sign,
        "frac_energy_in_grid_band": band,
    }
    return E, stats


def cycles(hw, sign):
    """Unwrapped phase change / 2pi of h_w and of its dominant-sign part."""
    ph = np.unwrap(np.angle(hw))
    literal = (ph[-1] - ph[0]) / (2 * np.pi)
    F = np.fft.fft(hw)
    F[sign * np.fft.fftfreq(WIN, DT) <= 0] = 0
    pa = np.unwrap(np.angle(np.fft.ifft(F)))
    dominant = (pa[-1] - pa[0]) / (2 * np.pi)
    return float(literal), float(dominant)


def lstsq_match(logz, K):
    """Least-squares c for K(l) = sum c_n exp(l log z_n); match and rank (tone_check)."""
    lags = np.arange(L, dtype=np.float64)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        V = np.exp(lags[:, None] * logz[None, :])
    c, _, rank, _ = np.linalg.lstsq(V, K, rcond=None)
    Khat = V @ c
    match = np.abs(np.vdot(Khat, K)) / (np.linalg.norm(Khat) * np.linalg.norm(K))
    return float(match), int(rank)


def lstsq_full(logz, K):
    """Least-squares fit with diagnostics: match, rank, c, cancellation ratio."""
    lags = np.arange(L, dtype=np.float64)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        V = np.exp(lags[:, None] * logz[None, :])
    c, _, rank, sv = np.linalg.lstsq(V, K, rcond=None)
    Khat = V @ c
    match = np.abs(np.vdot(Khat, K)) / (np.linalg.norm(Khat) * np.linalg.norm(K))
    col = np.linalg.norm(V, axis=0)
    cancel = float(np.sum((np.abs(c) * col) ** 2) / np.linalg.norm(Khat) ** 2)
    return {
        "match": float(match),
        "rank": int(rank),
        "c": c,
        "cancel_ratio": cancel,
        "cond_V": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf"),
        "col_norm": col,
    }


def grid_logz(N, sign):
    """log z for N tones on linspace(GRID_BAND), damping one e-fold over L."""
    f = np.linspace(GRID_BAND[0], GRID_BAND[1], N)
    return -1.0 / L + 1j * sign * 2 * np.pi * f * DT


def hankel_svd(K):
    """Singular values and right singular vectors of Y[i, j] = K[i + j]."""
    Y = np.ascontiguousarray(
        np.lib.stride_tricks.sliding_window_view(K, P_PENCIL + 1)
    )
    try:
        _, s, Vh = sla.svd(Y, full_matrices=False, lapack_driver="gesdd")
    except np.linalg.LinAlgError:
        _, s, Vh = sla.svd(Y, full_matrices=False, lapack_driver="gesvd")
    return Y.shape, s, Vh


def pencil_raw(Vh, N):
    """Matrix-pencil poles from the top-N right singular vectors, no reflection."""
    W = Vh[:N].T
    Phi = np.linalg.lstsq(W[:-1], W[1:], rcond=None)[0]
    return np.linalg.eigvals(Phi)


def reflect(z):
    """tone_check rule: |z| > 1 mapped to 1/conj(z)."""
    z = z.copy()
    out = np.abs(z) > 1
    z[out] = 1.0 / np.conj(z[out])
    return z, int(out.sum())


def clip(z):
    """Growing poles put on the unit circle (decay rate 0), frequency kept."""
    z = z.copy()
    out = np.abs(z) > 1
    z[out] = z[out] / np.abs(z[out])
    return z, int(out.sum())


def safe_log(z):
    return np.log(np.clip(np.abs(z), 1e-300, None)) + 1j * np.angle(z)


def build_kernels():
    """Kernels and template info for all MCS, exactly as tone_check."""
    f_psd, S = load_psd()
    templates = select_templates()
    pols = load_polarizations([t["row"] for t in templates.values()])
    out = {}
    for Mc in MCS:
        info = dict(templates[Mc])
        plus, cross = pols[info["row"]]
        m_idx, hw = whitened_window(plus, cross, f_psd, S)
        K = kernel_from(hw)
        E, spec = spectrum_stats(K)
        lit, dom = cycles(hw, spec["grid_sign"])
        info.update(spec)
        info["merger_index"] = m_idx
        info["cycles_window_unwrapped_hw"] = abs(lit)
        info["cycles_window_dominant_sign_part"] = abs(dom)
        out[Mc] = {"K": K, "hw": hw, "E": E, "info": info}
    return out
