"""Shared helpers for the resonator-bank derivation checks.

Units: seconds, hertz, angular frequency in rad/s. A resonator n is the
complex mode exp((-a_n + i w_n) t), t >= 0; sampled at FS it is z_n^l with
z_n = exp((-a_n + i w_n) / FS).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

GM_SUN = 1.3271244e20          # m^3 s^-2, IAU 2015 nominal
C_LIGHT = 299792458.0          # m/s
TSUN = GM_SUN / C_LIGHT ** 3   # s
FS = 2048.0
L0 = 8192                      # 4 s at 2048 Hz
MASSES = (0.85, 1.2, 1.8, 2.75)

PSD_H5 = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/"
          "train_windows_snr8_powerlaw.h5")
WF_H5 = ("/home/kyoon/SSM-BNS/DATA/aframe/train/"
         "end_o3_ratesandpops_bns_uniform_chirp.hdf5")
PSD_NPZ = os.path.join(RES, "o3_avg_psd_2000rows.npz")


# ------------------------------------------------------------------ chirp
def tau_of_f(f, mc):
    """Newtonian time before merger at GW frequency f."""
    return 5.0 / 256.0 * (TSUN * mc) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def f_of_tau(tau, mc):
    """Newtonian GW frequency a time tau before merger."""
    return (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** 0.375 * (TSUN * mc) ** (-0.625)


def phi_of_tau(tau, mc):
    """Newtonian GW phase, zero at merger."""
    return -2.0 * (tau / (5.0 * TSUN * mc)) ** 0.625


def fdot(f, mc):
    return 96.0 / 5.0 * np.pi ** (8.0 / 3.0) * (TSUN * mc) ** (5.0 / 3.0) * f ** (11.0 / 3.0)


def int_tau_df(f1, f2, mc):
    """Closed form of int_{f1}^{f2} tau(f) df = (3/5)(tau1 f1 - tau2 f2)."""
    return 0.6 * (tau_of_f(f1, mc) * f1 - tau_of_f(f2, mc) * f2)


def cycles(f1, f2, mc):
    """Closed form of the cycles from f1 to f2 = (8/5)(tau1 f1 - tau2 f2)."""
    return 1.6 * (tau_of_f(f1, mc) * f1 - tau_of_f(f2, mc) * f2)


def toy_chirp(mc, fmin=20.0, fmax=900.0, n=L0, fs=FS):
    """The 7 October toy chirp: tau^(-1/4) exp(i Phi), 4 s ending at merger,
    kept where fmin < f < fmax (same as qa/warp/warp_tones.py cchirp)."""
    tau = (n - np.arange(n)) / fs
    fr = f_of_tau(tau, mc)
    return tau ** -0.25 * (fr < fmax) * (fr > fmin) * np.exp(1j * phi_of_tau(tau, mc))


def kernel_of(h):
    """Matched-filter kernel K(l) = conj(h(end - l)), unit norm."""
    K = np.conj(h[::-1]).astype(np.complex128)
    return K / np.linalg.norm(K)


def group_delay(K, fs=FS, pad=1):
    """Group delay -d(arg K~)/d(omega) in seconds at the rfft-like grid of
    the zero-padded DFT, via Re[DFT(l K) / DFT(K)]."""
    n = K.size * pad
    l = np.arange(K.size)
    F = np.fft.fft(K, n)
    G = np.fft.fft(l * K, n)
    f = np.fft.fftfreq(n, 1.0 / fs)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = np.real(G / F) / fs
    return f, tg, np.abs(F) ** 2


# ------------------------------------------------------------------ noise
def o3_psd():
    """Arithmetic mean of the first 2000 training-window PSDs over both
    detectors (the whitening of tone_check.py and warp_tones.py)."""
    if os.path.exists(PSD_NPZ):
        z = np.load(PSD_NPZ)
        return z["f"], z["S"]
    import h5py
    with h5py.File(PSD_H5, "r") as f:
        S = f["psds"][:2000].mean(axis=(0, 1))
    fg = np.arange(S.size) * 0.5
    np.savez(PSD_NPZ, f=fg, S=S)
    return fg, S


# ------------------------------------------------------------------ resonator fits
def logz(a, f, fs=FS):
    """log z for decay rate a (1/s) and frequency f (Hz)."""
    return (-np.asarray(a, float) + 2j * np.pi * np.asarray(f, float)) / fs


def gram(lz, L):
    """G_nm = sum_{l<L} conj(z_n)^l z_m^l in closed form."""
    s = np.conj(lz)[:, None] + lz[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        G = np.expm1(L * s) / np.expm1(s)
    small = np.abs(s) < 1e-13
    G[small] = L
    return G


def proj_b(lz, K, chunk=256):
    """b_n = sum_l conj(z_n)^l K(l)."""
    l = np.arange(K.size, dtype=np.float64)
    b = np.empty(lz.size, complex)
    for s in range(0, lz.size, chunk):
        V = np.exp(l[:, None] * lz[None, s:s + chunk])
        b[s:s + chunk] = V.conj().T @ K
    return b


def fit_match(lz, K, rtol=1e-13, want_coef=False):
    """Least-squares fit of K by sum_n c_n z_n^l: returns the match
    ||P K|| / ||K|| (P = projection on the span), via the Gram matrix
    pseudo-inverse with eigenvalue cutoff rtol * max."""
    G = gram(lz, K.size)
    G = 0.5 * (G + G.conj().T)
    b = proj_b(lz, K)
    w, U = np.linalg.eigh(G)
    keep = w > rtol * w.max()
    ub = U[:, keep].conj().T @ b
    m2 = float(np.sum(np.abs(ub) ** 2 / w[keep]).real) / float(np.vdot(K, K).real)
    out = {"match": float(np.sqrt(max(m2, 0.0))), "rank": int(keep.sum()), "n": int(lz.size)}
    if want_coef:
        c = U[:, keep] @ (ub / w[keep])
        out["coef"] = c
        # weight inflation: sum |c_n|^2 ||v_n||^2 / ||V c||^2
        vn2 = np.real(np.diag(G))
        Vc2 = float(np.real(np.vdot(c, G @ c)))
        out["inflation"] = float(np.sum(np.abs(c) ** 2 * vn2) / Vc2)
    return out


def fit_match_direct(lz, K):
    """Same match by numpy lstsq on the explicit matrix (check only)."""
    l = np.arange(K.size, dtype=np.float64)
    V = np.exp(l[:, None] * lz[None, :])
    c, *_ = np.linalg.lstsq(V, K, rcond=None)
    Kh = V @ c
    return float(np.abs(np.vdot(Kh, K)) / (np.linalg.norm(Kh) * np.linalg.norm(K)))


def fit_tradeoff(lz, K, caps=(np.inf, 100.0, 10.0, 2.0), rtol=1e-13, want_coef=False):
    """Least-squares fit of K by sum_n c_n z_n^l, with optional ridge on the
    column-normalized weights so that the weight inflation
        R = sum_n |c_n|^2 |v_n|^2 / |sum_n c_n v_n|^2
    (1 for orthogonal columns, >> 1 when terms cancel) stays <= cap.
    Returns, per cap, the match |<V c, K>| / (|V c| |K|), R, and lambda."""
    K = K / np.linalg.norm(K)
    G = gram(lz, K.size)
    G = 0.5 * (G + G.conj().T)
    dn = np.sqrt(np.real(np.diag(G)))
    Gn = G / dn[:, None] / dn[None, :]
    bn = proj_b(lz, K) / dn
    lam_, U = np.linalg.eigh(Gn)
    keep = lam_ > rtol * lam_.max()
    lam_, U = lam_[keep], U[:, keep]
    beta2 = np.abs(U.conj().T @ bn) ** 2

    def stats(lmb):
        num = np.sum(beta2 / (lam_ + lmb))
        den = np.sum(lam_ * beta2 / (lam_ + lmb) ** 2)
        nrm = np.sum(beta2 / (lam_ + lmb) ** 2)
        return float(num / np.sqrt(den)), float(nrm / den)

    res = {"n": int(lz.size), "rank": int(keep.sum())}
    coefs = {}
    for cap in caps:
        m0, R0 = stats(0.0)
        if R0 <= cap:
            lmb = 0.0
        else:
            lo, hi = -16.0, 4.0                      # log10(lambda / lam_max)
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if stats(10 ** mid * lam_.max())[1] > cap:
                    lo = mid
                else:
                    hi = mid
            lmb = 10 ** hi * lam_.max()
        m, R = stats(lmb)
        key = "inf" if np.isinf(cap) else f"{cap:g}"
        res[key] = {"match": m, "inflation": R, "lambda_over_max": float(lmb / lam_.max())}
        if want_coef:
            cn = U @ ((U.conj().T @ bn) / (lam_ + lmb))
            coefs[key] = cn / dn
    if want_coef:
        res["coef"] = coefs
    return res


def match_float32(lz, coef, K):
    """Rebuild sum_n c_n z_n^l in complex64 (as ml4gw builds its kernels:
    exp of (log z) times the lag, then a weighted sum) and return its
    match with K."""
    import torch
    l32 = torch.arange(K.size, dtype=torch.float32)
    lz32 = torch.tensor(lz, dtype=torch.complex64)
    c32 = torch.tensor(coef, dtype=torch.complex64)
    out = torch.zeros(K.size, dtype=torch.complex64)
    for s in range(0, lz.size, 256):
        E = torch.exp(lz32[s:s + 256, None] * l32[None, :])
        out += (c32[s:s + 256, None] * E).sum(0)
    Kh = out.numpy().astype(np.complex128)
    nk = np.linalg.norm(Kh)
    if not np.isfinite(nk) or nk == 0:
        return float("nan")
    return float(np.abs(np.vdot(Kh, K)) / (nk * np.linalg.norm(K)))


def place(fnodes, density, n_total=None):
    """Frequencies at the half-integer quantiles of the cumulative count
    int density df on the node grid (density in resonators per Hz)."""
    df = np.gradient(fnodes)
    cum = np.cumsum(density * df)
    tot = cum[-1]
    N = int(round(tot)) if n_total is None else int(n_total)
    if N < 1:
        return np.zeros(0)
    targets = (np.arange(N) + 0.5) * tot / N
    return np.interp(targets, cum, fnodes)
