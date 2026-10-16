"""Tone count one S4D channel needs to reproduce a whitened BNS matched-filter kernel."""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "16"

import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import h5py  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as sla  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(16)

WF_PATH = Path(
    "/home/kyoon/SSM-BNS/DATA/aframe/train/"
    "end_o3_ratesandpops_bns_uniform_chirp.hdf5"
)
PSD_PATH = Path(
    "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/"
    "train_windows_snr8_powerlaw.h5"
)
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
THRESHOLDS = [0.90, 0.95, 0.97, 0.99]
P_PENCIL = L // 2
METHODS = ["A_best_fourier", "B_fixed_grid", "C_matrix_pencil"]
TITLES = {
    "A_best_fourier": "A: best-N Fourier (undamped, 1/T grid)",
    "B_fixed_grid": "B: fixed grid 20-800 Hz, a=-1/(L dt)",
    "C_matrix_pencil": "C: matrix pencil (free damped modes)",
}


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
    """Least-squares c for K(l) = sum c_n exp(l log z_n); match and rank."""
    lags = np.arange(L, dtype=np.float64)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        V = np.exp(lags[:, None] * logz[None, :])
    c, _, rank, _ = np.linalg.lstsq(V, K, rcond=None)
    Khat = V @ c
    match = np.abs(np.vdot(Khat, K)) / (np.linalg.norm(Khat) * np.linalg.norm(K))
    return float(match), int(rank)


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


def pencil_poles(Vh, N):
    """Matrix-pencil poles from the top-N right singular vectors."""
    W = Vh[:N].T
    Phi = np.linalg.lstsq(W[:-1], W[1:], rcond=None)[0]
    z = np.linalg.eigvals(Phi)
    outside = np.abs(z) > 1
    z[outside] = 1.0 / np.conj(z[outside])
    return z, int(outside.sum())


def safe_log(z):
    return np.log(np.clip(np.abs(z), 1e-300, None)) + 1j * np.angle(z)


def first_reaching(matches, th):
    for N in NS:
        if matches[N] >= th:
            return N
    return f">{NS[-1]}"


def make_figure(match, path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    fig.patch.set_facecolor("white")
    for ax, meth in zip(axes, METHODS):
        ax.set_facecolor("white")
        for Mc in MCS:
            ax.plot(NS, [match[meth][Mc][N] for N in NS], marker="o", label=f"Mc={Mc}")
        ax.set_xscale("log", base=2)
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.minorticks_off()
        ax.set_ylim(0, 1)
        ax.set_xlabel("N tones")
        ax.set_title(TITLES[meth], fontsize=10)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("match")
    axes[0].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor="white")
    plt.close(fig)


def main():
    t_start = time.time()
    f_psd, S = load_psd()
    templates = select_templates()
    pols = load_polarizations([t["row"] for t in templates.values()])

    match = {m: {Mc: {} for Mc in MCS} for m in METHODS}
    rank = {m: {Mc: {} for Mc in MCS} for m in METHODS[1:]}
    wall = {m: {Mc: {} for Mc in MCS} for m in METHODS}
    reflected = {Mc: {} for Mc in MCS}
    hankel = {}

    for Mc in MCS:
        info = templates[Mc]
        plus, cross = pols[info["row"]]
        m_idx, hw = whitened_window(plus, cross, f_psd, S)
        K = kernel_from(hw)
        E, spec = spectrum_stats(K)
        lit, dom = cycles(hw, spec["grid_sign"])
        info.update(spec)
        info["merger_index"] = m_idx
        info["cycles_window_unwrapped_hw"] = abs(lit)
        info["cycles_window_dominant_sign_part"] = abs(dom)
        print(f"Mc={Mc} row={info['row']} theta_jn={info['theta_jn']:.3f} "
              f"pos={spec['frac_energy_pos_freq']:.3f} "
              f"neg={spec['frac_energy_neg_freq']:.3f} "
              f"band={spec['frac_energy_in_grid_band']:.3f} "
              f"cycles={abs(lit):.1f}/{abs(dom):.1f}", flush=True)

        t = time.time()
        cum = np.cumsum(np.sort(E)[::-1]) / E.sum()
        for N in NS:
            match["A_best_fourier"][Mc][N] = float(np.sqrt(cum[N - 1]))
        wall["A_best_fourier"][Mc]["all_N"] = time.time() - t

        for N in NS:
            t = time.time()
            mt, rk = lstsq_match(grid_logz(N, spec["grid_sign"]), K)
            match["B_fixed_grid"][Mc][N] = mt
            rank["B_fixed_grid"][Mc][N] = rk
            wall["B_fixed_grid"][Mc][N] = time.time() - t
        print(f"  B done {[round(match['B_fixed_grid'][Mc][N], 4) for N in NS]}",
              flush=True)

        t = time.time()
        shape, s, Vh = hankel_svd(K)
        t_svd = time.time() - t
        e = np.cumsum(s**2) / np.sum(s**2)
        hankel[Mc] = {
            "shape": list(shape),
            "svd_wall_s": t_svd,
            "n_sv_for_energy": {
                f"{th:.2f}": int(np.searchsorted(e, th) + 1) for th in THRESHOLDS
            },
            "energy_frac_top_N_sv": {N: float(e[N - 1]) for N in NS},
            "sv_over_sv0_at_N": {N: float(s[N - 1] / s[0]) for N in NS},
        }
        print(f"  SVD {shape} {t_svd:.1f}s n_sv {hankel[Mc]['n_sv_for_energy']}",
              flush=True)
        for N in NS:
            t = time.time()
            z, n_ref = pencil_poles(Vh, N)
            mt, rk = lstsq_match(safe_log(z), K)
            match["C_matrix_pencil"][Mc][N] = mt
            rank["C_matrix_pencil"][Mc][N] = rk
            reflected[Mc][N] = n_ref
            wall["C_matrix_pencil"][Mc][N] = time.time() - t
        wall["C_matrix_pencil"][Mc]["svd"] = t_svd
        print(f"  C done {[round(match['C_matrix_pencil'][Mc][N], 4) for N in NS]}",
              flush=True)
        del Vh

    min_n = {
        m: {Mc: {f"{th:.2f}": first_reaching(match[m][Mc], th) for th in THRESHOLDS}
            for Mc in MCS}
        for m in METHODS
    }
    cost = {N: {"real_params": 4 * N, "complex_mac_per_kernel": N * L} for N in NS}

    make_figure(match, OUT_DIR / "match_vs_tones.png")

    def keys_to_str(d):
        if isinstance(d, dict):
            return {str(k): keys_to_str(v) for k, v in d.items()}
        return d

    results = {
        "config": {
            "waveforms": str(WF_PATH),
            "psd_file": str(PSD_PATH),
            "psd_rows_averaged": N_PSD_ROWS,
            "psd_detectors_averaged": 2,
            "sample_rate": FS,
            "window_samples": WIN,
            "post_merger_samples": POST,
            "kernel_length_L": L,
            "early_taper_samples": TAPER,
            "f_low_hz": F_LOW,
            "grid_band_hz": list(GRID_BAND),
            "grid_damping_a_real": -1.0 / (L * DT),
            "pencil_P": P_PENCIL,
            "N_values": NS,
            "match_thresholds": THRESHOLDS,
            "mass_frame_note": "parameters/mass_1, mass_2 are detector frame",
        },
        "templates": keys_to_str(templates),
        "match": keys_to_str(match),
        "min_N_for_match": keys_to_str(min_n),
        "hankel": keys_to_str(hankel),
        "lstsq_rank": keys_to_str(rank),
        "pencil_reflected_poles": keys_to_str(reflected),
        "cost_per_channel": keys_to_str(cost),
        "wall_time_s": keys_to_str(wall),
        "total_wall_s": time.time() - t_start,
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=1)

    print("\nmatch table (rows method/Mc, cols N)")
    print("method            Mc   " + " ".join(f"{N:>7d}" for N in NS))
    for m in METHODS:
        for Mc in MCS:
            row = " ".join(f"{match[m][Mc][N]:7.4f}" for N in NS)
            print(f"{m:17s} {Mc:4.1f} {row}")
    print("\nmin N for match >= 0.90/0.95/0.97/0.99")
    for m in METHODS:
        for Mc in MCS:
            print(f"{m:17s} {Mc:4.1f} {list(min_n[m][Mc].values())}")
    print(f"\ntotal wall {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
