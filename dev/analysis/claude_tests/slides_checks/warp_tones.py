"""Fourier tone counts before and after the 0PN warp u = (t_c - t)^(5/8).

Part A: the 7 Oct toy chirps (ckcheck/checks.py setup).
Part B: the whitened real templates of tone_check/tone_check.py.
"""
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "4"

import h5py
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
FS, T = 2048, 4.0
L0 = int(FS * T)
GM_SUN = 1.3271244e20
C_LIGHT = 299792458.0
TSUN = GM_SUN / C_LIGHT**3
TH = (0.9, 0.97, 0.99)
res = {}


def f_of_tau(tau, mc):
    return (1 / (8 * np.pi)) * (5 / tau) ** (3 / 8) * (TSUN * mc) ** (-5 / 8)


def tau_of_f(f, mc):
    return 5 * (8 * np.pi * f) ** (-8 / 3) * (TSUN * mc) ** (-5 / 3)


def phi_of_tau(tau, mc):
    return -2 * (tau / (5 * TSUN * mc)) ** (5 / 8)


def curve_of(x):
    p = np.sort(np.abs(np.fft.fft(x)) ** 2)[::-1]
    return p, np.sqrt(np.cumsum(p) / p.sum())


def n_for(c, th):
    return int(np.searchsorted(c, th)) + 1


def counts(x):
    _, c = curve_of(x)
    return {f"{t}": n_for(c, t) for t in TH} | {"match_at_32": float(c[31])}


def best_k(x, k):
    """Reconstruction from the k strongest DFT coefficients."""
    X = np.fft.fft(x)
    idx = np.argsort(np.abs(X))[::-1][:k]
    Y = np.zeros_like(X)
    Y[idx] = X[idx]
    return np.fft.ifft(Y)


def wmatch(a, b, w):
    num = np.abs(np.sum(w * np.conj(a) * b))
    return num / np.sqrt(np.sum(w * np.abs(a) ** 2) * np.sum(w * np.abs(b) ** 2))


def k_for_wmatch(x, w, th, norm_full=None):
    """Smallest k (strongest-first in u) whose reconstruction reaches match th
    in the t measure (weights w); norm_full rescales to the full-window norm."""
    X = np.fft.fft(x)
    order = np.argsort(np.abs(X))[::-1]

    def m(k):
        Y = np.zeros_like(X)
        Y[order[:k]] = X[order[:k]]
        y = np.fft.ifft(Y)
        val = wmatch(y, x, w)
        if norm_full is not None:
            val *= np.sqrt(np.sum(w * np.abs(x) ** 2)) / norm_full
        return val

    if m(x.size) < th:
        return None
    lo, hi = 1, x.size
    while lo < hi:
        mid = (lo + hi) // 2
        if m(mid) >= th:
            hi = mid
        else:
            lo = mid + 1
    return lo


def trig_interp_exact(x, tpos, chunk=256):
    """Band-limited (periodic trigonometric) value of samples x at fractional
    sample positions tpos; direct sum (slow, used as a check)."""
    N = x.size
    X = np.fft.fft(x) / N
    k = np.fft.fftfreq(N, 1.0 / N)
    if N % 2 == 0:  # split the Nyquist bin symmetrically
        X = np.append(X, X[N // 2] / 2)
        X[N // 2] /= 2
        k = np.append(k, N // 2)
        k[N // 2] = -N // 2
    out = np.empty(tpos.size, complex)
    for s in range(0, tpos.size, chunk):
        ph = np.exp(2j * np.pi * np.outer(tpos[s:s + chunk], k) / N)
        out[s:s + chunk] = ph @ X
    return out


UP = 32


def trig_interp(x, tpos):
    """Band-limited value at fractional positions: periodic FFT upsampling by
    UP, then a cubic spline on the upsampled samples."""
    from scipy.interpolate import CubicSpline
    N = x.size
    X = np.fft.fft(x)
    Y = np.zeros(N * UP, complex)
    Y[:N // 2] = X[:N // 2]
    Y[-N // 2 + 1:] = X[-N // 2 + 1:]
    Y[N // 2] = X[N // 2] / 2
    Y[-N // 2] = X[N // 2] / 2
    y = np.fft.ifft(Y) * UP
    grid = np.arange(N * UP + 1) / UP
    y = np.append(y, y[0])
    cs = CubicSpline(grid, y, bc_type="periodic")
    return cs(np.mod(tpos, N))


def inst_freq(z, block):
    """Cycles per sample from a linear phase fit in blocks; dominant sign part."""
    ph = np.unwrap(np.angle(z))
    nb = z.size // block
    j = np.arange(block)
    fr, cen = [], []
    for b in range(nb):
        seg = ph[b * block:(b + 1) * block]
        fr.append(np.polyfit(j, seg, 1)[0] / (2 * np.pi))
        cen.append(b * block + block / 2)
    return np.array(cen), np.array(fr)


def dominant(z):
    F = np.fft.fft(z)
    fr = np.fft.fftfreq(z.size)
    pos = np.sum(np.abs(F[fr > 0]) ** 2)
    neg = np.sum(np.abs(F[fr < 0]) ** 2)
    s = 1 if pos >= neg else -1
    F[s * fr <= 0] = 0
    return np.fft.ifft(F), s, float(max(pos, neg) / (pos + neg))


def drift_pct(z, block=64, edge=2):
    zd, s, frac = dominant(z)
    cen, fr = inst_freq(zd, block)
    a = np.abs(zd)
    amp = np.array([a[int(c - block / 2):int(c + block / 2)].mean() for c in cen])
    ok = amp > 0.05 * amp.max()
    ok[:edge] = False
    ok[-edge:] = False
    f = s * fr[ok]
    return {
        "nu_first_blocks_cyc_per_sample": float(np.median(f[:5])),
        "nu_last_blocks_cyc_per_sample": float(np.median(f[-5:])),
        "end_to_end_pct": float(100 * (np.median(f[-5:]) / np.median(f[:5]) - 1)),
        "max_minus_min_over_median_pct": float(100 * (f.max() - f.min()) / np.median(f)),
        "p5_p95_over_median_pct": float(100 * (np.percentile(f, 95) - np.percentile(f, 5)) / np.median(f)),
        "max_abs_nu_cyc_per_sample": float(np.max(np.abs(fr))),
        "dominant_sign_energy_frac": frac,
    }


def alias_check(z):
    F = np.abs(np.fft.fft(z)) ** 2
    fr = np.abs(np.fft.fftfreq(z.size))
    return {
        "power_frac_above_0.25_cyc_per_sample": float(F[fr > 0.25].sum() / F.sum()),
        "power_frac_above_0.4_cyc_per_sample": float(F[fr > 0.4].sum() / F.sum()),
    }


# ---------------- Part A: toy chirps -----------------
def cchirp(mc):
    tau = (L0 - np.arange(L0)) / FS
    fr = f_of_tau(tau, mc)
    return tau ** -0.25 * (fr < 900) * (fr > 20) * np.exp(1j * phi_of_tau(tau, mc))


A = {}
for mc in (1.0, 1.2, 1.4, 1.8, 2.4):
    h = cchirp(mc)
    unw = counts(h)
    tau_max = L0 / FS                       # first sample, tau = 4 s
    tau_cut = tau_of_f(900.0, mc)           # mask edge f = 900 Hz
    u0, u1 = tau_cut ** 0.625, tau_max ** 0.625
    c = (5 * TSUN * mc) ** (-0.625)        # Phi = -2 c u
    nu_u = c / np.pi                        # cycles per unit u (signed: Phi dec in u)
    cycles = nu_u * (u1 - u0)
    rows = {}
    for Nu in (4096, 8192, 16384):
        u = u1 - (np.arange(Nu) + 0.5) * (u1 - u0) / Nu   # time order: u decreasing
        # analytic 0PN signal in u
        hu = u ** -0.4 * np.exp(-2j * c * u)
        # band-limited interpolation of the sampled toy signal
        tpos = L0 - u ** 1.6 * FS                         # sample index of t = t_c - u^(8/5)
        hi = trig_interp(h, tpos)
        w = 1.6 * u ** 0.6                                # dt/du
        r = {"cyc_per_sample": float(nu_u * (u1 - u0) / Nu)}
        for name, z in (("analytic", hu), ("interp", hi)):
            cc = counts(z)
            cc["t_measure_k_for_0.9"] = k_for_wmatch(z, w, 0.9)
            cc["t_measure_k_for_0.97"] = k_for_wmatch(z, w, 0.97)
            r[name] = cc
        r["interp_vs_analytic_match_u"] = float(wmatch(hi, hu, np.ones(Nu)))
        r["alias_interp"] = alias_check(hi)
        rows[Nu] = r
    A[mc] = {"unwarped": unw, "tau_cut_s": tau_cut, "u_range": [u0, u1],
             "cycles_in_window": float(cycles), "warped": rows}
    print(f"toy Mc {mc}: unwarped {unw}; tau_cut {tau_cut*1e3:.3f} ms; cycles {cycles:.1f}")
    for Nu, r in rows.items():
        print(f"   Nu {Nu} ({r['cyc_per_sample']:.4f} cyc/sample): analytic {r['analytic']} | interp {r['interp']}"
              f" | interp~analytic {r['interp_vs_analytic_match_u']:.5f} | {r['alias_interp']}")
n90 = [A[m]["unwarped"]["0.9"] for m in (1.0, 1.4, 1.8, 2.4)]
print("figure set n90", n90, "median", np.median(n90))
res["toy"] = {str(k): v for k, v in A.items()}
_h = cchirp(1.2)
_t = np.sort(np.random.default_rng(0).uniform(0, L0 - 1, 512))
_a, _b = trig_interp(_h, _t), trig_interp_exact(_h, _t)
res["interp_check_rel_err"] = float(np.linalg.norm(_a - _b) / np.linalg.norm(_b))
print("spline-vs-exact trig interp rel err", res["interp_check_rel_err"])

# ---------------- Part B: whitened real templates -----------------
WF = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
PSDF = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
TC = json.load(open("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check/results.json"))
WIN, POST = 8192, 128
L = WIN - POST
TAPER = 512
with h5py.File(PSDF, "r") as f:
    S = f["psds"][:2000].mean(axis=(0, 1))
f_psd = np.arange(S.size) * 0.5
rows = {mc: TC["templates"][mc]["row"] for mc in ("1.0", "1.2", "1.5", "2.0")}
with h5py.File(WF, "r") as f:
    pol = {mc: (f["waveforms/plus"][r], f["waveforms/cross"][r]) for mc, r in rows.items()}


def whitened_window(plus, cross):
    h = plus - 1j * cross
    m = int(np.argmax(plus**2 + cross**2))
    seg = h[m - (L - 1): m + POST + 1]
    af = np.abs(np.fft.fftfreq(WIN, 1 / FS))
    keep = af >= 20.0
    H = np.fft.fft(seg)
    Hw = np.zeros_like(H)
    Hw[keep] = H[keep] / np.sqrt(np.interp(af[keep], f_psd, S))
    hw = np.fft.ifft(Hw)
    taper = np.ones(WIN)
    taper[:TAPER] = 0.5 * (1.0 - np.cos(np.pi * np.arange(TAPER) / TAPER))
    return m, hw * taper


def fit_tc(hw, mc_true):
    """Fit Phi(t) = a + b (t_c - t)^(5/8) on the dominant-sign phase where the
    0PN frequency (true Mc, t_c = peak) is 30-300 Hz; grid search t_c."""
    zd, s, _ = dominant(hw)
    ph = np.unwrap(np.angle(zd))
    n = np.arange(WIN)
    tau_peak = (L - 1 - n) / FS
    sel = np.zeros(WIN, bool)
    good = tau_peak > 0
    fr = np.zeros(WIN)
    fr[good] = f_of_tau(tau_peak[good], mc_true)
    sel = good & (fr > 30) & (fr < 300) & (n >= TAPER)
    best = None
    for dtc in np.arange(-0.02, 0.02, 1e-5):
        tt = tau_peak[sel] + dtc
        if tt.min() <= 0:
            continue
        Xd = np.stack([np.ones(tt.size), tt ** 0.625], 1)
        coef, *_ = np.linalg.lstsq(Xd, ph[sel], rcond=None)
        rms = np.sqrt(np.mean((ph[sel] - Xd @ coef) ** 2))
        if best is None or rms < best[0]:
            best = (rms, dtc, coef)
    rms, dtc, coef = best
    b = abs(coef[1])
    mc_fit = (b / 2) ** (-1.6) / (5 * TSUN)
    return {"dtc_s": float(dtc), "phase_rms_rad": float(rms), "mc_fit": float(mc_fit),
            "n_fit_samples": int(sel.sum())}


B = {}
for mc, (plus, cross) in pol.items():
    mct = TC["templates"][mc]["chirp_mass"]
    m, hw = whitened_window(plus, cross)
    K = hw[:L]
    unw = counts(K)
    E_full = np.sum(np.abs(K) ** 2) / FS
    fit = fit_tc(hw, mct)
    print(f"\nreal Mc {mc} (detector-frame Mc {mct:.4f}, row {rows[mc]}): unwarped {unw}; tc fit {fit}")
    entry = {"chirp_mass_det": mct, "unwarped": unw, "tc_fit": fit, "warped": {}}
    for tc_name, dtc in (("peak", 0.0), ("fit", fit["dtc_s"]),
                         ("fit-2ms", fit["dtc_s"] - 0.002), ("fit+2ms", fit["dtc_s"] + 0.002),
                         ("fit-5ms", fit["dtc_s"] - 0.005), ("fit+5ms", fit["dtc_s"] + 0.005)):
        tc_idx = (L - 1) + dtc * FS                          # t_c as fractional sample index
        for fcut in (300.0, 500.0, 900.0):
            tau_cut = tau_of_f(fcut, mct)
            if tc_idx - tau_cut * FS > L - 1:                # cut after the peak sample
                continue
            u1 = (tc_idx / FS) ** 0.625                       # sample 0
            u0 = tau_cut ** 0.625
            for Nu in (8192, 16384):
                u = u1 - (np.arange(Nu) + 0.5) * (u1 - u0) / Nu
                tpos = tc_idx - u ** 1.6 * FS
                z = trig_interp(hw, tpos)
                w = 1.6 * u ** 0.6
                du = (u1 - u0) / Nu
                E_kept = np.sum(w * np.abs(z) ** 2) * du
                cc = counts(z)
                cc["t_measure_k_for_0.9"] = k_for_wmatch(z, w, 0.9)
                cc["t_measure_k_for_0.97"] = k_for_wmatch(z, w, 0.97)
                # match against the whole unwarped window (t measure; nothing after the cut)
                nf = np.sqrt(E_full / du)
                cc["full_window_k_for_0.9"] = k_for_wmatch(z, w, 0.9, norm_full=nf)
                cc["full_window_max_match"] = float(np.sqrt(E_kept / E_full))
                cc["energy_frac_kept"] = float(E_kept / E_full)
                cc["tau_cut_ms"] = float(tau_cut * 1e3)
                cc["drift"] = drift_pct(z)
                cc["alias"] = alias_check(z)
                key = f"{tc_name}|fcut{int(fcut)}|Nu{Nu}"
                entry["warped"][key] = cc
                d = cc["drift"]
                print(f"  {key}: n90 {cc['0.9']} n97 {cc['0.97']} n99 {cc['0.99']} m32 {cc['match_at_32']:.3f}"
                      f" | t-meas k90 {cc['t_measure_k_for_0.9']} k97 {cc['t_measure_k_for_0.97']}"
                      f" | full-win k90 {cc['full_window_k_for_0.9']} (max {cc['full_window_max_match']:.4f})"
                      f" | drift e2e {d['end_to_end_pct']:+.2f}% p5-95 {d['p5_p95_over_median_pct']:.2f}%"
                      f" nu {d['nu_first_blocks_cyc_per_sample']:.4f}->{d['nu_last_blocks_cyc_per_sample']:.4f}"
                      f" max|nu| {d['max_abs_nu_cyc_per_sample']:.3f} dom {d['dominant_sign_energy_frac']:.3f}"
                      f" | >0.25 {cc['alias']['power_frac_above_0.25_cyc_per_sample']:.2e}")
                sys.stdout.flush()
    B[mc] = entry
res["real"] = B
json.dump(res, open(os.path.join(OUT, "results.json"), "w"), indent=1, default=float)
print("saved")
