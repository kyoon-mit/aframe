"""Second, independent check for the derivations section #oct16-nonlinearity-linear-part.

A. mpmath quadrature at 30 digits of r = corr(n, phi(n)), n ~ N(0, sigma^2),
   against the closed forms (ReLU, threshold at zero, sign, GELU, even
   functions, x^3 - 3x).
B. GELU: minimum of r(sigma) by root finding on dr/dsigma (mpmath), the sign
   of dr/dsigma on a grid, the range on 0.5 <= sigma <= 2, the small-sigma
   form 1 - 2 sigma^2 / pi.
C. Monte Carlo with an oscillating signal a cos(2 pi f t + theta), random
   theta per trial, K = 8192 samples at 2048 Hz, rho = 8: weights in step
   (c = s) and equal weights.
D. The setting of #semi-coherent: N = 600 pieces, each with signal rho/sqrt(N)
   and unit noise, equal weights (in step for a constant signal): Monte Carlo
   against r rho, r rho + r2 rho^2 / sqrt(2N), and the exact deflection
   sqrt(N) (m(a) - m(0)) / std phi(n) from mpmath.

Run single-threaded:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I check_nonlin_mp.py
and, for a second Monte Carlo of part C with other draws, with SEED_C=7 T_C=16000 in the environment.
"""
import json
import os
import time

import mpmath as mp
import numpy as np
from scipy import special

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))
mp.mp.dps = 30
lines = []
R = {}


def P(s=""):
    print(s, flush=True)
    lines.append(s)


# ---------------------------------------------------------------- A
def Phi_mp(x):
    return mp.ncdf(x)


MPF = {
    "ReLU": (lambda x: mp.mpf(0) if x < 0 else x, [0]),
    "threshold at zero": (lambda x: mp.mpf(1) if x > 0 else mp.mpf(0), [0]),
    "sign": (lambda x: mp.sign(x), [0]),
    "GELU": (lambda x: x * Phi_mp(x), [0]),
    "square": (lambda x: x * x, [0]),
    "absolute value": (lambda x: abs(x), [0]),
    "two-sided threshold |x| > sigma": (None, None),
    "cos x": (lambda x: mp.cos(x), [0]),
    "x^3 - 3x": (lambda x: x ** 3 - 3 * x, [0]),
}


def get_mp(name, s):
    if name.startswith("two-sided"):
        return (lambda x: mp.mpf(1) if abs(x) > s else mp.mpf(0)), [-s, 0, s]
    f, br = MPF[name]
    return f, br


def E_mp(f, s, br):
    dens = lambda x: mp.exp(-x * x / (2 * s * s)) / (s * mp.sqrt(2 * mp.pi))
    pts = [-mp.inf] + [mp.mpf(b) for b in br] + [mp.inf]
    return mp.quad(lambda x: f(x) * dens(x), pts)


def r_mp(name, s):
    f, br = get_mp(name, s)
    m1 = E_mp(f, s, br)
    m2 = E_mp(lambda x: f(x) ** 2, s, br)
    xn = E_mp(lambda x: x * f(x), s, br)
    sd = mp.sqrt(m2 - m1 ** 2)
    return xn / (s * sd), xn, m1, m2


def r_gelu_closed_mp(s):
    s = mp.mpf(s)
    c = s ** 2 / (1 + s ** 2)
    return (1 + (2 / mp.pi) * (mp.asin(c) - c + 2 * c / mp.sqrt(1 + 2 * s ** 2))) ** mp.mpf(-0.5)


CLOSED = {
    "ReLU": lambda s: 1 / mp.sqrt(2 - 2 / mp.pi),
    "threshold at zero": lambda s: mp.sqrt(2 / mp.pi),
    "sign": lambda s: mp.sqrt(2 / mp.pi),
    "GELU": r_gelu_closed_mp,
    "square": lambda s: mp.mpf(0),
    "absolute value": lambda s: mp.mpf(0),
    "two-sided threshold |x| > sigma": lambda s: mp.mpf(0),
    "cos x": lambda s: mp.mpf(0),
}

P("A. r by mpmath quadrature (30 digits) against closed forms")
R["A"] = []
worst = mp.mpf(0)
for name in CLOSED:
    sigmas = (0.25, 0.5, 1.0, 2.0, 4.0, 10.0) if name == "GELU" else (0.5, 1.0, 2.0)
    for s in sigmas:
        s_mp = mp.mpf(s)
        r, xn, m1, m2 = r_mp(name, s_mp)
        rc = CLOSED[name](s_mp)
        worst = max(worst, abs(r - rc))
        R["A"].append(dict(name=name, sigma=s, r=float(r), r_closed=float(rc), diff=float(abs(r - rc))))
        P(f"   {name:32s} sigma {s:5.2f}: r = {mp.nstr(r, 15):>18s}   closed {mp.nstr(rc, 15):>18s}")
P(f"   largest |difference| = {mp.nstr(worst, 3)}")
for s in (0.5, 1.0, 2.0):
    f, br = get_mp("x^3 - 3x", mp.mpf(s))
    xn = E_mp(lambda x: x * f(x), mp.mpf(s), br)
    R.setdefault("cubic", []).append(dict(sigma=s, E_n_phi=float(xn)))
    P(f"   x^3 - 3x, sigma {s}: E[n phi(n)] = {mp.nstr(xn, 12)} (3 sigma^4 - 3 sigma^2 = {3 * s ** 4 - 3 * s ** 2})")
# GELU mean and mean square against the closed forms of the section
for s in (0.5, 1.0, 2.0):
    s_mp = mp.mpf(s)
    f, br = get_mp("GELU", s_mp)
    m1 = E_mp(f, s_mp, br)
    m2 = E_mp(lambda x: f(x) ** 2, s_mp, br)
    c = s_mp ** 2 / (1 + s_mp ** 2)
    m1c = s_mp ** 2 / mp.sqrt(2 * mp.pi * (1 + s_mp ** 2))
    m2c = s_mp ** 2 * (mp.mpf(1) / 4 + mp.asin(c) / (2 * mp.pi)) + s_mp ** 4 / (mp.pi * (1 + s_mp ** 2) * mp.sqrt(1 + 2 * s_mp ** 2))
    P(f"   GELU sigma {s}: E GELU {mp.nstr(m1, 20)} (closed {mp.nstr(m1c, 20)}); E GELU^2 {mp.nstr(m2, 20)} (closed {mp.nstr(m2c, 20)})")
    R.setdefault("gelu_moments", []).append(dict(sigma=s, d1=float(abs(m1 - m1c)), d2=float(abs(m2 - m2c))))
P()

# ---------------------------------------------------------------- B
P("B. GELU r(sigma), closed form")
dr = lambda s: mp.diff(r_gelu_closed_mp, s)
s_star = mp.findroot(dr, mp.mpf("1.7"))
r_star = r_gelu_closed_mp(s_star)
grid = [mp.mpf(10) ** (mp.mpf(k) / 100) for k in range(-300, 301)]  # 1e-3 .. 1e3, 601 points
signs = [mp.sign(dr(g)) for g in grid]
changes = sum(1 for a, b in zip(signs[:-1], signs[1:]) if a != b)
neg_below = all(sg < 0 for g, sg in zip(grid, signs) if g < s_star)
pos_above = all(sg > 0 for g, sg in zip(grid, signs) if g > s_star)
r05, r2 = r_gelu_closed_mp(0.5), r_gelu_closed_mp(2)
small = [(sv, r_gelu_closed_mp(sv), 1 - 2 * mp.mpf(sv) ** 2 / mp.pi) for sv in (0.01, 0.05)]
R["B"] = dict(sigma_min=float(s_star), r_min=float(r_star), sign_changes=changes, decreasing_below=neg_below,
              increasing_above=pos_above, r_0p5=float(r05), r_2=float(r2),
              relu=float(1 / mp.sqrt(2 - 2 / mp.pi)), r_1e3=float(r_gelu_closed_mp(1000)))
P(f"   dr/dsigma = 0 at sigma = {mp.nstr(s_star, 12)}, r there = {mp.nstr(r_star, 12)}")
P(f"   sign of dr/dsigma on 601 points, 1e-3 to 1e3: {changes} change(s); negative below the root: {neg_below}; positive above: {pos_above}")
P(f"   r(0.5) = {mp.nstr(r05, 12)}, r(2) = {mp.nstr(r2, 12)}, so on [0.5, 2]: {mp.nstr(r_star, 6)} to {mp.nstr(r05, 6)}")
P(f"   r(1000) = {mp.nstr(r_gelu_closed_mp(1000), 12)}, ReLU value {mp.nstr(1 / mp.sqrt(2 - 2 / mp.pi), 12)}")
for sv, rv, approx in small:
    P(f"   sigma {sv}: r = {mp.nstr(rv, 12)}, 1 - 2 sigma^2/pi = {mp.nstr(approx, 12)}")
P()

# ---------------------------------------------------------------- C
Phi = special.ndtr
NP = {
    "GELU": lambda x: x * Phi(x),
    "ReLU": lambda x: np.maximum(x, 0.0),
    "threshold at zero": lambda x: (x > 0).astype(float),
    "square": lambda x: x * x,
    "absolute value": np.abs,
}
K, FS, F0, RHO = 8192, 2048.0, 101.3, 8.0
T, CH = int(os.environ.get("T_C", "8000")), 250  # second run: T_C=16000 SEED_C=7
t = np.arange(K) / FS
P(f"C. Monte Carlo: K = {K} samples at {FS:.0f} Hz, signal a cos(2 pi {F0} t + theta), theta random per trial,")
P(f"   ||s|| = rho sigma with rho = {RHO}; {T} trials, the same noise with and without the signal")
rng = np.random.default_rng(int(os.environ.get("SEED_C", "161016")))
R["C"] = []
for name, s in (("GELU", 0.5), ("GELU", 1.0), ("GELU", 1.678185), ("GELU", 2.0), ("ReLU", 1.0),
                ("threshold at zero", 1.0), ("square", 1.0), ("absolute value", 1.0)):
    phi = NP[name]
    r_th = float(r_mp(name, mp.mpf(s))[0]) if name != "GELU" else float(r_gelu_closed_mp(s))
    # r2 = corr(phi(n), n^2 - sigma^2) by mpmath
    f, br = get_mp(name, mp.mpf(s))
    m1 = E_mp(f, mp.mpf(s), br)
    m2 = E_mp(lambda x: f(x) ** 2, mp.mpf(s), br)
    xq = E_mp(lambda x: (x * x - s * s) * f(x), mp.mpf(s), br)
    r2_th = float(xq / (mp.sqrt(2) * s * s * mp.sqrt(m2 - m1 ** 2)))
    o0_in, d_in, o0_eq, d_eq = [], [], [], []
    for _ in range(T // CH):
        theta = rng.uniform(0, 2 * np.pi, (CH, 1))
        sig = np.cos(2 * np.pi * F0 * t[None, :] + theta)
        sig *= RHO * s / np.linalg.norm(sig, axis=1, keepdims=True)
        c_in = sig / np.linalg.norm(sig, axis=1, keepdims=True)
        n = s * rng.standard_normal((CH, K))
        f0 = phi(n)
        df = phi(n + sig) - f0
        o0_in.append((f0 * c_in).sum(1)); d_in.append((df * c_in).sum(1))
        o0_eq.append(f0.sum(1) / np.sqrt(K)); d_eq.append(df.sum(1) / np.sqrt(K))
    sd_exact = float(mp.sqrt(m2 - m1 ** 2))
    out, outx = {}, {}
    for tag, o0, d in (("in step", o0_in, d_in), ("equal", o0_eq, d_eq)):
        o0, d = np.concatenate(o0), np.concatenate(d)
        sd = o0.std(ddof=1)
        v = d.mean() / sd
        se = np.sqrt((d.std(ddof=1) / np.sqrt(T) / sd) ** 2 + v ** 2 / (2 * T))
        out[tag] = (float(v), float(se))
        # mean shift over the exact noise spread: tests the mean shift alone
        outx[tag] = (float(d.mean() / sd_exact), float(d.std(ddof=1) / np.sqrt(T) / sd_exact), float(sd / sd_exact))
    pred_in, pred_eq = r_th * RHO, r2_th * RHO ** 2 / np.sqrt(2 * K)
    R["C"].append(dict(name=name, sigma=s, in_step=out["in step"], equal=out["equal"], pred_in=pred_in, pred_eq=pred_eq,
                       in_step_exact_spread=outx["in step"], equal_exact_spread=outx["equal"]))
    P(f"   {name:18s} sigma {s:5.3f}: in step {out['in step'][0]:6.3f} +- {out['in step'][1]:.3f} (r rho {pred_in:6.3f});"
      f" equal weights {out['equal'][0]:7.4f} +- {out['equal'][1]:.4f} (r2 rho^2/sqrt(2K) {pred_eq:.4f})")
    P(f"   {'':18s} {'':11s}  with the exact spread: in step {outx['in step'][0]:7.4f} +- {outx['in step'][1]:.4f};"
      f" equal weights {outx['equal'][0]:7.4f} +- {outx['equal'][1]:.4f};"
      f" measured/exact spread {outx['in step'][2]:.4f}, {outx['equal'][2]:.4f}")
# exact expected in-step SNR of the threshold at zero for this cosine: E H(s + n) = Phi(s / sigma),
# averaged over 4000 phases; and the first-order value with the s^3 term, r rho (1 - rho^2 / (4 K))
th = (np.arange(4000) + 0.5) / 4000 * 2 * np.pi
acc = []
for j in range(0, 4000, 250):
    sig = np.cos(2 * np.pi * F0 * t[None, :] + th[j:j + 250, None])
    sig *= RHO / np.linalg.norm(sig, axis=1, keepdims=True)
    acc.append(((sig / RHO) * (Phi(sig) - 0.5)).sum(1) / 0.5)
exact_thr = float(np.concatenate(acc).mean())
cubic_thr = float(np.sqrt(2 / np.pi) * RHO * (1 - RHO ** 2 / (4 * K)))
R["C_threshold_exact"] = dict(exact=exact_thr, first_order=float(np.sqrt(2 / np.pi) * RHO), with_cubic=cubic_thr)
P(f"   threshold at zero, in step, exact mean for this cosine (4000 phases): {exact_thr:.5f};"
  f" r rho {np.sqrt(2 / np.pi) * RHO:.5f}; with the s^3 term r rho (1 - rho^2/(4K)) {cubic_thr:.5f}")
P()

# ---------------------------------------------------------------- D
N, TD = 600, 20000
a = RHO / np.sqrt(N)
P(f"D. #semi-coherent setting: N = {N} pieces z_k = {a:.4f} + n_k, unit noise, equal weights, {TD} trials")
rng = np.random.default_rng(1966)
R["D"] = []
for name in ("GELU", "ReLU", "threshold at zero", "square", "absolute value"):
    phi = NP[name]
    f, br = get_mp(name, mp.mpf(1))
    m0 = E_mp(f, mp.mpf(1), br)
    m2 = E_mp(lambda x: f(x) ** 2, mp.mpf(1), br)
    sd = mp.sqrt(m2 - m0 ** 2)
    ma = E_mp(lambda x: f(x + a), mp.mpf(1), [b - a for b in br])
    exact = float(mp.sqrt(N) * (ma - m0) / sd)
    xn = E_mp(lambda x: x * f(x), mp.mpf(1), br)
    xq = E_mp(lambda x: (x * x - 1) * f(x), mp.mpf(1), br)
    first = float(xn / sd) * RHO
    second = first + float(xq / (mp.sqrt(2) * sd)) * RHO ** 2 / np.sqrt(2 * N)
    o0, d = [], []
    for _ in range(TD // 2000):
        n = rng.standard_normal((2000, N))
        f0 = phi(n)
        o0.append(f0.sum(1)); d.append((phi(n + a) - f0).sum(1))
    o0, d = np.concatenate(o0), np.concatenate(d)
    v = d.mean() / o0.std(ddof=1)
    se = np.sqrt((d.std(ddof=1) / np.sqrt(TD) / o0.std(ddof=1)) ** 2 + v ** 2 / (2 * TD))
    # closed forms of m(a), sigma = 1
    am = mp.mpf(a)
    m_closed = {
        "GELU": am * mp.ncdf(am / mp.sqrt(2)) + mp.npdf(am / mp.sqrt(2)) / mp.sqrt(2),
        "ReLU": am * mp.ncdf(am) + mp.npdf(am),
        "threshold at zero": mp.ncdf(am),
        "square": am ** 2 + 1,
        "absolute value": am * (2 * mp.ncdf(am) - 1) + 2 * mp.npdf(am),
    }[name]
    exact_closed = float(mp.sqrt(N) * (m_closed - m0) / sd)
    h_closed = {"GELU": 3 / (4 * mp.sqrt(mp.pi)), "ReLU": 1 / mp.sqrt(2 * mp.pi), "threshold at zero": mp.mpf(0),
                "square": mp.mpf(2), "absolute value": mp.sqrt(2 / mp.pi)}[name]
    P(f"   {name:18s}: h = E[(n^2 - 1) phi(n)] = {mp.nstr(xq, 12)} (closed form {mp.nstr(h_closed, 12)})")
    R["D"].append(dict(name=name, mc=float(v), se=float(se), first=first, first_second=second, exact=exact,
                       exact_closed=exact_closed, m_a=float(ma), m_a_closed=float(m_closed)))
    P(f"   {name:18s}: MC {v:6.3f} +- {se:.3f}; first order r rho {first:6.3f}; with the power term {second:6.3f};"
      f" exact {exact:8.5f} (closed form of m(a): {exact_closed:8.5f})")
P()
P(f"wall time {time.time() - T0:.1f} s")
TAG = "" if "SEED_C" not in os.environ else f"_seed{os.environ['SEED_C']}_T{T}"
json.dump(R, open(os.path.join(HERE, f"check_nonlin_mp{TAG}.json"), "w"), indent=1, default=float)
open(os.path.join(HERE, f"check_nonlin_mp{TAG}.txt"), "w").write("\n".join(lines) + "\n")
