"""Checks for the derivations section #oct16-nonlinearity-linear-part.

r = corr(n, phi(n)), n ~ N(0, sigma^2), for the identity, ReLU, threshold at
zero, sign, GELU, square, absolute value and a two-sided threshold.

1. Numerical integration (scipy.integrate.quad) against closed forms.
2. GELU: closed form for r(sigma), its minimum, limits, and the range on
   0.5 <= sigma <= 2; GELU of PyTorch against x * Phi(x).
3. Monte Carlo of r from 10^7 Gaussian draws per case.
4. Monte Carlo of the sum of K = 8192 pieces at SNR 8, weights in step and
   equal weights, against r * rho and r2 * rho^2 / sqrt(2 K).
5. Exactness of r * rho at any amplitude for GELU and ReLU: m(a) - m(-a) = a.

Run single-threaded:
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -I check_nonlin.py
"""
import json
import os
import time

import numpy as np
from scipy import integrate, optimize, special

T0 = time.time()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "check_nonlin.json")
SQ2PI = np.sqrt(2 * np.pi)
Phi = special.ndtr


def gelu(x):
    return x * Phi(x)


FUNCS = {
    "identity": (lambda x: x, ()),
    "ReLU": (lambda x: np.maximum(x, 0.0), (0.0,)),
    "threshold at zero": (lambda x: (np.asarray(x) > 0).astype(float), (0.0,)),
    "sign": (np.sign, (0.0,)),
    "GELU": (gelu, ()),
    "square": (lambda x: np.asarray(x) ** 2, ()),
    "absolute value": (np.abs, (0.0,)),
    "two-sided threshold |x| > sigma": (None, None),  # set per sigma
}


def get(name, sigma):
    if name.startswith("two-sided"):
        return (lambda x: (np.abs(np.asarray(x)) > sigma).astype(float)), (-sigma, sigma)
    return FUNCS[name]


def gauss_E(f, sigma, breaks=()):
    """E f(n), n ~ N(0, sigma^2): adaptive integration on [-14 sigma, 14 sigma], split at kinks."""
    lo, hi = -14.0 * sigma, 14.0 * sigma
    pts = [lo] + sorted(b for b in breaks if lo < b < hi) + [hi]
    p = lambda x: np.exp(-0.5 * (x / sigma) ** 2) / (sigma * SQ2PI)
    tot = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        tot += integrate.quad(lambda x: float(f(x)) * p(x), a, b, epsabs=1e-15, epsrel=1e-12, limit=500)[0]
    return tot


def moments(phi, sigma, breaks):
    m1 = gauss_E(phi, sigma, breaks)
    m2 = gauss_E(lambda x: float(phi(x)) ** 2, sigma, breaks)
    xn = gauss_E(lambda x: x * float(phi(x)), sigma, breaks)
    xq = gauss_E(lambda x: (x * x - sigma ** 2) * float(phi(x)), sigma, breaks)
    sd = np.sqrt(max(m2 - m1 ** 2, 0.0))
    return dict(mean=m1, sd=sd, gain=xn / sigma ** 2, r=xn / (sigma * sd), r2=xq / (np.sqrt(2) * sigma ** 2 * sd))


def r_gelu_closed(sigma):
    s2 = sigma ** 2
    c = s2 / (1 + s2)
    return (1 + (2 / np.pi) * (np.arcsin(c) - c + 2 * c / np.sqrt(1 + 2 * s2))) ** -0.5


def gelu_moments_closed(sigma):
    s2 = sigma ** 2
    c = s2 / (1 + s2)
    mean = s2 / np.sqrt(2 * np.pi * (1 + s2))
    second = s2 * (0.25 + np.arcsin(c) / (2 * np.pi) + s2 / (np.pi * (1 + s2) * np.sqrt(1 + 2 * s2)))
    return mean, second


CLOSED_R = {
    "identity": lambda s: 1.0,
    "ReLU": lambda s: 1 / np.sqrt(2 - 2 / np.pi),
    "threshold at zero": lambda s: np.sqrt(2 / np.pi),
    "sign": lambda s: np.sqrt(2 / np.pi),
    "GELU": r_gelu_closed,
    "square": lambda s: 0.0,
    "absolute value": lambda s: 0.0,
    "two-sided threshold |x| > sigma": lambda s: 0.0,
}

R = {"integration": [], "gelu": {}, "mc_r": [], "mc_sum": [], "exact_odd": []}
lines = []


def P(s=""):
    print(s, flush=True)
    lines.append(s)


# 1. numerical integration against closed forms
P("1. r = corr(n, phi(n)) by numerical integration, against closed forms")
P(f"{'function':34s} {'sigma':>6s} {'gain E[n phi]/s^2':>18s} {'sd phi':>10s} {'r (integration)':>16s} {'r (closed)':>11s} {'r2':>9s}")
worst = 0.0
for name in FUNCS:
    sigmas = (0.25, 0.5, 1.0, 1.678, 2.0, 4.0, 10.0) if name == "GELU" else (0.5, 1.0, 2.0)
    for s in sigmas:
        phi, br = get(name, s)
        mo = moments(phi, s, br)
        rc = CLOSED_R[name](s)
        worst = max(worst, abs(mo["r"] - rc))
        R["integration"].append(dict(name=name, sigma=s, **mo, r_closed=rc))
        P(f"{name:34s} {s:6.3f} {mo['gain']:18.12f} {mo['sd']:10.6f} {mo['r']:16.12f} {rc:11.8f} {mo['r2']:9.6f}")
P(f"largest |r(integration) - r(closed)| = {worst:.2e}")
for s in (0.5, 1.0, 2.0):
    phi, br = get("GELU", s)
    m1 = gauss_E(phi, s, br)
    m2 = gauss_E(lambda x: float(phi(x)) ** 2, s, br)
    c1, c2 = gelu_moments_closed(s)
    P(f"GELU sigma {s}: E GELU integration {m1:.14f} closed {c1:.14f}; E GELU^2 integration {m2:.14f} closed {c2:.14f}")
P()

# 2. GELU: minimum, limits, range on [0.5, 2]
res = optimize.minimize_scalar(r_gelu_closed, bounds=(0.1, 10.0), method="bounded", options=dict(xatol=1e-10))
grid = np.geomspace(1e-3, 1e3, 400001)
rg = r_gelu_closed(grid)
d = np.diff(rg)
turns = int(np.sum(np.sign(d[1:]) != np.sign(d[:-1])))
sel = (grid >= 0.5) & (grid <= 2.0)
phi_min, _ = get("GELU", res.x)
r_min_int = moments(phi_min, res.x, ())["r"]
R["gelu"] = dict(sigma_min=res.x, r_min=res.fun, r_min_integration=r_min_int, turns_of_slope_on_grid=turns,
                 r_at_small=float(r_gelu_closed(1e-3)), r_at_large=float(r_gelu_closed(1e3)),
                 relu_r=1 / np.sqrt(2 - 2 / np.pi),
                 range_0p5_2=[float(rg[sel].min()), float(rg[sel].max())],
                 r_at=dict((str(s), float(r_gelu_closed(s))) for s in (0.25, 0.5, 1.0, 2.0, 4.0, 10.0)))
P("2. GELU, closed form r(sigma) = [1 + (2/pi)(arcsin c - c + 2c/sqrt(1 + 2 sigma^2))]^(-1/2), c = sigma^2/(1 + sigma^2)")
P(f"   minimum {res.fun:.9f} at sigma = {res.x:.6f} (integration there: {r_min_int:.9f})")
P(f"   slope of r changes sign {turns} time(s) on 400,001 points from sigma 1e-3 to 1e3")
P(f"   r(1e-3) = {r_gelu_closed(1e-3):.9f}, r(1e3) = {r_gelu_closed(1e3):.9f}, ReLU value {1/np.sqrt(2-2/np.pi):.9f}")
P(f"   on 0.5 <= sigma <= 2: {rg[sel].min():.6f} to {rg[sel].max():.6f}")
try:
    import torch
    torch.set_num_threads(1)
    x = torch.linspace(-12, 12, 240001, dtype=torch.float64)
    dt = (torch.nn.functional.gelu(x) - x * 0.5 * (1 + torch.erf(x / np.sqrt(2)))).abs().max().item()
    dn = np.abs(torch.nn.GELU()(x).numpy() - gelu(x.numpy())).max()
    R["gelu"]["torch_vs_xPhi"] = [dt, float(dn)]
    P(f"   torch {torch.__version__} nn.GELU() against x Phi(x) on [-12, 12], float64: max difference {dn:.2e}")
except Exception as e:  # noqa: BLE001
    P(f"   torch check skipped: {e}")
P()

# 3. Monte Carlo of r
P("3. Monte Carlo of r = corr(n, phi(n)), 10^7 draws per case (standard error (1 - r^2)/sqrt(N))")
rng = np.random.default_rng(20261016)
NMC, CH = 10_000_000, 2_000_000
cases = [("GELU", 0.5), ("GELU", 1.0), ("GELU", float(res.x)), ("GELU", 2.0), ("ReLU", 1.0),
         ("threshold at zero", 1.0), ("square", 1.0), ("absolute value", 1.0)]
for name, s in cases:
    phi, _ = get(name, s)
    acc = np.zeros(5)
    for _ in range(NMC // CH):
        n = s * rng.standard_normal(CH)
        y = phi(n)
        acc += [n.sum(), y.sum(), (n * n).sum(), (y * y).sum(), (n * y).sum()]
    sn, sy, snn, syy, sny = acc / NMC
    r = (sny - sn * sy) / np.sqrt((snn - sn ** 2) * (syy - sy ** 2))
    se = (1 - r ** 2) / np.sqrt(NMC)
    rc = CLOSED_R[name](s)
    R["mc_r"].append(dict(name=name, sigma=s, r_mc=r, se=se, r_closed=rc))
    P(f"   {name:20s} sigma {s:6.3f}: r = {r:.5f} +- {se:.5f}   (closed {rc:.5f}, difference {(r - rc) / se:+.1f} se)")
P()

# 4. Monte Carlo of the in-step sum and the equal-weight sum
K, RHO, T, TCH = 8192, 8.0, 20000, 250
P(f"4. Sum of K = {K} pieces, signal +-a alternating, rho = ||s||/sigma = {RHO}, {T} noise draws,")
P("   same draws with and without the signal; SNR = mean(o(n+s) - o(n)) / std(o(n))")
signs = np.where(np.arange(K) % 2 == 0, 1.0, -1.0)
mc_cases = [("identity", 1.0), ("GELU", 0.5), ("GELU", 1.0), ("GELU", 2.0), ("ReLU", 1.0),
            ("threshold at zero", 1.0), ("square", 1.0), ("absolute value", 1.0)]
rng = np.random.default_rng(15)
for name, s in mc_cases:
    phi, br = get(name, s)
    a = RHO * s / np.sqrt(K)
    sig = a * signs
    c_in = signs / np.sqrt(K)          # in step, unit norm
    c_eq = np.ones(K) / np.sqrt(K)     # equal weights, unit norm
    o0i, d_i, o0e, d_e = [], [], [], []
    for _ in range(T // TCH):
        n = s * rng.standard_normal((TCH, K))
        f0 = phi(n)
        f1 = phi(n + sig)
        df = f1 - f0
        o0i.append(f0 @ c_in); d_i.append(df @ c_in)
        o0e.append(f0 @ c_eq); d_e.append(df @ c_eq)
    out = {}
    for tag, o0, dd in (("in step", o0i, d_i), ("equal", o0e, d_e)):
        o0 = np.concatenate(o0); dd = np.concatenate(dd)
        sd = o0.std(ddof=1)
        v = dd.mean() / sd
        se = np.sqrt((dd.std(ddof=1) / np.sqrt(T) / sd) ** 2 + v ** 2 / (2 * T))
        out[tag] = (v, se)
    mo = moments(phi, s, br)
    pred_in = mo["r"] * RHO
    pred_eq = mo["r2"] * RHO ** 2 / np.sqrt(2 * K)
    R["mc_sum"].append(dict(name=name, sigma=s, a=a, in_step=out["in step"], equal=out["equal"],
                            pred_in=pred_in, pred_eq=pred_eq))
    P(f"   {name:20s} sigma {s:4.2f} a {a:.4f}: in step {out['in step'][0]:7.3f} +- {out['in step'][1]:.3f} (r rho {pred_in:6.3f});"
      f" equal weights {out['equal'][0]:7.4f} +- {out['equal'][1]:.4f} (r2 rho^2/sqrt(2K) {pred_eq:.4f})")
P()

# 5. exactness of the odd part for GELU and ReLU
P("5. m(a) - m(-a) - a, m(a) = E phi(a + n), by numerical integration (zero means r rho is exact for +- pairs)")
for name in ("GELU", "ReLU", "sign"):
    for s in (0.5, 1.0, 2.0):
        for a in (0.1, 1.0, 3.0):
            phi, br = get(name, s)
            ma = gauss_E(lambda x: float(phi(x + a)), s, tuple(b - a for b in br))
            mma = gauss_E(lambda x: float(phi(x - a)), s, tuple(b + a for b in br))
            R["exact_odd"].append(dict(name=name, sigma=s, a=a, diff=ma - mma - a))
    worst = max(abs(e["diff"]) for e in R["exact_odd"] if e["name"] == name)
    P(f"   {name:5s}: largest |m(a) - m(-a) - a| over sigma 0.5, 1, 2 and a 0.1, 1, 3 = {worst:.2e}")
    if name == "sign":
        e = max((e for e in R["exact_odd"] if e["name"] == name), key=lambda e: abs(e["diff"]))
        P(f"   sign: largest at sigma {e['sigma']}, a {e['a']}: {e['diff']:.4f}")
P()

# 6. a function that is not even but has r = 0 at one spread: x^3 - 3 x (sigma = 1)
P("6. phi(x) = x^3 - 3x: E[n phi(n)] by numerical integration")
for s in (0.5, 1.0, 2.0):
    v = gauss_E(lambda x: x * (x ** 3 - 3 * x), s)
    R.setdefault("cubic", []).append(dict(sigma=s, E_n_phi=v, exact=3 * s ** 4 - 3 * s ** 2))
    P(f"   sigma {s}: E[n phi(n)] = {v:.12f} (3 sigma^4 - 3 sigma^2 = {3 * s ** 4 - 3 * s ** 2:.12f})")
P()
P(f"wall time {time.time() - T0:.1f} s")
json.dump(R, open(OUT, "w"), indent=1, default=float)
open(OUT.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n")
