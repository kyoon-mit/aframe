"""Independent recomputation of the segmented Fisher bounds and detection numbers.

Own code: uniform fine frequency grid with Simpson weights (not Gauss-Legendre),
full Fisher matrices built and inverted directly (no closed forms), segment
edges from tau(f) inverted analytically.
"""
import numpy as np
from scipy import stats, optimize

TSUN = 4.925490947641267e-06
PSD_FILE = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
p = np.load(PSD_FILE)["psd"]
FP = np.arange(p.shape[1]) * 0.5
SNET = 2 / (1 / p[0] + 1 / p[1])
print("PSD shape", p.shape, "grid max", FP[-1])


def S(f):
    return np.interp(f, FP, SNET)


def f_of_tau(tau, mc):
    return (5 / (256 * tau)) ** 0.375 * (mc * TSUN) ** -0.625 / np.pi


def f_isco(mc):
    return 1 / (6 ** 1.5 * np.pi * TSUN * 2 ** 1.2 * mc)


def grid(f0, f1, n=400001):
    f = np.linspace(f0, f1, n)
    w = np.ones(n)
    w[1:-1:2], w[2:-1:2] = 4, 2
    return f, w * (f[1] - f[0]) / 3


def fisher_bounds(mc, R, rho=8.0, T=4.0, psd=S, fhi=None, free_time=False, offset=0.0):
    flo = f_of_tau(T, mc)
    fhi = fhi or min(f_isco(mc), 1024.0)
    f, q = grid(flo, fhi)
    wt = q * f ** (-7 / 3) / psd(f)
    wt = wt / wt.sum() * rho ** 2                         # rho^2 <.> weights
    a = -(5 / 128) * (np.pi * mc * TSUN * f) ** (-5 / 3)
    b = 2 * np.pi * f
    tau = 5 / 256 * (mc * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
    seg = np.floor((tau + offset * R) / R).astype(int) if R < T else np.zeros_like(f, int)
    ids = np.unique(seg)
    nseg = len(ids)
    if not free_time:
        # params: lnMc, tc, phi_1..phi_N
        D = np.zeros((2 + nseg, len(f)))
        D[0], D[1] = a, b
        for i, k in enumerate(ids):
            D[2 + i, seg == k] = -1.0
    else:
        D = np.zeros((1 + 2 * nseg, len(f)))
        D[0] = a
        for i, k in enumerate(ids):
            D[1 + 2 * i, seg == k] = b[seg == k]
            D[2 + 2 * i, seg == k] = -1.0
    G = (D * wt) @ D.T
    sc = 1 / np.sqrt(np.diag(G))
    Ci = np.linalg.inv(G * sc[:, None] * sc[None]) * sc[:, None] * sc[None]
    return np.sqrt(Ci[0, 0]), nseg


print("\nsigma(ln Mc) in %, SNR 8 (claims: coh 0.0282/0.0331/0.0597; R0.44 0.1771/0.2050/0.3147;"
      " R1 0.0874/0.1014/0.1584; R2 0.0489/0.0569/0.0924; free-time R0.44 0.3112/0.3184/0.4163)")
for mc in (0.85, 1.2, 2.75):
    coh, _ = fisher_bounds(mc, 4.0)
    row = [f"coh {100 * coh:.4f}"]
    for R in (2.0, 1.0, 0.44):
        s, n = fisher_bounds(mc, R)
        row.append(f"R{R} {100 * s:.4f} (x{s / coh:.2f}, N={n})")
    s, n = fisher_bounds(mc, 0.44, free_time=True)
    row.append(f"free-time R0.44 {100 * s:.4f} (x{s / coh:.2f})")
    print(f"Mc {mc}: " + "; ".join(row))

# law of total covariance check on Mc 1.2, R 0.44
mc = 1.2
flo, fhi = f_of_tau(4.0, mc), min(f_isco(mc), 1024.0)
f, q = grid(flo, fhi)
wt = q * f ** (-7 / 3) / S(f)
wt /= wt.sum()
a = -(5 / 128) * (np.pi * mc * TSUN * f) ** (-5 / 3)
b = 2 * np.pi * f
tau = 5 / 256 * (mc * TSUN) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
seg = np.floor(tau / 0.44).astype(int)
V = np.stack([a, b])
mu = V @ wt
tot = (V - mu[:, None]) * wt @ (V - mu[:, None]).T
within = np.zeros((2, 2))
between = np.zeros((2, 2))
for k in np.unique(seg):
    m = seg == k
    wk = wt[m].sum()
    muk = V[:, m] @ wt[m] / wk
    within += (V[:, m] - muk[:, None]) * wt[m] @ (V[:, m] - muk[:, None]).T
    between += wk * np.outer(muk - mu, muk - mu)
print("\ntotal-covariance identity, max rel diff:", np.abs(tot - within - between).max() / np.abs(tot).max())
shares = [wt[seg == k].sum() for k in np.unique(seg)]
print("shares of rho^2 per 0.44 s segment (merger first):", np.round(shares, 3))

# detection numbers
print("\ndetection at FAP 1e-4: (claims N=1: 18.42, 4.173; N=2 4.527; N=4 4.979; N=9 5.667; N=10 5.772)")
for N in (1, 2, 4, 9, 10):
    th = stats.chi2.isf(1e-4, 2 * N)
    r = optimize.brentq(lambda r: stats.ncx2.sf(th, 2 * N, r * r) - 0.5, 1, 20)
    print(f"N={N}: threshold {th:.3f}, SNR50 {r:.4f}")

# Cutler and Flanagan 1994 Newtonian check: S ~ (f0/f)^4 + 2(1 + (f/f0)^2), f0 = 70 Hz, from 10 Hz
cf = lambda f: (70 / f) ** 4 + 2 * (1 + (f / 70) ** 2)
for fhi, lab in ((40000.0, "to 40 kHz"), (f_isco(1.2188), "to f_ISCO(1.4+1.4)")):
    mc = 1.0
    f, q = grid(10.0, fhi, 2000001)
    wt = q * f ** (-7 / 3) / cf(f)
    wt = wt / wt.sum() * 100.0
    a = -(5 / 128) * (np.pi * mc * TSUN * f) ** (-5 / 3)
    D = np.stack([a, 2 * np.pi * f, -np.ones_like(f)])
    G = (D * wt) @ D.T
    sc = 1 / np.sqrt(np.diag(G))
    C = np.linalg.inv(G * sc[:, None] * sc[None]) * sc[:, None] * sc[None]
    print(f"CF94 noise {lab}: sigma(lnMc)/Mc^(5/3) = {np.sqrt(C[0, 0]):.4e}, dtc = {1e3 * np.sqrt(C[1, 1]):.4f} ms,"
          f" dphi = {np.sqrt(C[2, 2]):.4f} rad (claims 1.238e-5/0.404 ms/0.2485 rad; ISCO 1.247e-5/0.417/0.254)")
