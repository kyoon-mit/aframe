"""Sensitivity cost of adding powers of K pieces versus one in-phase sum.

Coherent: |z|^2, z complex, noise Re/Im ~ N(0,1): chi2(2 dof), signal noncentrality rho^2.
Power sum over K pieces with free phases: sum_k |z_k|^2 ~ ncx2(2K, rho^2) when the K pieces
together hold all the signal energy rho^2.
"""
import numpy as np
from scipy import stats, optimize

Ks = [1, 2, 4, 8, 16, 32, 64]

def rho_needed(K, pfa, pd):
    thr = stats.chi2.isf(pfa, 2 * K)
    f = lambda r: stats.ncx2.sf(thr, 2 * K, r * r) - pd
    return optimize.brentq(f, 1e-6, 200.0, xtol=1e-10)

for pfa, pd in [(0.05, 0.95), (1e-4, 0.5)]:
    r1 = rho_needed(1, pfa, pd)
    print(f"P_FA={pfa:g} P_D={pd:g}: coherent rho={r1:.4f}")
    for K in Ks:
        rK = rho_needed(K, pfa, pd)
        print(f"   K={K:3d} rho={rK:.4f} ratio={rK / r1:.4f}  K^(1/4)={K ** 0.25:.4f}")

# deflection (mean shift / noise std) check by Monte Carlo
rng = np.random.default_rng(12345)
n = 200000
rho = 8.0
print("Monte Carlo deflection check, rho=8, n=%d" % n)
for K in [1, 4, 16]:
    amp = rho / np.sqrt(K)                      # equal energy per piece, total rho^2
    ph = rng.uniform(0, 2 * np.pi, size=(n, K))  # independent phase per piece
    sig = amp * np.exp(1j * ph)
    noise = rng.standard_normal((n, K)) + 1j * rng.standard_normal((n, K))
    s_h1 = np.sum(np.abs(sig + noise) ** 2, axis=1)
    s_h0 = np.sum(np.abs(rng.standard_normal((n, K)) + 1j * rng.standard_normal((n, K))) ** 2, axis=1)
    d_mc = (s_h1.mean() - s_h0.mean()) / s_h0.std()
    d_th = rho ** 2 / (2 * np.sqrt(K))
    print(f"   K={K:2d} deflection MC={d_mc:.3f} theory rho^2/(2 sqrt K)={d_th:.3f}")
    # ROC point check at P_FA=0.05 against ncx2
    thr = stats.chi2.isf(0.05, 2 * K)
    print(f"        P_D at P_FA=0.05: MC={np.mean(s_h1 > thr):.4f} ncx2={stats.ncx2.sf(thr, 2 * K, rho * rho):.4f}")
