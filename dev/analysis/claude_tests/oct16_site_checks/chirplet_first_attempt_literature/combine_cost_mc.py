"""Monte Carlo check of the required-SNR table: P_D at the tabulated rho should match the target."""
import numpy as np
from scipy import stats
rng = np.random.default_rng(7)
n = 200000
table = {(0.05, 0.95): {1: 3.9298, 4: 4.7690, 16: 6.0310, 64: 7.9377},
         (1e-4, 0.5): {1: 4.1727, 4: 4.9785, 16: 6.2830, 64: 8.3097}}
for (pfa, pd), rows in table.items():
    for K, rho in rows.items():
        thr = stats.chi2.isf(pfa, 2 * K)
        amp = rho / np.sqrt(K)
        ph = rng.uniform(0, 2 * np.pi, size=(n, K))
        z = amp * np.exp(1j * ph) + rng.standard_normal((n, K)) + 1j * rng.standard_normal((n, K))
        s1 = np.sum(np.abs(z) ** 2, axis=1)
        z0 = rng.standard_normal((n, K)) + 1j * rng.standard_normal((n, K))
        s0 = np.sum(np.abs(z0) ** 2, axis=1)
        print(f"P_FA={pfa:g} target P_D={pd}: K={K:2d} rho={rho}: MC P_D={np.mean(s1 > thr):.4f}, MC P_FA={np.mean(s0 > thr):.5f}")
