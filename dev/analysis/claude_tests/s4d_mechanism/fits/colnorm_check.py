"""Raw (growing allowed) pencil poles: LS match with unit-norm columns vs plain lstsq."""
import numpy as np
import common as C

ker = np.load("kernels.npz")
pp = np.load("pencil_poles.npz")
lags = np.arange(C.L, dtype=np.float64)
for Mc in C.MCS:
    K = ker[f"K_{Mc}"]
    row = []
    for N in C.NS:
        z = pp[f"z_{Mc}_{N}"]
        lz = C.safe_log(z)
        # column l2 norm in log domain, then normalized columns
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            V = np.exp(lags[:, None] * lz[None, :])
        # normalize each column by its max magnitude (exp(max(0, L-1)*Re lz)) to avoid overflow
        logmax = np.maximum(0.0, (C.L - 1) * lz.real)
        Vn = np.exp(lags[:, None] * lz[None, :] - logmax[None, :])
        Vn /= np.linalg.norm(Vn, axis=0)
        c, _, rk, sv = np.linalg.lstsq(Vn, K, rcond=None)
        Kh = Vn @ c
        m = abs(np.vdot(Kh, K)) / np.linalg.norm(Kh)
        row.append((N, round(float(m), 4), rk, f"{sv[0]/sv[-1]:.1e}"))
    print(Mc, row, flush=True)
