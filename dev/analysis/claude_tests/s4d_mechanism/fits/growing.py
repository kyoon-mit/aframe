"""Raw (growing allowed) pencil poles, LS with columns scaled to unit norm; saves growing.json."""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"
import json
import numpy as np
import common as C

ker = np.load(C.OUT_DIR / "kernels.npz")
pp = np.load(C.OUT_DIR / "pencil_poles.npz")
lags = np.arange(C.L, dtype=np.float64)
out = {}
for Mc in C.MCS:
    K = ker[f"K_{Mc}"]
    out[str(Mc)] = {}
    for N in C.NS:
        z = pp[f"z_{Mc}_{N}"]
        lz = C.safe_log(z)
        logmax = np.maximum(0.0, (C.L - 1) * lz.real)
        Vn = np.exp(lags[:, None] * lz[None, :] - logmax[None, :])
        Vn /= np.linalg.norm(Vn, axis=0)
        c, _, rk, sv = np.linalg.lstsq(Vn, K, rcond=None)
        Kh = Vn @ c
        m = float(abs(np.vdot(Kh, K)) / np.linalg.norm(Kh))
        out[str(Mc)][str(N)] = {"match": m, "rank": int(rk), "cond": float(sv[0] / sv[-1]),
                                "max_abs_z": float(np.abs(z).max()),
                                "max_efolds_over_L": float(C.L * np.log(np.abs(z)).max())}
    print(Mc, {N: round(out[str(Mc)][str(N)]["match"], 4) for N in C.NS},
          {N: round(out[str(Mc)][str(N)]["max_abs_z"], 5) for N in C.NS}, flush=True)
json.dump(out, open(C.OUT_DIR / "growing.json", "w"), indent=1)
