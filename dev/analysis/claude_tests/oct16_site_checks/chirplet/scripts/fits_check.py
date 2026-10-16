"""Re-evaluate saved chirplet fits to whitened kernels with independent synthesis code.

For each saved fit (theta = [t, log s, f, c], amplitudes a): rebuild the kernel in
float64 and complex64, compute the match to the stored kernel, the cancellation
ratio (summed energy of the terms over the energy of their sum), and the amplitudes
re-solved by least squares (numpy lstsq) for the saved nonlinear parameters.
"""
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"

import glob
import json
import sys

import numpy as np

FS = 2048.0
KER = np.load(sys.argv[1])
OUT = sys.argv[2]
DIRS = sys.argv[3:]


def atoms(theta, M, L, dtype=np.float64):
    t, logs, f, c = (theta[i * M:(i + 1) * M].astype(dtype) for i in range(4))
    lag = np.arange(L, dtype=dtype)[:, None]
    x = lag - t[None, :]
    s = np.exp(logs)[None, :]
    ph = 2 * np.pi * f[None, :] * x / FS + np.pi * c[None, :] * (x / FS) ** 2
    return np.exp(-0.5 * (x / s) ** 2) * np.exp(1j * ph)


def match(a, b):
    return float(abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b)))


rows = []
for d in DIRS:
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        r = json.load(open(p))
        z = np.load(p[:-5] + ".npz")
        Mc, M = r["Mc"], r["M"]
        K = KER[f"K_{Mc}"]
        L = K.size
        G = atoms(z["theta"], M, L)
        a = z["a"]
        Kh = G @ a
        m64 = match(Kh, K)
        G32 = atoms(z["theta"], M, L, np.float32).astype(np.complex64)
        m32 = match((G32 @ a.astype(np.complex64)).astype(np.complex128), K)
        a2, *_ = np.linalg.lstsq(G, K, rcond=None)
        m_ls = match(G @ a2, K)
        col = np.linalg.norm(G, axis=0)
        cancel = float(np.sum((np.abs(a) * col) ** 2) / np.linalg.norm(Kh) ** 2)
        G0 = atoms(z["theta0"], M, L)
        a0, *_ = np.linalg.lstsq(G0, K, rcond=None)
        m0 = match(G0 @ a0, K)
        f = z["theta"][2 * M:3 * M]
        row = {"file": os.path.relpath(p, os.path.dirname(d)), "Mc": Mc, "M": M, "start": r["start"],
               "reported_final": r["match_final"], "reported_f32": r["match_final_float32"],
               "reported_start": r["match_start"], "m64": m64, "m32": m32, "m_ls_resolved": m_ls,
               "m_start_ls": m0, "cancel": cancel, "f_range": [float(np.min(np.abs(f))), float(np.max(np.abs(f)))]}
        rows.append(row)
        print(f"{row['file']:42s} M {M:3d} {r['start']:9s} start {m0:.4f} (rep {r['match_start']:.4f}) "
              f"final f64 {m64:.4f} f32 {m32:.4f} lstsq {m_ls:.4f} (rep {r['match_final']:.4f}/{r['match_final_float32']:.4f}) "
              f"cancel {cancel:.2f} |f| {row['f_range'][0]:.0f}-{row['f_range'][1]:.0f} Hz")
json.dump(rows, open(OUT, "w"), indent=1)
