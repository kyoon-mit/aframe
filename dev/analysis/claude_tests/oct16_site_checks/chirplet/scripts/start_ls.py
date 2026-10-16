"""Leading-order chirplet layout with amplitudes solved by least squares against each whitened kernel.

Same layout as start_check.py (P chirplets, width 0.4 of the spacing, 20 to 1024 Hz) at the chirp mass
and frequency sign found there; the chain is shifted by the lag of the best cross-correlation, and the
complex amplitudes are the least-squares solution. Match: fraction of the kernel's norm captured.
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import start_check_lib as S  # noqa: E402

KER = np.load(sys.argv[1])
best = json.load(open(sys.argv[2]))
out = {}
for name in ("1.0", "1.2", "1.5", "2.0"):
    K = KER[f"K_{name}"]
    b = best[name]["chain12"]
    res = {}
    for P in (8, 10, 12, 16):
        G = S.chain_atoms(b["mc_best"], P)
        if b["sign"] == "-":
            G = np.conj(G)
        k = G.sum(1)
        n = 2 * S.L
        c = np.fft.ifft(np.conj(np.fft.fft(k, n)) * np.fft.fft(K, n))
        j = int(np.argmax(np.abs(c)))
        lag = j if j < S.L else j - n
        Gs = np.roll(G, lag, axis=0)
        if lag > 0:
            Gs[:lag] = 0
        elif lag < 0:
            Gs[lag:] = 0
        a, *_ = np.linalg.lstsq(Gs, K, rcond=None)
        res[P] = {"lag": lag, "match_ls": float(np.linalg.norm(Gs @ a) / np.linalg.norm(K)),
                  "match_direct": float(abs(np.vdot(Gs.sum(1), K)) / np.linalg.norm(Gs.sum(1)))}
    out[name] = res
    print(name, {P: (r["lag"], round(r["match_direct"], 3), round(r["match_ls"], 3)) for P, r in res.items()}, flush=True)
json.dump(out, open(sys.argv[3], "w"), indent=1)
