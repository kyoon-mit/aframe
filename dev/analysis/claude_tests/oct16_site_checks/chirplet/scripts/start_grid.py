"""Best match to each whitened kernel over the 256 bank chirp masses (log-spaced 0.85 to 2.75),
for chirplet chains of 12 laid along leading-order chirps with no fit (both signs, best lag)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import json
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import start_check_lib as S  # noqa: E402

KER = np.load(sys.argv[1])
grid = np.exp(np.linspace(np.log(0.85), np.log(2.75), 256))
print("grid spacing", grid[1] / grid[0] - 1)


def best_corr(A, B):
    n = 2 * S.L
    c = np.fft.ifft(np.conj(np.fft.fft(A, n)) * np.fft.fft(B, n))
    return float(np.max(np.abs(c)) / (np.linalg.norm(A) * np.linalg.norm(B)))


out = {}
for name in ("1.0", "1.2", "1.5", "2.0"):
    K = KER[f"K_{name}"]
    mc0 = float(name)
    near = grid[(grid > 0.95 * mc0) & (grid < 1.03 * mc0)]
    best = (0, None)
    for m in near:
        k = S.chain_atoms(m, 12).sum(1)
        for kk in (k, np.conj(k)):
            v = best_corr(kk, K)
            if v > best[0]:
                best = (v, float(m))
    out[name] = {"match": best[0], "mc": best[1]}
    print(name, round(best[0], 4), round(best[1], 4), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
