"""Check the fit code's exact gradient against my own finite differences of my own
variable-projection objective (numpy), at the saved start along the track and at a random start."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
os.environ["CHIRPLET_THREADS"] = "1"
import sys
import numpy as np

SP = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad"
sys.path.insert(0, SP + "/oct16fix/chirplet/fits")
import core as C  # drafter's exact-gradient code

KER = np.load(SP + "/s4dmath/fits/kernels.npz")
FS, L = 2048.0, 8064
lag = np.arange(L, dtype=float)


def fobj(th, M, K):
    t, ls, f, c = (th[i * M:(i + 1) * M] for i in range(4))
    x = lag[:, None] - t[None, :]
    G = np.exp(-0.5 * (x / np.exp(ls)[None, :]) ** 2) * np.exp(1j * (2 * np.pi * f[None, :] * x / FS + np.pi * c[None, :] * (x / FS) ** 2))
    a, *_ = np.linalg.lstsq(G, K, rcond=None)
    r = K - G @ a
    return float(np.vdot(r, r).real)


for name, which in (("Mc1.2_M8_track_e", "theta0"), ("Mc1.2_M8_track_e", "theta"), ("Mc1.2_M8_random_1", "theta0")):
    sub = "runs_persist"
    d = np.load(f"{SP}/oct16fix/chirplet/fits/{sub}/{name}.npz")
    M = 8
    K = KER["K_1.2"]
    th = d[which].astype(float)
    obj = C.Objective(K, M)
    fv, g = obj.value_grad(th)
    f0 = fobj(th, M, K)
    scale = np.concatenate([np.full(M, 1.0), np.full(M, 1e-3), np.full(M, 1e-3), np.full(M, 1e-1)])
    gfd = np.empty_like(g)
    for i in range(th.size):
        e = np.zeros_like(th)
        e[i] = scale[i] * 1e-3
        gfd[i] = (fobj(th + e, M, K) - fobj(th - e, M, K)) / (2 * e[i])
    rel = np.abs(g - gfd) / np.maximum(np.abs(g), 1e-300)
    big = np.abs(g) > 1e-3 * np.abs(g).max()
    print(f"{name} {which}: f code {fv:.10f} mine {f0:.10f} (match {np.sqrt(max(0,1-f0)):.4f}); |g| range {np.abs(g).min():.2e}..{np.abs(g).max():.2e}; "
          f"rel diff median {np.median(rel):.1e}, max over components >1e-3 of max {rel[big].max():.1e}, overall max {rel.max():.1e}")
