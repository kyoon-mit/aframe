"""Rebuild every saved chirplet fit from its parameters (numpy only, no torch).

Model (stated in the drafter's core.py docstring, re-implemented here):
K(l) = sum_k a_k exp(-(l - t_k)^2 / (2 s_k^2)) exp(i (2 pi f_k x / fs + pi c_k (x / fs)^2)), x = l - t_k.
theta = [t (M), log s (M), f (M), c (M)].
"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import glob, json, sys
import numpy as np

FS = 2048.0
SP = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad"
KER = np.load(SP + "/s4dmath/fits/kernels.npz")
L = 8064


def G_of(theta, M, dt=np.float64):
    t, ls, f, c = (theta[i * M:(i + 1) * M].astype(dt) for i in range(4))
    l = np.arange(L, dtype=dt)
    x = l[:, None] - t[None, :]
    s = np.exp(ls)
    pi = dt(np.pi)
    ph = 2 * pi * f[None, :] * x / dt(FS) + pi * c[None, :] * (x / dt(FS)) ** 2
    env = np.exp(-dt(0.5) * (x / s[None, :]) ** 2)
    return env * np.exp(1j * ph)


rows = []
for path in sorted(glob.glob(SP + "/oct16fix/chirplet/fits/runs*/*.npz")):
    js = json.load(open(path[:-4] + ".json"))
    d = np.load(path)
    M = int(js["M"])
    K = KER[f"K_{js['Mc']}"]
    th = d["theta"]
    G = G_of(th, M)
    a_ls, *_ = np.linalg.lstsq(G, K, rcond=None)
    Kh = G @ a_ls
    m_ls = abs(np.vdot(Kh, K)) / np.linalg.norm(Kh) / np.linalg.norm(K)
    Kh2 = G @ d["a"]
    m_saved_a = abs(np.vdot(Kh2, K)) / np.linalg.norm(Kh2) / np.linalg.norm(K)
    G32 = G_of(th, M, np.float32).astype(np.complex64)
    Kh32 = (G32 @ d["a"].astype(np.complex64)).astype(np.complex128)
    m32 = abs(np.vdot(Kh32, K)) / np.linalg.norm(Kh32) / np.linalg.norm(K)
    col = np.linalg.norm(G, axis=0)
    cancel = float(np.sum((np.abs(a_ls) * col) ** 2) / np.linalg.norm(Kh) ** 2)
    # start parameters, if saved
    m0 = None
    if "theta0" in d.files:
        G0 = G_of(d["theta0"], M)
        a0, *_ = np.linalg.lstsq(G0, K, rcond=None)
        m0 = float(np.linalg.norm(G0 @ a0) / np.linalg.norm(K))
    f = th[2 * M:3 * M]
    s = np.exp(th[M:2 * M])
    rows.append(dict(file=path.split("chirplet/")[1], Mc=js["Mc"], M=M, start=js["start"], match_json=js["match_final"],
                     match_ls=float(m_ls), match_saved_a=float(m_saved_a), match_c64=float(m32), cancel=cancel,
                     match_start_json=js.get("match_start"), match_start_rebuilt=m0,
                     n_pos=int((f > 0).sum()), n_neg=int((f < 0).sum()), fmin=float(np.abs(f).min()), fmax=float(np.abs(f).max()),
                     smin=float(s.min()), smax=float(s.max())))
    r = rows[-1]
    print(f"{r['file']:45s} Mc {r['Mc']} M {M:2d} {r['start']:9s} json {r['match_json']:.4f} lstsq {r['match_ls']:.4f} "
          f"savedA {r['match_saved_a']:.4f} c64 {r['match_c64']:.4f} cancel {cancel:.3f} start {m0 if m0 is None else round(m0,3)} "
          f"(json {r['match_start_json'] if r['match_start_json'] is None else round(r['match_start_json'],3)}) pos/neg {r['n_pos']}/{r['n_neg']} |f| {r['fmin']:.0f}-{r['fmax']:.0f} s {r['smin']:.0f}-{r['smax']:.0f}")
json.dump(rows, open(sys.argv[1], "w"), indent=1)
