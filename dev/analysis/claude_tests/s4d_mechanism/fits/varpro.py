"""Variable-projection fit of p decaying poles (|z| <= 1) to a whitened chirp kernel.

Parameters per pole: x = log(alpha + 1/L) with alpha >= 0 the decay per sample,
y = beta * L / (2 pi) with beta the frequency in rad per sample. Amplitudes are
the least-squares solution at every step (closed-form Gram matrix, Cholesky).
Objective f = ||K - V c||^2 = 1 - match^2 (unit-norm K).
"""

import argparse
import json
import os
import sys
import time

_thr = sys.argv[sys.argv.index("--threads") + 1] if "--threads" in sys.argv else "4"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _thr

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--mcs", type=str, default="1.0,1.2,1.5,2.0")
ap.add_argument("--starts", type=str, default="chirp,pencil_clipped,pencil_reflected")
ap.add_argument("--ps", type=str, default="8,16,32,64,128,256,512,1024")
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--maxiter", type=int, default=30000)
ap.add_argument("--wall", type=float, default=1800.0, help="seconds per fit")
ap.add_argument("--outdir", type=str, default="runs")
ap.add_argument("--gradcheck", action="store_true")
args = ap.parse_args()

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = str(args.threads)

import scipy.optimize as so  # noqa: E402
import torch  # noqa: E402

import common as C  # noqa: E402

torch.set_num_threads(args.threads)
DEV = torch.device(args.device)
L = C.L
A0 = 1.0 / L  # alpha offset in x = log(alpha + A0)
A_MAX = 1.0  # largest decay per sample
N_REFINE = 2  # corrected semi-normal equation steps
CK_SET = set(np.unique(np.round(np.sqrt(2.0) ** np.arange(0, 60)).astype(int)).tolist()) | set(range(500, 100001, 500))
OUT = C.OUT_DIR / args.outdir
OUT.mkdir(exist_ok=True)


def cexpm1(w):
    """exp(w) - 1 for complex w, accurate near 0."""
    x, y = w.real, w.imag
    re = torch.expm1(x) * torch.cos(y) - 2.0 * torch.sin(0.5 * y) ** 2
    im = torch.exp(x) * torch.sin(y)
    return torch.complex(re, im)


class Objective:
    def __init__(self, K):
        self.K = torch.tensor(K, dtype=torch.complex128, device=DEV)
        self.lag = torch.arange(L, dtype=torch.float64, device=DEV)
        self.nfev = 0
        self.n_jitter = 0
        self.last = None

    def unpack(self, theta):
        p = theta.size // 2
        x = torch.tensor(theta[:p], dtype=torch.float64, device=DEV)
        y = torch.tensor(theta[p:], dtype=torch.float64, device=DEV)
        alpha = torch.clamp(torch.exp(x) - A0, min=0.0)
        beta = 2.0 * np.pi * y / L
        return x, alpha, beta

    def solve(self, s):
        """V, c, r for log-poles s (complex, Re <= 0)."""
        V = torch.exp(self.lag[:, None] * s[None, :])
        w = s.conj()[:, None] + s[None, :]
        small = w.abs() < 1e-10
        den = torch.where(small, torch.ones_like(w), cexpm1(w))
        G = torch.where(small, L * (1.0 + 0.5 * (L - 1) * w), cexpm1(L * w) / den)
        d = 1.0 / torch.sqrt(G.diagonal().real)
        Gs = d[:, None] * G * d[None, :]
        b = (V.conj().T @ self.K) * d
        eye = torch.eye(G.shape[0], dtype=G.dtype, device=DEV)
        for jit in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
            Lc, info = torch.linalg.cholesky_ex(Gs + jit * eye if jit else Gs)
            if int(info) == 0:
                break
        if jit:
            self.n_jitter += 1
        cs = torch.cholesky_solve(b[:, None], Lc)[:, 0]
        c = cs * d
        r = self.K - V @ c
        for _ in range(N_REFINE):
            db = (V.conj().T @ r) * d
            c = c + torch.cholesky_solve(db[:, None], Lc)[:, 0] * d
            r = self.K - V @ c
        return V, c, r

    def __call__(self, theta):
        self.nfev += 1
        x, alpha, beta = self.unpack(theta)
        s = torch.complex(-alpha, beta)
        V, c, r = self.solve(s)
        f = float((r.abs() ** 2).sum())
        q = (self.lag * r.conj()) @ V
        g = c * q
        dfdx = 2.0 * g.real * torch.exp(x)
        dfdy = 2.0 * g.imag * (2.0 * np.pi / L)
        grad = torch.cat([dfdx, dfdy]).cpu().numpy()
        self.last = f
        return f, grad


def theta_from(alpha, beta):
    x = np.log(np.clip(alpha, 0.0, A_MAX) + A0)
    y = beta * L / (2 * np.pi)
    return np.concatenate([x, y])


def poles_from(theta):
    p = theta.size // 2
    alpha = np.clip(np.exp(theta[:p]) - A0, 0.0, None)
    beta = 2 * np.pi * theta[p:] / L
    beta = np.angle(np.exp(1j * beta))  # wrap to (-pi, pi]
    return alpha, beta


def chirp_start(trk, Mc, p, frac_pos):
    """One pole per band of equal int tau df; ring time = tau(f) at band centre."""
    n_pos = int(round(p * frac_pos))
    alloc = {"pos": n_pos, "neg": p - n_pos}
    alphas, betas = [], []
    for key, sign in (("pos", 1), ("neg", -1)):
        k = alloc[key]
        if k == 0:
            continue
        af, tau, n = trk[f"af_{Mc}_{key}"], trk[f"tau_{Mc}_{key}"], trk[f"n_{Mc}_{key}"]
        targets = (np.arange(k) + 0.5) / k * n[-1]
        idx = np.clip(np.searchsorted(n, targets), 0, af.size - 1)
        fk, tk = af[idx], tau[idx]
        alphas.append(C.DT / tk)
        betas.append(sign * 2 * np.pi * fk * C.DT)
    return np.concatenate(alphas), np.concatenate(betas), alloc


def pencil_start(pp, Mc, p, mode):
    z = pp[f"z_{Mc}_{p}"]
    z2, _ = C.reflect(z) if mode == "reflected" else C.clip(z)
    lz = C.safe_log(z2)
    return -lz.real, lz.imag


def match_of(alpha, beta, K):
    return C.lstsq_full(-alpha + 1j * beta, K)


def match_float32(alpha, beta, c, K):
    """Match when poles, amplitudes and the kernel sum are evaluated in complex64."""
    s = torch.tensor(-alpha + 1j * beta, dtype=torch.complex64)
    lag = torch.arange(L, dtype=torch.float32)
    V = torch.exp(lag[:, None] * s[None, :])
    Kh = (V @ torch.tensor(c, dtype=torch.complex64)).numpy().astype(np.complex128)
    return float(np.abs(np.vdot(Kh, K)) / (np.linalg.norm(Kh) * np.linalg.norm(K)))


def gradcheck(obj, theta, n=6, h=1e-6):
    f0, g0 = obj(theta)
    rng = np.random.default_rng(0)
    out = []
    for i in rng.choice(theta.size, n, replace=False):
        e = np.zeros_like(theta)
        e[i] = h
        fp, _ = obj(theta + e)
        fm, _ = obj(theta - e)
        out.append((int(i), float(g0[i]), float((fp - fm) / (2 * h))))
    return out


def run_fit(K, alpha0, beta0, tag, meta):
    obj = Objective(K)
    th0 = theta_from(alpha0, beta0)
    p = alpha0.size
    bounds = [(np.log(A0), np.log(A_MAX + A0))] * p + [(None, None)] * p
    f_start, _ = obj(th0)
    hist = [(0, 0, 1.0 - f_start)]
    t0 = time.time()
    state = {"it": 0}
    ck_iter, ck_theta = [0], [th0.copy()]

    def cb(intermediate_result):
        state["it"] += 1
        it = state["it"]
        if it % 25 == 0 or it < 10:
            hist.append((it, obj.nfev, 1.0 - float(intermediate_result.fun)))
        if it in CK_SET:
            ck_iter.append(it)
            ck_theta.append(np.array(intermediate_result.x, copy=True))
        if time.time() - t0 > args.wall:
            raise StopIteration

    res = so.minimize(obj, th0, jac=True, method="L-BFGS-B", bounds=bounds, callback=cb,
                      options={"maxiter": args.maxiter, "maxfun": 2 * args.maxiter,
                               "ftol": 1e-15, "gtol": 1e-11, "maxcor": 30})
    wall = time.time() - t0
    alpha, beta = poles_from(res.x)
    hist.append((state["it"], obj.nfev, 1.0 - float(res.fun)))
    ver = match_of(alpha, beta, K)
    ver0 = match_of(np.clip(alpha0, 0, A_MAX), beta0, K)
    out = dict(meta)
    out.update({
        "tag": tag, "p": int(p),
        "match_start_objective": float(np.sqrt(max(1.0 - f_start, 0.0))),
        "match_start_lstsq": ver0["match"],
        "match_final_objective": float(np.sqrt(max(1.0 - res.fun, 0.0))),
        "match_final_lstsq": ver["match"],
        "match_final_float32": match_float32(alpha, beta, ver["c"], K),
        "lstsq_rank": ver["rank"], "cond_V": ver["cond_V"], "cancel_ratio": ver["cancel_ratio"],
        "n_undamped": int(np.sum(alpha == 0.0)),
        "n_iter": int(state["it"]), "nfev": int(obj.nfev), "n_jitter": int(obj.n_jitter),
        "message": str(res.message), "wall_s": wall,
        "history_iter_nfev_matchsq": hist,
    })
    if ck_iter[-1] != state["it"]:
        ck_iter.append(state["it"])
        ck_theta.append(np.array(res.x, copy=True))
    np.savez(OUT / f"{tag}.npz", alpha=alpha, beta=beta, c=ver["c"], col_norm=ver["col_norm"],
             alpha0=alpha0, beta0=beta0, ck_iter=np.array(ck_iter), ck_theta=np.array(ck_theta))
    json.dump(out, open(OUT / f"{tag}.json", "w"), indent=1)
    print(f"{tag}: start {out['match_start_lstsq']:.4f} -> {out['match_final_lstsq']:.4f} "
          f"(obj {out['match_final_objective']:.4f}, f32 {out['match_final_float32']:.4f}) it {out['n_iter']} nfev {out['nfev']} "
          f"undamped {out['n_undamped']} cancel {out['cancel_ratio']:.2f} cond {out['cond_V']:.1e} "
          f"jit {out['n_jitter']} {wall:.0f}s [{out['message'][:40]}]", flush=True)
    return out


def main():
    ker = np.load(C.OUT_DIR / "kernels.npz")
    trk = np.load(C.OUT_DIR / "tracks.npz")
    pp = np.load(C.OUT_DIR / "pencil_poles.npz")
    tj = json.load(open(C.OUT_DIR / "tracks.json"))
    mcs = [float(m) for m in args.mcs.split(",")]
    ps = [int(p) for p in args.ps.split(",")]
    starts = args.starts.split(",")
    for Mc in mcs:
        K = ker[f"K_{Mc}"]
        frac_pos = tj[str(Mc)]["energy_frac"]["pos"]
        for p in ps:
            for st in starts:
                if st == "chirp":
                    a0, b0, alloc = chirp_start(trk, Mc, p, frac_pos)
                    meta = {"start": "chirp", "Mc": Mc, "alloc": alloc}
                else:
                    mode = "reflected" if "refl" in st else "clipped"
                    a0, b0 = pencil_start(pp, Mc, p, mode)
                    meta = {"start": f"pencil_{mode}", "Mc": Mc}
                tag = f"{meta['start']}_Mc{Mc}_p{p}"
                if args.gradcheck:
                    obj = Objective(K)
                    print(tag, "gradcheck (i, analytic, numeric):", gradcheck(obj, theta_from(a0, b0)), flush=True)
                    continue
                if (OUT / f"{tag}.json").exists():
                    print(tag, "exists, skip", flush=True)
                    continue
                run_fit(K, a0, b0, tag, meta)


if __name__ == "__main__":
    main()
