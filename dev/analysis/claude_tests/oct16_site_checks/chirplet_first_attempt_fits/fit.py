"""Fit M chirplets to one kernel by variable projection and L-BFGS-B.

Starts:
  track_n      M points uniform in the quarter-cycle tiling count n(l) of the track
  track_e      M points at equal steps of the sign's energy, from the merger
  track_n_jJ   track_n with seeded jitter J
  random_J     t ~ U(0, L), |f| ~ U(20, 800) Hz, c = 0, log s ~ U(log 32, log 1024)
  *_dom        all chirplets on the dominant sign (default: split by sign energy)
"""

import argparse
import json
import time

import numpy as np
import scipy.optimize as so

import core as C

FS, L = C.FS, C.L
TRK = None
PLATEAU_EVERY = 100     # iterations between plateau checks
PLATEAU_WINDOW = 500    # stop a fit when the match gained less than PLATEAU_DMATCH over this many iterations
PLATEAU_DMATCH = 1e-6


def track():
    global TRK
    if TRK is None:
        TRK = (np.load(C.HERE / "track.npz"), json.load(open(C.HERE / "track.json")))
    return TRK


def allocation(Mc, M, dom_only):
    npz, tj = track()
    t = tj[str(Mc)]
    dom = t["dominant_sign"]
    e = t["energy_by_sign"]
    frac_dom = (e["pos"] if dom > 0 else e["neg"]) / (e["pos"] + e["neg"])
    n_dom = M if dom_only else max(1, int(round(M * frac_dom)))
    return {dom: n_dom, -dom: M - n_dom}, dom


def track_points(Mc, sign, dom, n, rule):
    """t, f, c, s for n chirplets on one sign along the dominant track (mirrored for the other sign)."""
    npz, _ = track()
    lag = np.arange(L)
    grid = npz[f"grid_{Mc}"]
    if rule == "n":
        nq = npz[f"n_q_{Mc}"]
        targets = (np.arange(n) + 0.5) / n * nq[-1]
        t = np.interp(targets, nq, lag)
    else:
        A2 = npz[f"A2_{'pos' if sign > 0 else 'neg'}_{Mc}"]
        cum = np.cumsum(A2) / A2.sum()
        targets = (np.arange(n) + 0.5) / n
        t = np.interp(targets, cum, lag)
    mirror = sign / dom
    f = mirror * np.interp(t, grid, npz[f"f_{Mc}"])
    c = mirror * np.interp(t, grid, npz[f"fdot_{Mc}"])
    s = np.clip(0.5 * np.interp(t, grid, npz[f"Tq_{Mc}"]), 2.0, 4.0 * L)
    return t, f, c, s


def start_theta(Mc, M, start):
    dom_only = start.endswith("_dom")
    base = start[:-4] if dom_only else start
    alloc, dom = allocation(Mc, M, dom_only)
    parts = []
    if base.startswith("random"):
        seed = int(base.split("_")[1])
        rng = np.random.default_rng(1000 * seed + M + int(round(100 * Mc)))
        for sign, n in alloc.items():
            if n == 0:
                continue
            t = rng.uniform(0, L - 1, n)
            f = sign * rng.uniform(20.0, 800.0, n)
            c = np.zeros(n)
            s = np.exp(rng.uniform(np.log(32.0), np.log(1024.0), n))
            parts.append((t, f, c, s))
    else:
        rule = "e" if base == "track_e" else "n"
        for sign, n in alloc.items():
            if n == 0:
                continue
            parts.append(track_points(Mc, sign, dom, n, rule))
        if "_j" in base:
            seed = int(base.split("_j")[1])
            rng = np.random.default_rng(7919 * seed + M + int(round(100 * Mc)))
            out = []
            for t, f, c, s in parts:
                n = t.size
                t = np.clip(t + rng.normal(0, 0.25, n) * s, 0, L - 1)
                f = f + rng.normal(0, 0.1, n) * FS / (2 * np.pi * s)
                c = c * (1 + rng.normal(0, 0.1, n))
                s = np.clip(s * np.exp(rng.normal(0, 0.2, n)), 2.0, 4.0 * L)
                out.append((t, f, c, s))
            parts = out
    t, f, c, s = (np.concatenate([p[i] for p in parts]) for i in range(4))
    theta = np.concatenate([t, np.log(s), np.clip(f, -C.F_MAX, C.F_MAX), c])
    return theta, {int(k): int(v) for k, v in alloc.items()}


def scales(theta, M):
    s = np.exp(theta[M:2 * M])
    return np.concatenate([s, np.ones(M), FS / (2 * np.pi * s), FS**2 / (np.pi * s**2)])


def optimize(K, M, theta0, maxiter_cycle, n_cycles, wall):
    obj = C.Objective(K, M)
    bnd = C.bounds_for(M)
    theta = theta0.copy()
    f_start, _ = obj.value_grad(theta)
    hist = [(0, 0, f_start)]
    ckpt = [theta.copy()]
    t0 = time.time()
    it_total = 0
    msgs = []
    f_prev = f_start
    for cyc in range(n_cycles):
        sc = scales(theta, M)
        ref = theta.copy()
        zb = []
        for (lo, hi), r, s_ in zip(bnd, ref, sc):
            zb.append((None if lo is None else (lo - r) / s_, None if hi is None else (hi - r) / s_))

        def fun(z):
            fv, g = obj.value_grad(ref + z * sc)
            return fv, g * sc

        state = {"it": 0, "x": np.zeros_like(theta), "f": f_prev, "wall": False, "plateau": False, "trace": []}

        def cb(intermediate_result):
            state["it"] += 1
            state["x"] = np.array(intermediate_result.x, copy=True)
            state["f"] = float(intermediate_result.fun)
            if state["it"] % 50 == 0:
                hist.append((it_total + state["it"], obj.nfev, state["f"]))
            if state["it"] % PLATEAU_EVERY == 0:
                state["trace"].append(state["f"])
                n_back = PLATEAU_WINDOW // PLATEAU_EVERY
                if len(state["trace"]) > n_back:
                    m_now = np.sqrt(max(1.0 - state["trace"][-1], 0.0))
                    m_then = np.sqrt(max(1.0 - state["trace"][-1 - n_back], 0.0))
                    if m_now - m_then < PLATEAU_DMATCH:
                        state["plateau"] = True
                        raise StopIteration
            if time.time() - t0 > wall:
                state["wall"] = True
                raise StopIteration

        try:
            res = so.minimize(fun, np.zeros_like(theta), jac=True, method="L-BFGS-B", bounds=zb,
                              callback=cb, options={"maxiter": maxiter_cycle, "maxfun": 2 * maxiter_cycle,
                                                    "ftol": 1e-12, "gtol": 1e-10, "maxcor": 30})
            z, fv, msg = res.x, float(res.fun), str(res.message)
        except StopIteration:
            z, fv, msg = state["x"], state["f"], "stopped"
        if state["wall"]:
            msg = "wall: " + msg
        if state["plateau"]:
            msg = "plateau: " + msg
        it_total += state["it"]
        theta = ref + z * sc
        ckpt.append(theta.copy())
        hist.append((it_total, obj.nfev, fv))
        msgs.append(msg[:60])
        rel = (f_prev - fv) / max(f_prev, 1e-300)
        f_prev = fv
        if rel < 1e-9 or state["wall"] or state["plateau"] or time.time() - t0 > wall:
            break
    return theta, f_start, hist, ckpt, it_total, obj.nfev, obj.n_jitter, msgs, time.time() - t0


def run_job(job, outdir):
    Mc, M, start = job["Mc"], job["M"], job["start"]
    tag = f"Mc{Mc}_M{M}_{start}"
    out_json = outdir / f"{tag}.json"
    if out_json.exists():
        return json.load(open(out_json))
    K = C.load_kernels()[Mc]
    theta0, alloc = start_theta(Mc, M, start)
    theta, f_start, hist, ckpt, nit, nfev, njit, msgs, wall = optimize(
        K, M, theta0, job.get("maxiter_cycle", 5000), job.get("n_cycles", 6), job.get("wall", 1800.0))
    obj = C.Objective(K, M)
    det = obj.details(theta)
    det0 = obj.details(theta0)
    m32 = C.match_float32(theta, det["a"], K, M)
    # float32 match at each checkpoint, to find the best float32-valid point on the path
    path = []
    for th in ckpt:
        d = obj.details(th)
        path.append((d["match"], C.match_float32(th, d["a"], K, M), d["cancel_ratio"]))
    t, logs, f, c = C.unpack(theta, M)
    rec = {
        "Mc": Mc, "M": M, "start": start, "alloc": alloc,
        "match_start": det0["match"], "match_final": det["match"], "match_final_float32": m32,
        "match_from_objective": float(np.sqrt(max(1.0 - hist[-1][2], 0.0))),
        "cancel_ratio": det["cancel_ratio"], "cond_colnorm_G": det["cond_colnorm_G"],
        "n_iter": int(nit), "nfev": int(nfev), "n_jitter": int(njit), "messages": msgs, "wall_s": wall,
        "path_match_f64_f32_cancel": path, "history_iter_nfev_f": hist,
        "chirplets": [
            {"t_samples": float(t[k]), "s_samples": float(np.exp(logs[k])), "f_hz": float(f[k]), "c_hz_per_s": float(c[k]),
             "a_re": float(det["a"][k].real), "a_im": float(det["a"][k].imag),
             "energy_share": float((abs(det["a"][k]) * det["col_norm"][k]) ** 2 / np.sum((np.abs(det["a"]) * det["col_norm"]) ** 2))}
            for k in np.argsort(t)
        ],
    }
    np.savez(outdir / f"{tag}.npz", theta=theta, theta0=theta0, a=det["a"], ckpt=np.array(ckpt))
    json.dump(rec, open(out_json, "w"), indent=1)
    print(f"{tag}: {rec['match_start']:.4f} -> {rec['match_final']:.4f} (f32 {m32:.4f}) it {nit} nfev {nfev} "
          f"cancel {rec['cancel_ratio']:.2f} cond {rec['cond_colnorm_G']:.1e} jit {njit} {wall:.0f}s {msgs[-1] if msgs else ''}",
          flush=True)
    return rec


def gradcheck(Mc, M, start, n=8, h=1e-6):
    K = C.load_kernels()[Mc]
    theta, _ = start_theta(Mc, M, start)
    obj = C.Objective(K, M)
    f0, g0 = obj.value_grad(theta)
    sc = scales(theta, M)
    rng = np.random.default_rng(0)
    out = []
    for i in rng.choice(theta.size, n, replace=False):
        e = np.zeros_like(theta)
        e[i] = h * sc[i]
        fp, _ = obj.value_grad(theta + e)
        fm, _ = obj.value_grad(theta - e)
        out.append((int(i), float(g0[i]), float((fp - fm) / (2 * h * sc[i]))))
    return f0, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", type=float, default=1.2)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--start", type=str, default="track_n")
    ap.add_argument("--outdir", type=str, default="runs_test")
    ap.add_argument("--wall", type=float, default=600.0)
    ap.add_argument("--gradcheck", action="store_true")
    args = ap.parse_args()
    if args.gradcheck:
        f0, out = gradcheck(args.mc, args.M, args.start)
        print("f0", f0)
        for i, ga, gn in out:
            print(f"  i={i} analytic {ga:.6e} numeric {gn:.6e} rel {abs(ga - gn) / max(abs(gn), 1e-300):.2e}")
    else:
        od = C.HERE / args.outdir
        od.mkdir(exist_ok=True)
        run_job({"Mc": args.mc, "M": args.M, "start": args.start, "wall": args.wall}, od)
