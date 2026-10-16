"""Random restarts of the VarPro fit at small p: is the best of the 3 ordered starts near the best reachable?

Random start: |f| log-uniform over the 0.1-99.9% energy band, sign drawn with the kernel's
energy fraction per sign, ring time log-uniform in [10 ms, 10 s]. Seeded.
"""

import json
import os
import sys

P_LIST = [int(x) for x in os.environ.get("RESTART_PS", "32").split(",")]
N_RESTART = int(os.environ.get("N_RESTART", "16"))
MCS_RUN = [float(x) for x in os.environ.get("RESTART_MCS", "1.0,1.2,1.5,2.0").split(",")]
sys.argv = ["varpro.py", "--device", os.environ.get("RESTART_DEVICE", "cpu"), "--threads", os.environ.get("RESTART_THREADS", "1"), "--maxiter", "30000", "--wall", "600",
            "--outdir", "restarts"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "varpro.py")).read().split("def main():")[0])


def random_start(rng, p, tj, Mc):
    m = str(Mc)
    fp = tj[m]["energy_frac"]["pos"]
    sign = np.where(rng.random(p) < fp, 1.0, -1.0)
    f = np.empty(p)
    for s, key in ((1.0, "pos"), (-1.0, "neg")):
        lo, hi = tj[m][key]["band_hz"]
        k = sign == s
        f[k] = np.exp(rng.uniform(np.log(lo), np.log(hi), k.sum()))
    ring = np.exp(rng.uniform(np.log(0.01), np.log(10.0), p))
    return C.DT / ring, sign * 2 * np.pi * f * C.DT


def main():
    ker = np.load(C.OUT_DIR / "kernels.npz")
    tj = json.load(open(C.OUT_DIR / "tracks.json"))
    summary = {}
    for Mc in MCS_RUN:
        K = ker[f"K_{Mc}"]
        for p in P_LIST:
            res = []
            for seed in range(int(os.environ.get("SEED0", "0")), int(os.environ.get("SEED0", "0")) + N_RESTART):
                rng = np.random.default_rng(1000 * p + seed + int(10 * Mc) * 100000)
                a0, b0 = random_start(rng, p, tj, Mc)
                tag = f"random{seed}_Mc{Mc}_p{p}"
                if (OUT / f"{tag}.json").exists():
                    out = json.load(open(OUT / f"{tag}.json"))
                else:
                    out = run_fit(K, a0, b0, tag, {"start": f"random{seed}", "Mc": Mc})
                res.append({"seed": seed, "f64": out["match_final_lstsq"], "f32": out["match_final_float32"],
                            "cancel": out["cancel_ratio"], "stop": out["message"], "n_iter": out["n_iter"]})
            f32 = np.array([r["f32"] for r in res])
            summary[f"{Mc}_{p}"] = {"runs": res, "max_f32": float(f32.max()), "median_f32": float(np.median(f32)),
                                    "min_f32": float(f32.min())}
            print(f"Mc {Mc} p {p}: random restarts f32 max {f32.max():.4f} median {np.median(f32):.4f} min {f32.min():.4f}",
                  flush=True)
    json.dump(summary, open(C.OUT_DIR / f"restarts_summary_{'_'.join(str(m) for m in MCS_RUN)}_p{'_'.join(str(p) for p in P_LIST)}_s{os.environ.get('SEED0', '0')}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
