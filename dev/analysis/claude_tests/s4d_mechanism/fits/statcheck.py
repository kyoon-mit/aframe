"""Projected gradient of the VarPro objective at the final poles of each fit (p <= 128)."""
import glob, json, os, sys
sys.argv = ["varpro.py", "--device", "cpu", "--threads", "2"]
exec(open("varpro.py").read().split("def main():")[0])
ker = np.load("kernels.npz")
out = {}
for fn in sorted(glob.glob("runs/*_p*.json")):
    if fn.endswith(".path.json"):
        continue
    rj = json.load(open(fn))
    if rj["p"] > 128:
        continue
    d = np.load(fn.replace(".json", ".npz"))
    th = theta_from(d["alpha"], d["beta"])
    obj = Objective(ker[f"K_{rj['Mc']}"])
    f, g = obj(th)
    p = rj["p"]
    lo = np.log(A0)
    pg = g.copy()
    at_lo = np.isclose(th[:p], lo, atol=1e-12)
    pg[:p][at_lo & (g[:p] > 0)] = 0.0
    out[rj["tag"]] = {"f": f, "max_abs_proj_grad": float(np.abs(pg).max()), "n_at_lower_bound": int(at_lo.sum())}
    print(f"{rj['tag']:32s} 1-match^2 {f:.6f} max|proj grad| {np.abs(pg).max():.2e} at alpha=0: {int(at_lo.sum())}")
json.dump(out, open("statcheck.json", "w"), indent=1)
