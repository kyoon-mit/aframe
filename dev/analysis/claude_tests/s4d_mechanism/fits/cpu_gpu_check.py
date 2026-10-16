"""Same random start (Mc 2.0, p=64, seeds 0-1) fitted on CPU, to compare with the GPU restart runs."""
import json, os, sys
sys.argv = ["varpro.py", "--device", "cpu", "--threads", "2", "--maxiter", "30000", "--wall", "900", "--outdir", "cpu_gpu_check"]
exec(open("varpro.py").read().split("def main():")[0])


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


ker = np.load(C.OUT_DIR / "kernels.npz"); tj = json.load(open(C.OUT_DIR / "tracks.json"))
Mc, p = 2.0, 64
for seed in (0, 1):
    rng = np.random.default_rng(1000 * p + seed + int(10 * Mc) * 100000)
    a0, b0 = random_start(rng, p, tj, Mc)
    run_fit(ker[f"K_{Mc}"], a0, b0, f"random{seed}_Mc{Mc}_p{p}_cpu", {"start": f"random{seed}", "Mc": Mc})
