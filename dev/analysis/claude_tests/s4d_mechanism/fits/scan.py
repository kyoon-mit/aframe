"""Chirp-following placement without optimization: match vs p on a fine grid (float64 lstsq and float32)."""

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"
sys.argv = ["varpro.py", "--device", "cpu", "--threads", "3"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "varpro.py")).read().split("def main():")[0])

PS = [16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048]


def main():
    ker = np.load(C.OUT_DIR / "kernels.npz")
    trk = np.load(C.OUT_DIR / "tracks.npz")
    tj = json.load(open(C.OUT_DIR / "tracks.json"))
    out = {}
    for Mc in C.MCS:
        K = ker[f"K_{Mc}"]
        fp = tj[str(Mc)]["energy_frac"]["pos"]
        ntb = {k: tj[str(Mc)][k]["int_tau_df"] for k in ("pos", "neg")}
        rows = []
        for p in PS:
            a0, b0, alloc = chirp_start(trk, Mc, p, fp)
            v = match_of(a0, b0, K)
            f32 = match_float32(a0, b0, v["c"], K)
            ratio = {k: alloc[k] / ntb[k] for k in alloc}
            rows.append({"p": p, "alloc": alloc, "p_over_int_tau_df": ratio, "match": v["match"],
                         "match_float32": f32, "cancel_ratio": v["cancel_ratio"], "cond_V": v["cond_V"],
                         "rank": v["rank"]})
            print(f"Mc {Mc} p {p:5d} alloc {alloc} p/int(tau df) {ratio} match {v['match']:.4f} "
                  f"f32 {f32:.4f} cancel {v['cancel_ratio']:.2f} rank {v['rank']}", flush=True)
        out[str(Mc)] = rows
    json.dump(out, open(C.OUT_DIR / "scan_chirp_placement.json", "w"), indent=1)


if __name__ == "__main__":
    main()
