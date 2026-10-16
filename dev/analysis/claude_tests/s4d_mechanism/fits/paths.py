"""Evaluate every checkpoint of every fit: float64 lstsq match, complex64 match, cancellation, pole spacing."""

import glob
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = os.environ.get("PATHS_THREADS", "3")
sys.argv = ["varpro.py", "--device", "cpu", "--threads", os.environ.get("PATHS_THREADS", "3")]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "varpro.py")).read().split("def main():")[0])

MAX_CK = 24  # checkpoints evaluated per run (log-spaced subset when more)


def min_sep_over_width(alpha, beta):
    """Smallest |s_n - s_m| over the larger decay (+1/L) of the pair."""
    s = -alpha + 1j * beta
    p = s.size
    best = np.inf
    for i0 in range(0, p, 512):
        D = np.abs(s[i0:i0 + 512, None] - s[None, :])
        W = np.maximum(alpha[i0:i0 + 512, None], alpha[None, :]) + 1.0 / L
        R = D / W
        idx = np.arange(i0, min(i0 + 512, p))
        R[idx - i0, idx] = np.inf
        best = min(best, float(R.min()))
    return best


def main():
    pattern = sys.argv_extra if hasattr(sys, "argv_extra") else os.environ.get("PATHS_GLOB", "runs/*.npz")
    files = sorted(f for f in glob.glob(str(C.OUT_DIR / pattern)) if not f.endswith(".bestf32.npz"))
    ker = np.load(C.OUT_DIR / "kernels.npz")
    for fn in files:
        out_fn = fn.replace(".npz", ".path.json")
        if os.path.exists(out_fn) and os.path.getmtime(out_fn) > os.path.getmtime(fn):
            continue
        meta = json.load(open(fn.replace(".npz", ".json")))
        K = ker[f"K_{meta['Mc']}"]
        d = np.load(fn)
        its, ths = d["ck_iter"], d["ck_theta"]
        sel = np.arange(its.size)
        if its.size > MAX_CK:
            sel = np.unique(np.round(np.geomspace(1, its.size, MAX_CK)).astype(int) - 1)
            sel = np.unique(np.concatenate([[0], sel, [its.size - 1]]))
        rows = []
        for k in sel:
            a, b = poles_from(ths[k])
            v = match_of(a, b, K)
            rows.append({
                "iter": int(its[k]),
                "match_f64": v["match"],
                "match_f32": match_float32(a, b, v["c"], K),
                "cancel_ratio": v["cancel_ratio"],
                "cond_V": v["cond_V"],
                "min_sep_over_width": min_sep_over_width(a, b),
                "n_undamped": int(np.sum(a < 1e-12)),
            })
        best = max(range(len(rows)), key=lambda i: rows[i]["match_f32"])
        res = {"tag": meta["tag"], "Mc": meta["Mc"], "p": meta["p"], "start": meta["start"],
               "path": rows, "best_f32_index": best, "best_f32": rows[best],
               "final": rows[-1], "first": rows[0]}
        a, b = poles_from(ths[sel[best]])
        np.savez(fn.replace(".npz", ".bestf32.npz"), alpha=a, beta=b)
        json.dump(res, open(out_fn, "w"), indent=1)
        print(f"{meta['tag']}: start f64 {rows[0]['match_f64']:.4f} | final f64 {rows[-1]['match_f64']:.4f} "
              f"f32 {rows[-1]['match_f32']:.4f} cancel {rows[-1]['cancel_ratio']:.1e} | best f32 "
              f"{rows[best]['match_f32']:.4f} at it {rows[best]['iter']} (f64 {rows[best]['match_f64']:.4f}, "
              f"cancel {rows[best]['cancel_ratio']:.1e}, minsep {rows[best]['min_sep_over_width']:.2e})", flush=True)


if __name__ == "__main__":
    main()
