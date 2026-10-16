"""Draw the T2 (segment order) and T3 (segment phase) manipulations per window, repeating draws
until the window's in-step match with its own original is below 0.5 (plan.json)."""
import json
import os

import numpy as np
import torch

import mcommon as mc

P = mc.PLAN["manipulation"]
THR, MAXD = P["match_threshold"], P["max_draws"]
X, true, merger, labels = mc.load_clean()
n = len(X)
Xt = torch.as_tensor(X)
out = {}


def matches(fn, params, idx, chunk=500):
    res = np.empty(len(idx))
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        x = Xt[ii].to(mc.DEV).double()
        m, _ = mc.instep_match(fn(x, params[s:s + chunk]), x)
        res[s:s + chunk] = m.cpu().numpy()
    return res


# taper-only control
mt = np.concatenate([mc.instep_match(mc.taper_only(Xt[s:s + 500].to(mc.DEV).double()),
                                     Xt[s:s + 500].to(mc.DEV).double())[0].cpu().numpy()
                     for s in range(0, n, 500)])
np.save(os.path.join(mc.HERE, "match_taper.npy"), mt)
out["taper_only_match"] = {"min": float(mt.min()), "median": float(np.median(mt)), "max": float(mt.max())}
mc.log("taper", out["taper_only_match"])

for test_id, name, fn in [(2, "T2", mc.shuffle), (3, "T3", mc.phase)]:
    cand = []
    for i in range(n):
        rng = np.random.default_rng([test_id, i])
        if test_id == 2:
            cand.append(np.stack([rng.permutation(8) for _ in range(MAXD)]))
        else:
            cand.append(rng.uniform(0, 2 * np.pi, size=(MAXD, 8)))
    cand = np.stack(cand)                                   # (n, MAXD, 8)
    chosen = np.full(n, -1)
    best_m = np.full(n, np.inf)
    best_d = np.zeros(n, int)
    first_m = None
    for r in range(MAXD):
        idx = np.where(chosen < 0)[0]
        if len(idx) == 0:
            break
        m = matches(fn, cand[idx, r], idx)
        if r == 0:
            first_m = m.copy()
        better = m < best_m[idx]
        best_m[idx[better]] = m[better]
        best_d[idx[better]] = r
        ok = m < THR
        chosen[idx[ok]] = r
        best_m[idx[ok]] = m[ok]
        best_d[idx[ok]] = r
        if r in (0, 1, 2, 4, 9, 19, 49, 99):
            mc.log(name, f"round {r}: still failing {int((chosen < 0).sum())}")
    passed = chosen >= 0
    use = np.where(passed, chosen, best_d)
    params = cand[np.arange(n), use]
    np.savez(os.path.join(mc.HERE, f"manip_{name}.npz"), params=params, draw=use, passed=passed,
             match=best_m, first_params=cand[:, 0], first_match=first_m)
    out[name] = {
        "pass_share": float(passed.mean()), "n_failed": int((~passed).sum()),
        "draws_needed_median_mean_max": [float(np.median(use[passed] + 1)), float(np.mean(use[passed] + 1)),
                                         int(use[passed].max() + 1)],
        "match_used_median_max": [float(np.median(best_m)), float(best_m.max())],
        "match_first_draw": {"median": float(np.median(first_m)), "p90": float(np.percentile(first_m, 90)),
                             "max": float(first_m.max()), "share_below_0.5": float((first_m < THR).mean())},
        "failed_windows_min_match": [float(v) for v in best_m[~passed]][:50],
        "failed_windows_chirp_mass": [float(v) for v in true[~passed]][:50],
    }
    mc.log(name, json.dumps(out[name]))
out["maxrss_GB"] = mc.rss_gb()
json.dump(out, open(os.path.join(mc.HERE, "manip_summary.json"), "w"), indent=1)
