"""T1 (descriptive): keep only a tapered sub-window before the merger, every estimator."""
import json
import os

import numpy as np
import torch

import mcommon as mc

X, true, merger, labels = mc.load_clean()
n = len(X)
P = mc.PLAN["T1"]
KEYS = ["s4d_ck69", "s4d_ck81", "free_ck46", "resnet_ck37", "bank"]
variants = [("full", 4.0, None)]
for Ls in P["lengths_s"]:
    for d in P["end_before_merger_s"]:
        if d + Ls <= 3.5 + 1e-9:
            variants.append((f"len{Ls}_end{d}", Ls, d))


def masks(Ls, d, s0, s1):
    M = np.zeros((s1 - s0, mc.L))
    if d is None:
        M[:] = mc.tukey(mc.L, mc.RAMP)
        return M
    ns = int(round(Ls * mc.FS))
    ds = int(round(d * mc.FS))
    w = mc.tukey(ns, min(mc.RAMP, ns // 4))
    for j, i in enumerate(range(s0, s1)):
        e = min(int(merger[i]), mc.L - 1) + 1 - ds
        M[j, e - ns:e] = w
        assert e - ns >= 0
    return M


kept = {}
for name, Ls, d in variants:
    k = []
    for s in range(0, n, 500):
        M = masks(Ls, d, s, min(n, s + 500))
        x = X[s:s + 500]
        k.append(((x * M[:, None]) ** 2).sum((1, 2)) / (x ** 2).sum((1, 2)))
    kept[name] = np.concatenate(k)
res = {"variants": {name: {"length_s": Ls, "end_before_merger_s": d,
                           "snr2_kept_quantiles_10_50_90": [float(v) for v in np.percentile(kept[name], [10, 50, 90])]}
                    for name, Ls, d in variants}}
for key in KEYS:
    f = mc.load_estimator(key)
    for name, Ls, d in variants:
        e = []
        for s in range(0, n, 500):
            M = masks(Ls, d, s, min(n, s + 500))
            e.append(f((X[s:s + 500] * M[:, None]).astype(np.float32)))
        e = np.concatenate(e)
        err = np.abs(e - true) / true
        res["variants"][name][key] = {"within_1pct": float((err < 0.01).mean()),
                                      "within_1pct_ci": mc.boot_frac(err < 0.01),
                                      "within_5pct": float((err < 0.05).mean()),
                                      "median_abs_rel_err": float(np.median(err))}
        np.save(os.path.join(mc.HERE, f"est_T1_{name}_{key}.npy"), e)
    mc.log(key, {nm: round(res["variants"][nm][key]["within_1pct"], 3) for nm, _, _ in variants}, f"rss {mc.rss_gb():.2f} GB")
    del f
    torch.cuda.empty_cache()
res["maxrss_GB"] = mc.rss_gb()
json.dump(res, open(os.path.join(mc.HERE, "results_T1.json"), "w"), indent=1)
