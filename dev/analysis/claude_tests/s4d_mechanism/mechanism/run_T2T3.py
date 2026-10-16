"""T2 and T3 with the taper-only control, every estimator, and the pre-registered verdicts."""
import json
import os

import numpy as np
import torch

import mcommon as mc

X, true, merger, labels = mc.load_clean()
n = len(X)
Xt = torch.as_tensor(X)
man = {k: dict(np.load(os.path.join(mc.HERE, f"manip_{k}.npz"))) for k in ("T2", "T3")}
VARIANTS = {
    "taper": lambda x, s: mc.taper_only(x),
    "T2": lambda x, s: mc.shuffle(x, man["T2"]["params"][s]),
    "T2_first": lambda x, s: mc.shuffle(x, man["T2"]["first_params"][s]),
    "T3": lambda x, s: mc.phase(x, man["T3"]["params"][s]),
    "T3_first": lambda x, s: mc.phase(x, man["T3"]["first_params"][s]),
}
KEYS = ["s4d_ck69", "s4d_ck81", "free_ck46", "resnet_ck37", "bank"]
R = mc.PLAN["rules"]["T2_T3"]


def make(variant, s0, s1):
    x = Xt[s0:s1].to(mc.DEV)
    return VARIANTS[variant](x, slice(s0, s1)).float().cpu().numpy()


est = {}
for key in KEYS:
    f = mc.load_estimator(key)
    est[key] = {"clean": np.load(os.path.join(mc.HERE, f"est_clean_{key}.npy"))}
    for v in VARIANTS:
        e = np.concatenate([f(make(v, s, min(n, s + 500))) for s in range(0, n, 500)])
        est[key][v] = e
        np.save(os.path.join(mc.HERE, f"est_{v}_{key}.npy"), e)
    mc.log(key, {v: round(float(mc.within1(e, true).mean()), 4) for v, e in est[key].items()},
           f"rss {mc.rss_gb():.2f} GB")
    del f
    torch.cuda.empty_cache()


def verdict(drop):
    if drop <= R["H_local_max"]:
        return "supports H_local"
    if drop >= R["H_global_min"]:
        return "supports H_global"
    return "inconclusive"


out = {"n": n, "plan_sha256_note": "see write-up", "tests": {}}
for test in ("T2", "T3"):
    passed = man[test]["passed"].astype(bool)
    res = {"n_windows_evaluated": int(passed.sum()), "pass_share": float(passed.mean()),
           "check_met": bool(passed.mean() >= mc.PLAN["manipulation"]["pass_share_required"]), "by_estimator": {}}
    for key in KEYS:
        ok = {v: mc.within1(est[key][v], true) for v in ("clean", "taper", test, test + "_first")}
        a_clean = float(ok["clean"][passed].mean())
        a_taper = float(ok["taper"][passed].mean())
        a_test = float(ok[test][passed].mean())
        drop = (a_clean - a_test) / a_clean
        taper_drop = (a_clean - a_taper) / a_clean
        a_clean_all = float(ok["clean"].mean())
        a_first = float(ok[test + "_first"].mean())
        r = {
            "within1_clean": a_clean, "within1_clean_ci": mc.boot_frac(ok["clean"][passed]),
            "within1_taper_only": a_taper, "within1_taper_only_ci": mc.boot_frac(ok["taper"][passed]),
            "within1_manipulated": a_test, "within1_manipulated_ci": mc.boot_frac(ok[test][passed]),
            "relative_drop_vs_clean": drop, "relative_drop_vs_clean_ci": mc.boot_drop(ok["clean"][passed], ok[test][passed]),
            "relative_drop_vs_taper_only": (a_taper - a_test) / a_taper if a_taper > 0 else None,
            "relative_drop_vs_taper_only_ci": mc.boot_drop(ok["taper"][passed], ok[test][passed]),
            "taper_only_relative_drop": taper_drop,
            "taper_check_passed": bool(taper_drop < 0.30),
            "first_draw_all_windows": {"within1_clean": a_clean_all, "within1_manipulated": a_first,
                                       "relative_drop_vs_clean": (a_clean_all - a_first) / a_clean_all},
            "within_2_5_10pct_manipulated": [float((np.abs(est[key][test][passed] - true[passed]) / true[passed] < t / 100).mean())
                                             for t in (2, 5, 10)],
            "within_2_5_10pct_clean": [float((np.abs(est[key]["clean"][passed] - true[passed]) / true[passed] < t / 100).mean())
                                       for t in (2, 5, 10)],
        }
        r["rule_verdict"] = verdict(drop)
        if not r["taper_check_passed"]:
            r["rule_verdict_note"] = "taper alone explains the drop: not counted as support"
        res["by_estimator"][key] = r
        mc.log(test, key, f"clean {a_clean:.4f} taper {a_taper:.4f} manip {a_test:.4f} drop {drop:.3f} "
               f"CI {r['relative_drop_vs_clean_ci']} -> {r['rule_verdict']}")
    out["tests"][test] = res
out["maxrss_GB"] = mc.rss_gb()
json.dump(out, open(os.path.join(mc.HERE, "results_T2T3.json"), "w"), indent=1)
