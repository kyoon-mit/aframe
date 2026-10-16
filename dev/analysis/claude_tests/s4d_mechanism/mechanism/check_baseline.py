"""Unmanipulated clean windows: every estimator, compared with the harness's saved estimates;
tool checks (analytic signal, match of the original with itself)."""
import json
import os

import numpy as np
import torch

import mcommon as mc

X, true, merger, labels = mc.load_clean()
out = {"n": len(X)}
# tool checks
x = torch.as_tensor(X[:64], device=mc.DEV).double()
xa = mc.analytic(x)
out["analytic_real_minus_x_max_abs"] = float((xa.real - x).abs().max())
out["analytic_imag_dot_x_over_norm2_max"] = float(((xa.imag * x).sum(-1).abs() / x.pow(2).sum(-1)).max())
m, _ = mc.instep_match(x, x)
out["self_match_min_max"] = [float(m.min()), float(m.max())]
ident = np.tile(np.arange(8), (64, 1))
m, _ = mc.instep_match(mc.shuffle(x, ident), x)
mt, _ = mc.instep_match(mc.taper_only(x), x)
out["identity_shuffle_equals_taper_only_max_abs"] = float((mc.shuffle(x, ident) - mc.taper_only(x)).abs().max())
out["zero_phase_equals_taper_only_max_abs"] = float((mc.phase(x, np.zeros((64, 8))) - mc.taper_only(x)).abs().max())
out["taper_energy_kept_mean"] = float(np.mean(mc.TAPER ** 2))
mc.log("tool checks", json.dumps(out))
for key in ["s4d_ck69", "s4d_ck81", "free_ck46", "resnet_ck37", "bank"]:
    est_fn = mc.load_estimator(key)
    est = np.concatenate([est_fn(X[i:i + 500]) for i in range(0, len(X), 500)])
    res = {"within_1pct": float(mc.within1(est, true).mean())}
    if key != "bank":
        name, tag = mc.NETS[key]
        saved = np.load(mc.harness.DATA_DIR / f"est_clean__{name}__{tag}.npy")
        res["saved_within_1pct_f16"] = float(mc.within1(saved, true).mean())
        res["max_rel_diff_vs_saved_f16"] = float(np.max(np.abs(est - saved) / saved))
        res["decision_flips_vs_saved_f16"] = int((mc.within1(est, true) != mc.within1(saved, true)).sum())
    else:
        res["grid_mc_min_max"] = [float(est_fn.mc.min()), float(est_fn.mc.max())]
        err = np.abs(est - true) / true
        res["within_2_5_10pct"] = [float((err < t / 100).mean()) for t in (2, 5, 10)]
    np.save(os.path.join(mc.HERE, f"est_clean_{key}.npy"), est)
    out[key] = res
    mc.log(key, json.dumps(res), f"rss {mc.rss_gb():.2f} GB",
           f"gpu {torch.cuda.max_memory_allocated() / 2**30:.2f} GB" if torch.cuda.is_available() else "")
    del est_fn
    torch.cuda.empty_cache()
out["maxrss_GB"] = mc.rss_gb()
json.dump(out, open(os.path.join(mc.HERE, "check_baseline.json"), "w"), indent=1)
