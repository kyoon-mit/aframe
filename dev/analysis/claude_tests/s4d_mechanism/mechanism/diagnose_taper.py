"""Post hoc, not pre-registered: which part of the taper hurts each estimator (window-edge ramps or
internal ramps), and T2/T3 repeated with ramps only at the 7 internal boundaries (same draws)."""
import json
import os

import numpy as np
import torch

import mcommon as mc

X, true, merger, labels = mc.load_clean()
n = len(X)
Xt = torch.as_tensor(X)
man = {k: dict(np.load(os.path.join(mc.HERE, f"manip_{k}.npz"))) for k in ("T2", "T3")}
r = mc.ramp(mc.RAMP)
start = np.ones(mc.L); start[:mc.RAMP] = r
end = np.ones(mc.L); end[-mc.RAMP:] = r[::-1]
internal = mc.TAPER.copy(); internal[:mc.RAMP] = 1; internal[-mc.RAMP:] = 1
hard = np.ones(mc.L); hard[:mc.RAMP] = 0
WIN = {"ramp_start_only": start, "ramp_end_only": end, "ramp_both_window_edges": start * end,
       "zero_first_64_samples": hard, "internal_ramps_only": internal, "all_segment_ramps": mc.TAPER}


def win(name):
    return torch.as_tensor(WIN[name], device=mc.DEV)


def t2_int(x, s):
    B = x.shape[0]
    idx = torch.as_tensor(man["T2"]["params"][s], device=x.device)[:, None, :, None].expand(-1, 2, -1, mc.SEG)
    y = torch.gather(x.reshape(B, 2, mc.NSEG, mc.SEG), 2, idx).reshape(B, 2, mc.L)
    return y * win("internal_ramps_only").to(y.dtype)


def t3_int(x, s):
    xa = mc.analytic(x.double())
    ph = torch.as_tensor(man["T3"]["params"][s], dtype=torch.float64, device=x.device).repeat_interleave(mc.SEG, 1)
    return ((xa * torch.exp(1j * ph)[:, None]).real * win("internal_ramps_only")).to(x.dtype)


VAR = {k: (lambda k: (lambda x, s: x * win(k).to(x.dtype)))(k) for k in WIN}
VAR["T2_internal_ramps_only"] = t2_int
VAR["T3_internal_ramps_only"] = t3_int
out = {"note": "post hoc, not pre-registered; does not change any pre-registered verdict", "n": n}
# in-step check for the post hoc T2/T3 versions
for v in ("T2_internal_ramps_only", "T3_internal_ramps_only", "internal_ramps_only"):
    m = np.concatenate([mc.instep_match(VAR[v](Xt[s:s + 500].to(mc.DEV).double(), slice(s, s + 500)),
                                        Xt[s:s + 500].to(mc.DEV).double())[0].cpu().numpy() for s in range(0, n, 500)])
    out[v + "_instep_match"] = {"median": float(np.median(m)), "max": float(m.max()), "share_below_0.5": float((m < 0.5).mean())}
    np.save(os.path.join(mc.HERE, f"match_{v}.npy"), m)
    mc.log(v, out[v + "_instep_match"])
for key in ["s4d_ck69", "s4d_ck81", "free_ck46", "resnet_ck37", "bank"]:
    f = mc.load_estimator(key)
    clean = np.load(os.path.join(mc.HERE, f"est_clean_{key}.npy"))
    res = {"clean": float(mc.within1(clean, true).mean())}
    for v in VAR:
        e = np.concatenate([f(VAR[v](Xt[s:s + 500].to(mc.DEV), slice(s, s + 500)).float().cpu().numpy())
                            for s in range(0, n, 500)])
        np.save(os.path.join(mc.HERE, f"est_diag_{v}_{key}.npy"), e)
        ok = mc.within1(e, true)
        res[v] = float(ok.mean())
        res[v + "_ci"] = mc.boot_frac(ok)
        err = np.abs(e - true) / true
        res[v + "_within5"] = float((err < 0.05).mean())
    okc = mc.within1(clean, true)
    for t in ("T2", "T3"):
        okt = mc.within1(np.load(os.path.join(mc.HERE, f"est_diag_{t}_internal_ramps_only_{key}.npy")), true)
        oki = mc.within1(np.load(os.path.join(mc.HERE, f"est_diag_internal_ramps_only_{key}.npy")), true)
        res[f"{t}_internal_relative_drop_vs_clean"] = float((okc.mean() - okt.mean()) / okc.mean())
        res[f"{t}_internal_relative_drop_vs_clean_ci"] = mc.boot_drop(okc, okt)
        res[f"{t}_internal_relative_drop_vs_internal_control"] = float((oki.mean() - okt.mean()) / oki.mean())
        res[f"{t}_internal_relative_drop_vs_internal_control_ci"] = mc.boot_drop(oki, okt)
    out[key] = res
    mc.log(key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if not k.endswith("_ci")})
    del f
    torch.cuda.empty_cache()
out["maxrss_GB"] = mc.rss_gb()
json.dump(out, open(os.path.join(mc.HERE, "diagnose_taper.json"), "w"), indent=1)
