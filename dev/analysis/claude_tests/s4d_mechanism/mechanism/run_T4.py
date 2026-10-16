"""T4: peak times of first-layer channel envelopes on clean whitened chirps, against the time the
chirp passes each channel's frequency; clean S4D (ck_69, check ck_81) and references."""
import copy
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr

import mcommon as mc

sys.path.insert(0, mc.KERNELS)
import kcommon as kc  # noqa: E402

X, true, merger, labels = mc.load_clean()
n = len(X)
tm = merger / mc.FS                                   # merger time in the window, s
RULE = mc.PLAN["rules"]["T4"]


def stats(delta, freq, name, extra=None):
    """delta (n, C) peak time - merger (s); freq (C,) or None."""
    med = np.median(delta, 0)
    med_abs = np.median(np.abs(delta), 0)
    spread = float(np.percentile(med, 90) - np.percentile(med, 10))
    clustered = float((med_abs <= 0.1).mean())
    r = {"n_channels": int(delta.shape[1]), "spread_p90_minus_p10_s": spread,
         "clustered_share_median_abs_delta_le_0.1s": clustered,
         "channel_median_delta_s": [float(v) for v in med],
         "channel_median_abs_delta_s": [float(v) for v in med_abs],
         "per_window_share_within_0.1s_median": float(np.median((np.abs(delta) <= 0.1).mean(1))),
         "delta_quantiles_all_s": [float(v) for v in np.percentile(delta, [5, 25, 50, 75, 95])]}
    if freq is not None:
        rf = rankdata(freq)
        rd = rankdata(delta, axis=1)
        rd = rd - rd.mean(1, keepdims=True)
        rfc = rf - rf.mean()
        den = np.sqrt((rd ** 2).sum(1) * (rfc ** 2).sum())
        rho_w = np.where(den > 0, (rd * rfc).sum(1) / np.where(den > 0, den, 1), 0.0)
        r["rho_per_window_median"] = float(np.median(rho_w))
        r["rho_per_window_quantiles"] = [float(v) for v in np.percentile(rho_w, [10, 25, 50, 75, 90])]
        r["rho_per_window_share_gt_0.7"] = float((rho_w > 0.7).mean())
        r["rho_channel_medians"] = float(spearmanr(freq, med).correlation)
        r["channel_freq_Hz"] = [float(v) for v in freq]
        is_local = r["rho_per_window_median"] > 0.7 and spread >= 1.0 and clustered <= 0.5
    else:
        r["rho_per_window_median"] = None
        is_local = False
    is_global = clustered > 0.5 and not is_local
    r["rule_verdict"] = "supports H_local" if is_local else ("supports H_global" if is_global else "inconclusive")
    if extra:
        r.update(extra)
    mc.log(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                  if not isinstance(v, list) or len(v) <= 5})
    return r


def env_peak(y):
    """argmax over time of |analytic(y)| for real y (..., T)."""
    return mc.analytic(y).abs().argmax(-1)


out = {"n_windows": n}

# ------------------------------------------------ whitened chirp itself (no filter): where a pass-through peaks
pk = []
for s in range(0, n, 250):
    x = torch.as_tensor(X[s:s + 250], device=mc.DEV).double()
    pk.append(mc.analytic(x).abs().pow(2).sum(1).argmax(-1).cpu().numpy())
pk = np.concatenate(pk)
d0 = pk / mc.FS - tm
out["whitened_chirp_envelope_peak_minus_merger_s"] = {"quantiles_5_25_50_75_95": [float(v) for v in np.percentile(d0, [5, 25, 50, 75, 95])],
                                                      "share_within_0.1s": float((np.abs(d0) <= 0.1).mean())}
mc.log("chirp envelope", out["whitened_chirp_envelope_peak_minus_merger_s"])


# ------------------------------------------------ clean S4D, first layer
def s4d_dominant(lay):
    modes = kc.s4d_modes(copy.deepcopy(lay.kernel).cpu(), 2048.0)
    E = kc.mode_energy(modes["z"], modes["c"], mc.L) / 2
    w = E / E.sum(1, keepdims=True)
    fa = np.abs(modes["f"])
    inb = (fa >= 20) & (fa < 1024)
    dom = np.full(w.shape[0], np.nan)
    for c in range(w.shape[0]):
        if inb[c].any():
            j = int(np.argmax(np.where(inb[c], w[c], -1)))
            dom[c] = fa[c, j]
    return dom


for key in ("s4d_ck69", "s4d_ck81"):
    net = mc.load_estimator(key)
    m = net.arch.net
    lay = m.s4_layers[0]
    dom = s4d_dominant(lay)
    if key == "s4d_ck69":
        tab = json.load(open(os.path.join(mc.KERNELS, "clean_s4d_layer1_table.json")))["channels"]
        dom_tab = np.array([abs(c["dominant_f"]) if "dominant_f" in c else np.nan for c in tab])
        both = ~np.isnan(dom_tab)
        out["s4d_ck69_dominant_recomputed_vs_table_max_abs_Hz"] = float(np.nanmax(np.abs(dom[both] - dom_tab[both])))
        assert np.array_equal(np.isnan(dom), np.isnan(dom_tab))
    sel = np.where((dom >= 20) & (dom <= 150))[0]
    W = m.encoder.weight.detach().double()                      # (64, 2)
    kd = copy.deepcopy(lay.kernel).double()
    with torch.no_grad():
        K = kd(mc.L).to(mc.DEV)                                 # (64, L) float64
        K32 = lay.kernel(mc.L).double().to(mc.DEV)
    D = lay.D.detach().double().to(mc.DEV)
    KF = torch.fft.rfft(K, n=2 * mc.L)
    peaks, peaks_noD = [], []
    with torch.no_grad():
        for s in range(0, n, 25):
            x = torch.as_tensor(X[s:s + 25], device=mc.DEV).double()
            u = torch.einsum("hd,bdl->bhl", W.to(mc.DEV), x)
            yk = torch.fft.irfft(torch.fft.rfft(u, n=2 * mc.L) * KF, n=2 * mc.L)[..., :mc.L]
            peaks.append(env_peak(yk + D[:, None] * u).cpu().numpy())
            peaks_noD.append(env_peak(yk).cpu().numpy())
    peaks = np.concatenate(peaks)                               # (n, 64)
    peaks_noD = np.concatenate(peaks_noD)
    np.savez(os.path.join(mc.HERE, f"T4_{key}.npz"), peaks=peaks, peaks_noD=peaks_noD, dominant_f=dom, sel=sel)
    # decomposition check: hooked pre-GELU tensor of layer 1 = a * y + c * (step + D) per window and channel
    cap = {}
    h = lay.activation.register_forward_hook(lambda mod, inp, o: cap.__setitem__("pre", inp[0].detach()))
    xs = torch.as_tensor(X[:8], device=mc.DEV)
    with torch.no_grad():
        net.arch(xs)
    h.remove()
    pre = cap["pre"].double()                                   # (8, 64, L) float32 network values
    with torch.no_grad():
        x = xs.double()
        u = torch.einsum("hd,bdl->bhl", W.to(mc.DEV), x)
        y = torch.fft.irfft(torch.fft.rfft(u, n=2 * mc.L) * KF, n=2 * mc.L)[..., :mc.L] + D[:, None] * u
        step = torch.cumsum(K, -1) + D[:, None]                # response to a constant 1
        A = torch.stack([y, step[None].expand_as(y)], -1)      # (8, 64, L, 2)
        coef = torch.linalg.lstsq(A, pre[..., None]).solution  # (8, 64, 2, 1)
        fit = (A @ coef)[..., 0]
        resid = (pre - fit).norm(dim=-1) / pre.norm(dim=-1)
        chirp_part = (coef[..., 0, 0, None] * y).norm(dim=-1)
        const_part = (coef[..., 1, 0, None] * step[None]).norm(dim=-1)
        pk_pre = []
        for b in range(8):
            yy = pre[b] - coef[b, :, 1, 0, None] * step
            pk_pre.append(env_peak(yy).cpu().numpy())
    pk_pre = np.stack(pk_pre)
    chk = {"kernel_float64_vs_float32_max_abs": float((K - K32).abs().max()), "kernel_max_abs": float(K.abs().max()),
           "pre_gelu_fit_relative_residual_max": float(resid.max()),
           "pre_gelu_fit_relative_residual_median": float(resid.median()),
           "const_response_over_chirp_part_norm_median_selected": float((const_part / chirp_part)[:, sel].median()),
           "const_response_over_chirp_part_norm_max_selected": float((const_part / chirp_part)[:, sel].max()),
           "peak_sample_hooked_minus_mine_max_abs_selected": int(np.abs(pk_pre[:, sel] - peaks[:8, sel]).max())}
    mc.log(key, "checks", chk)
    delta = peaks[:, sel] / mc.FS - tm[:, None]
    tau = kc.tau_of_f(dom[sel][None, :], true[:, None])        # (n, C) s before merger
    pass_rel = -tau                                            # passing time - merger
    inside = (tm[:, None] - tau) >= 0
    extra = {"checks": chk, "channels": [int(c) for c in sel],
             "share_of_window_channel_pairs_where_chirp_passes_f_inside_window": float(inside.mean()),
             "spearman_delta_vs_passing_time_pooled": float(spearmanr(delta.ravel(), pass_rel.ravel()).correlation),
             "median_abs_delta_minus_passing_time_where_inside_s": float(np.median(np.abs(delta - pass_rel)[inside])) if inside.any() else None}
    out[key] = stats(delta, dom[sel], key, extra)
    out[key + "_kernel_only_noD_descriptive"] = stats(peaks_noD[:, sel] / mc.FS - tm[:, None], dom[sel], key + " no D")
    del net
    torch.cuda.empty_cache()

# ------------------------------------------------ free kernels (clean), complex kernels, both detectors
net = mc.load_estimator("free_ck46")
body = net.arch.body
tab = json.load(open(os.path.join(mc.KERNELS, "clean_free_table.json")))["channels"]
fpk = np.array([c["f_peak"] for c in tab])
sel = np.where((fpk >= 20) & (fpk <= 150))[0]
with torch.no_grad():
    k = body.kernels()[sel].to(mc.DEV)                         # (C, L) complex64 unit norm
    kf = torch.fft.fft(k, n=2 * mc.L)
    peaks = []
    for s in range(0, n, 16):
        x = torch.as_tensor(X[s:s + 16], device=mc.DEV).float()
        y = torch.fft.ifft(torch.fft.fft(x, n=2 * mc.L)[:, :, None] * kf, n=2 * mc.L)[..., :mc.L]
        peaks.append(y.abs().pow(2).sum(1).argmax(-1).cpu().numpy())
peaks = np.concatenate(peaks)
np.savez(os.path.join(mc.HERE, "T4_free_ck46.npz"), peaks=peaks, f_peak=fpk, sel=sel)
delta = peaks / mc.FS - tm[:, None]
out["free_ck46"] = stats(delta, fpk[sel], "free_ck46", {"channels": [int(c) for c in sel]})
del net
torch.cuda.empty_cache()

# ------------------------------------------------ ResNet first convolution
net = mc.load_estimator("resnet_ck37")
w = net.arch.net.conv1.weight.detach().double().cpu()          # (64, 2, 7)
f = np.arange(0, 1024.001, 0.5)
kk = np.arange(7) - 3
Hf = np.einsum("cdk,fk->cdf", w.numpy(), np.exp(-2j * np.pi * f[:, None] * kk[None] / mc.FS))
P = (np.abs(Hf) ** 2).sum(1)
fres = f[P.argmax(-1)]
sel = np.where((fres >= 20) & (fres <= 150))[0]
peaks = []
with torch.no_grad():
    wd = w.to(mc.DEV)
    for s in range(0, n, 100):
        x = torch.as_tensor(X[s:s + 100], device=mc.DEV).double()
        y = F.conv1d(x, wd, stride=2, padding=3)                 # (b, 64, 4096) at 1024 Hz
        peaks.append(env_peak(y).cpu().numpy())
peaks = np.concatenate(peaks)
np.savez(os.path.join(mc.HERE, "T4_resnet_ck37.npz"), peaks=peaks, f_peak=fres, sel=sel)
out["resnet_ck37_channel_peak_freq_Hz_all"] = [float(v) for v in fres]
if len(sel) >= 2:
    delta = 2 * peaks[:, sel] / mc.FS - tm[:, None]
    out["resnet_ck37"] = stats(delta, fres[sel], "resnet_ck37", {"channels": [int(c) for c in sel]})
else:
    out["resnet_ck37"] = {"n_channels": int(len(sel)), "note": "fewer than 2 channels peak in 20-150 Hz"}
    mc.log("resnet", out["resnet_ck37"])
# descriptive: all 64 channels
delta_all = 2 * peaks / mc.FS - tm[:, None]
out["resnet_ck37_all_channels_descriptive"] = stats(delta_all, fres, "resnet all channels")
del net
torch.cuda.empty_cache()

# ------------------------------------------------ template bank: each template's correlation output
bank = mc.Bank()
_, lag = bank.snr2(X, batch=8, want_lag=True)                  # (n, J)
delta = (bank.merge + lag) / mc.FS - tm[:, None]
np.save(os.path.join(mc.HERE, "T4_bank_lag.npy"), lag)
out["bank"] = stats(delta, None, "bank")
best = np.abs(np.log(bank.mc)[None, :] - np.log(true)[:, None]).argmin(1)
db = delta[np.arange(n), best]
out["bank_true_template_delta_quantiles_s"] = [float(v) for v in np.percentile(db, [5, 25, 50, 75, 95])]
out["maxrss_GB"] = mc.rss_gb()
json.dump(out, open(os.path.join(mc.HERE, "results_T4.json"), "w"), indent=1)
mc.log("done", f"rss {mc.rss_gb():.2f} GB")
