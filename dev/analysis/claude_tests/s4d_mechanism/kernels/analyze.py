"""Per-channel kernels of one trained network: resonances, transfer function, group delay,
match to whitened chirp templates. usage: python analyze.py NAME

Writes <NAME>_modes.npz (S4D only), <NAME>_transfer.npz, <NAME>_match.npz, <NAME>_channels.json.
Match conventions:
  complex kernels (long_conv): sqrt(max_t 2|k * x|^2 / ||x||^2), the network's own energy over
    SNR^2 (k unit norm, x real whitened template), as in LongConvEnergyNet.peak_energy;
  real kernels (clean S4D first layer): max_t |k * x_a| / (||k|| ||x||), the peak of the
    output envelope over SNR, maximized over the template's phase.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

torch.set_num_threads(8)
name = sys.argv[1]
t0 = time.time()
m, ck, cfg = kc.build(name)
kind = kc.NETS[name]["kind"]
info = {"name": name, "checkpoint": kc.NETS[name]["ck"], "epoch": int(ck["epoch"])}
for key, v in ck.get("callbacks", {}).items():
    if "ModelCheckpoint" in key:
        info["best_model_score_within_1pct"] = float(v["best_model_score"])
checks = {}

# ------------------------------------------------------------------ kernels
modes = None
D = None
body = m if kind == "long_conv" else (m.body if name == "clean_free" else None)
if body is not None:                       # LongConvEnergyNet
    fs = 2048.0 / body.decimate
    L = body.length
    real_kernel = False
    if body.kernel_type == "s4d":
        modes = kc.s4d_modes(body.kernel, fs)
        k = kc.kernel_from_modes(modes["z"], modes["c"], L)
        raw_norm = np.linalg.norm(k, axis=-1)
        k = k / raw_norm[:, None]
        # the module's own kernel code on 8 channels (the full call needs > 4 GB at d_state 512)
        from train.experimental.s4d_energy.arch import ComplexS4DKernel
        sub = list(range(0, k.shape[0], max(1, k.shape[0] // 8)))[:8]
        ref = ComplexS4DKernel(len(sub), N=2 * body.kernel.A_imag.shape[1]).double()
        with torch.no_grad():
            for p in ("log_dt", "log_A_real", "A_imag", "C"):
                getattr(ref, p).copy_(getattr(body.kernel, p)[sub])
            kr = ref(L).numpy()
        kr = kr / np.linalg.norm(kr, axis=-1, keepdims=True)
        checks["kernel_vs_module_max_abs_diff_unit_norm_8ch"] = float(np.abs(kr - k[sub]).max())
        if body.kernel.A_imag.shape[1] <= 32:
            with torch.no_grad():
                kfull = body.kernels().numpy()
            checks["kernel_vs_module_kernels()_max_abs_diff_all_channels"] = float(np.abs(kfull - k).max())
    else:
        with torch.no_grad():
            k = body.kernels().numpy().astype(np.complex128)
        raw_norm = np.ones(k.shape[0])
    readout = body.out.weight.detach().double().norm(dim=0).numpy() if kind == "long_conv" else None
    kt = k                                  # kernel used for the transfer function
    kmatch, kmatchD = k, None
else:                                      # clean S4D, first S4D layer
    fs = 2048.0
    L = 8192
    real_kernel = True
    lay = m.net.s4_layers[0]
    modes = kc.s4d_modes(lay.kernel, fs)
    kreal = kc.kernel_from_modes(modes["z"], modes["c"], L).real        # S4DKernel: 2 Re(sum C' z^l)
    with torch.no_grad():
        kref = lay.kernel(L).double().numpy()
    checks["kernel_vs_module_max_abs_diff"] = float(np.abs(kref - kreal).max())
    checks["kernel_module_max_abs"] = float(np.abs(kref).max())
    D = lay.D.detach().double().numpy()
    raw_norm = np.linalg.norm(kreal, axis=-1)
    kt = kreal
    kmatch = kreal / raw_norm[:, None]
    kD = kreal.copy()
    kD[:, 0] += D
    kmatchD = kD / np.linalg.norm(kD, axis=-1, keepdims=True)
    readout = None
C = kt.shape[0]
nyq = fs / 2
info.update(fs=fs, L=L, channels=C, nyquist=nyq, real_kernel=real_kernel)

# ------------------------------------------------------------------ modes table
chan = [dict(channel=c) for c in range(C)]
if modes is not None:
    f, ring, alias = modes["f"], modes["ring"], modes["alias"]
    E = kc.mode_energy(modes["z"], modes["c"], L)
    if real_kernel:
        E = E / 2                          # Re(c z^l) carries half of |c z^l|^2 away from 0 and Nyquist
    w = E / E.sum(1, keepdims=True)
    interf = raw_norm ** 2 / E.sum(1)
    fa = np.abs(f)
    inb = (fa >= 20) & (fa < nyq)
    np.savez(os.path.join(kc.HERE, f"{name}_modes.npz"), f=f, ring=ring, alias=alias, weight=w,
             z=modes["z"], c=modes["c"], dt=modes["dt"], a=modes["a"], b=modes["b"], inband=inb,
             kernel_norm2_over_sum_mode_energy=interf)
    for c in range(C):
        ch = chan[c]
        ch.update(n_modes=int(f.shape[1]), n_inband=int(inb[c].sum()),
                  n_inband_positive=int((inb[c] & (f[c] > 0)).sum()),
                  n_aliased=int(alias[c].sum()),
                  mode_weight_inband=float(w[c, inb[c]].sum()),
                  log_dt=float(np.log(modes["dt"][c])),
                  kernel_norm2_over_sum_mode_energy=float(interf[c]))
        if inb[c].any():
            j = int(np.argmax(np.where(inb[c], w[c], -1)))
            ch.update(dominant_f=float(f[c, j]), dominant_ring=float(ring[c, j]), dominant_weight=float(w[c, j]))
    checks["log_dt_at_clamp_bounds"] = int(((modes["dt"] <= math.exp(-11.5) * 1.0001) | (modes["dt"] >= math.exp(2.3) * 0.9999)).sum())
    checks["a_at_clamp_max"] = int((modes["a"] >= math.exp(4.6) * 0.9999).sum())
if D is not None:
    for c in range(C):
        chan[c]["D"] = float(D[c])
        chan[c]["D2_over_kernel_norm2"] = float(D[c] ** 2 / raw_norm[c] ** 2)

# ------------------------------------------------------------------ transfer function, group delay
nfft = 4 * L
fr, H, gd = kc.transfer(kt, fs, nfft)
P = np.abs(H) ** 2
fa_grid = np.abs(fr)
band = (fa_grid >= 20) & (fa_grid < nyq)
lag_e = np.abs(kt) ** 2
cum = np.cumsum(lag_e, -1) / lag_e.sum(-1, keepdims=True)
for c in range(C):
    ch = chan[c]
    pc = P[c]
    tot = pc.sum()
    ch["energy_share_20Hz_to_nyquist"] = float(pc[band].sum() / tot)
    ch["energy_share_negative_freq"] = None if real_kernel else float(pc[fr < 0].sum() / tot)
    fb = fa_grid[band]
    pb = pc[band]
    o = np.argsort(fb, kind="stable")
    cs = np.cumsum(pb[o]) / pb.sum()
    ch["f10"], ch["f50"], ch["f90"] = [float(fb[o][min(np.searchsorted(cs, q), len(cs) - 1)]) for q in (0.1, 0.5, 0.9)]
    ch["f_peak"] = float(fa_grid[np.argmax(np.where(band, pc, -1))])
    sel = band & (pc > 1e-3 * pc[band].max())
    ww = pc[sel]
    ch["group_delay_mean_s"] = float((gd[c, sel] * ww).sum() / ww.sum())
    ch["lag50_s"] = float((np.searchsorted(cum[c], 0.5) + 1) / fs)
    ch["lag90_s"] = float((np.searchsorted(cum[c], 0.9) + 1) / fs)
    if readout is not None:
        ch["readout_weight_norm"] = float(readout[c])
keep = slice(0, nfft, 4)                    # fs / L = 0.25 Hz bins
fsave = fr[keep]
order = np.argsort(fsave)
np.savez_compressed(os.path.join(kc.HERE, f"{name}_transfer.npz"), f=fsave[order],
                    absH=np.abs(H[:, keep])[:, order].astype(np.float32),
                    group_delay_s=np.where(P[:, keep] > 1e-4 * P.max(1, keepdims=True), gd[:, keep], np.nan)[:, order].astype(np.float32))

# ------------------------------------------------------------------ matches
# m_pos: unit-norm kernel against the analytic template, max_t |k * x_a| / ||x_a|| (task convention)
# m_neg: the same for conj(k), the kernel's negative-frequency side acting on the real input
# m_net: sqrt(max_t 2|k * x|^2 / ||x||^2), the network's own energy over SNR^2 (complex kernels)
# m_real: real kernels, max_t |k * x_a| / (||k|| ||x||), the output peak over SNR at the best phase
# in-band match: each side divided by the root of its share of kernel energy in 20 Hz..Nyquist,
#   the match of that side alone as a band-limited analytic kernel (out-of-band energy sees no data)
x, mc = kc.templates(rate=int(fs), ifo=0)
x1, _ = kc.templates(rate=int(fs), ifo=1)
J = x.shape[0]
pos_band = (fr >= 20) & (fr < nyq)
neg_band = (fr <= -20) & (fr > -nyq)
s_pos = P[:, pos_band].sum(1) / P.sum(1)
s_neg = P[:, neg_band].sum(1) / P.sum(1)
res = dict(chirp_mass=mc)
# band-limited analytic template (20 Hz..Nyquist) for the in-band match, normalized by its own norm
xb2 = 2 * kc.band_norm2(x, fs, 20, nyq)                    # ||x_a,band||^2
bl = dict(analytic=True, fband=(20, nyq), fs=fs, xnorm2=xb2)
checks["template_snr2_share_20Hz_to_nyquist_min"] = float((xb2 / 2 / (x ** 2).sum(-1)).min())
if real_kernel:
    E, _ = kc.peak_energy_ratio(kmatch, x, analytic=True)
    m_real = np.sqrt(E)
    m_pos = m_neg = m_real / math.sqrt(2)
    res["match_real"] = m_real.astype(np.float32)
    res["match_real_L1"] = np.sqrt(kc.peak_energy_ratio(kmatch, x1, analytic=True)[0]).astype(np.float32)
    Eb, targ = kc.peak_energy_ratio(kmatch, x, **bl)
    inband = np.sqrt(Eb / ((s_pos + s_neg) / 2)[None])     # positive in-band half holds half the band energy
    ED, _ = kc.peak_energy_ratio(kmatchD, x, analytic=True)
    EDb, _ = kc.peak_energy_ratio(kmatchD, x, **bl)
    PD = np.abs(H + D[:, None]) ** 2
    sD = PD[:, band].sum(1) / PD.sum(1)
    res["match_real_with_D"] = np.sqrt(ED).astype(np.float32)
    res["inband_match_with_D"] = np.sqrt(EDb / (sD / 2)[None]).astype(np.float32)
    side = np.zeros_like(inband, dtype=np.int8)
else:
    En, _ = kc.peak_energy_ratio(kmatch, x)
    res["match_net"] = np.sqrt(En).astype(np.float32)
    Ep, _ = kc.peak_energy_ratio(kmatch, x, analytic=True)
    Em, _ = kc.peak_energy_ratio(np.conj(kmatch), x, analytic=True)
    m_pos, m_neg = np.sqrt(Ep / 2), np.sqrt(Em / 2)
    res["match_pos_L1"] = np.sqrt(kc.peak_energy_ratio(kmatch, x1, analytic=True)[0] / 2).astype(np.float32)
    Epb, targ = kc.peak_energy_ratio(kmatch, x, **bl)
    Emb, targm = kc.peak_energy_ratio(np.conj(kmatch), x, **bl)
    with np.errstate(divide="ignore", invalid="ignore"):
        ip = np.where(s_pos[None] >= 0.01, np.sqrt(Epb / s_pos[None]), np.nan)
        im = np.where(s_neg[None] >= 0.01, np.sqrt(Emb / s_neg[None]), np.nan)
    inband = np.fmax(ip, im)
    side = np.where(np.nan_to_num(im, nan=-1) > np.nan_to_num(ip, nan=-1), 1, 0).astype(np.int8)
    targ = np.where(side == 1, targm, targ)
res.update(match_pos=m_pos.astype(np.float32), match_neg=m_neg.astype(np.float32),
           inband_match=inband.astype(np.float32), inband_side_negative=side, t_best=targ.astype(np.int32))
# matched kernels of the templates: self-match checks and their group delay
xa = kc.analytic_part(x)                                   # unit-norm analytic templates
km = np.conj(xa[:, ::-1])                                  # time-reversed conjugate, output peak at t = L-1
Es, _ = kc.peak_energy_ratio(km, x)
checks["template_self_match_network_convention_min_max"] = [float(np.sqrt(np.diag(Es)).min()), float(np.sqrt(np.diag(Es)).max())]
Es2, _ = kc.peak_energy_ratio(km, x, analytic=True)
checks["template_self_match_analytic_convention_min_max"] = [float(np.sqrt(np.diag(Es2) / 2).min()), float(np.sqrt(np.diag(Es2) / 2).max())]
Es3, _ = kc.peak_energy_ratio(km, x, **bl)
checks["template_self_match_inband_convention_min_max"] = [float(np.sqrt(np.diag(Es3)).min()), float(np.sqrt(np.diag(Es3)).max())]
kr_m = x[:, ::-1] / np.linalg.norm(x, axis=-1, keepdims=True)
Er, _ = kc.peak_energy_ratio(kr_m, x, analytic=True)
checks["template_self_match_real_kernel_min_max"] = [float(np.sqrt(np.diag(Er)).min()), float(np.sqrt(np.diag(Er)).max())]
np.savez_compressed(os.path.join(kc.HERE, f"{name}_match.npz"), **res)

fm, Hm, gdm = kc.transfer(km, fs, nfft)
Pm = np.abs(Hm) ** 2
Xs = np.abs(np.fft.fft(x, n=nfft, axis=-1))               # real template magnitude on the transfer grid
for c in range(C):
    ch = chan[c]
    ch["energy_share_pos_band"], ch["energy_share_neg_band"] = float(s_pos[c]), float(s_neg[c])
    j = int(np.nanargmax(inband[:, c])) if np.isfinite(inband[:, c]).any() else int(np.argmax(m_pos[:, c]))
    neg = bool(side[j, c])
    ch.update(best_inband_match=float(np.nanmax(inband[:, c])) if np.isfinite(inband[:, c]).any() else None,
              best_chirp_mass=float(mc[j]), best_template=j, inband_side="negative" if neg else "positive",
              best_output_time_s=float(targ[j, c] / fs),
              best_match_pos=float(m_pos[:, c].max()), best_match_neg=float(m_neg[:, c].max()))
    if real_kernel:
        ch["best_match_real"] = float(m_real[:, c].max())
        ch["best_match_real_L1"] = float(res["match_real_L1"][:, c].max())
        ch["best_match_real_with_D"] = float(res["match_real_with_D"][:, c].max())
        ch["best_inband_match_with_D"] = float(res["inband_match_with_D"][:, c].max())
        ch["energy_share_band_with_D"] = float(sD[c])
    else:
        ch["best_match_net"] = float(res["match_net"][:, c].max())
        ch["best_match_pos_L1"] = float(res["match_pos_L1"][:, c].max())
    # largest in-band match this magnitude response allows on the chosen side (phases aligned)
    sb = neg_band if neg else pos_band
    Kmag = np.abs(H[c, sb])
    Xm = Xs[j, sb]                                         # |X(|f|)|, the template is real
    ch["magnitude_bound_inband_match"] = float((Kmag * Xm).sum() / np.sqrt((Kmag ** 2).sum() * (Xs[j, pos_band] ** 2).sum()))
    # group-delay slope against the matched kernel's group delay over the channel's 10-90% band, chosen side
    sel = band & (fa_grid >= ch["f10"]) & (fa_grid <= ch["f90"]) & (P[c] > 1e-3 * P[c][band].max())
    if neg:                                # conj(k) at +f has k's group delay at -f
        idx = np.where(sel & (fr < 0))[0]
        target = np.interp(-fr[idx], fm[fm > 0], gdm[j][fm > 0])
        good = np.interp(-fr[idx], fm[fm > 0], Pm[j][fm > 0]) > 1e-3 * Pm[j][fm > 20].max()
    else:
        idx = np.where(sel & (fr > 0))[0]
        target = gdm[j, idx]
        good = Pm[j, idx] > 1e-3 * Pm[j][fm > 20].max()
    if good.sum() >= 3 and np.ptp(target[good]) > 0:
        W = P[c, idx][good]
        X_ = target[good]
        Y_ = gd[c, idx][good]
        xm_ = (W * X_).sum() / W.sum()
        ym_ = (W * Y_).sum() / W.sum()
        ch["gd_slope_vs_chirp"] = float((W * (X_ - xm_) * (Y_ - ym_)).sum() / (W * (X_ - xm_) ** 2).sum())
        ch["chirp_delay_span_in_band_s"] = float(np.ptp(X_))
        ch["gd_offset_from_chirp_s"] = float(ym_ - xm_)
    else:
        ch["gd_slope_vs_chirp"] = None
        ch["chirp_delay_span_in_band_s"] = 0.0
        ch["gd_offset_from_chirp_s"] = None

# ------------------------------------------------------------------ load check: network output on templates
gen = torch.Generator().manual_seed(5)
del H, gd, P, Hm, gdm, Pm, Xs
m = m.float()
if body is not None and body.kernel_type == "s4d":
    # kernels verified against the module's code above; the module's own call needs > 4 GB at d_state 512
    kk = torch.tensor(k, dtype=torch.complex64)
    body.kernels = lambda: kk
B = 5
with torch.no_grad():
    X2 = np.stack([kc.templates(2048, 0)[0], kc.templates(2048, 1)[0]], 1)   # (J, 2, 8192)
    rho = np.sqrt((X2 ** 2).sum((-1, -2)))
    out = {}
    for snr in (12.0, 30.0):
        Xs_ = torch.tensor(X2 * (snr / rho)[:, None, None], dtype=torch.float32)
        if kind == "long_conv":
            ests = []
            for r in range(4):
                Xn = Xs_ + torch.randn(Xs_.shape, generator=gen)
                logits = torch.cat([m(Xn[i:i + B]) for i in range(0, J, B)])
                ests.append(m.estimate(logits)[0].exp().double().numpy())
            est = np.concatenate(ests)
            truth = np.tile(mc, 4)
            tag = "white_noise_4_draws"
        else:
            y = torch.cat([m(Xs_[i:i + B]) for i in range(0, J, B)]).double()
            sd = ck["state_dict"]
            est = y[:, 0].numpy() * float(sd["y_std"][0]) + float(sd["y_mean"][0])
            truth = mc
            tag = "noiseless"
        err = np.abs(est - truth) / truth
        out[f"network_snr{int(snr)}_{tag}_within_1pct"] = float((err < 0.01).mean())
        out[f"network_snr{int(snr)}_{tag}_within_2pct"] = float((err < 0.02).mean())
    checks["load_check_on_bank_templates"] = out

info["checks"] = checks
info["seconds"] = round(time.time() - t0, 1)
with open(os.path.join(kc.HERE, f"{name}_channels.json"), "w") as fh:
    json.dump({"info": info, "channels": chan}, fh, indent=1)
print(json.dumps(info, indent=1))
bm = np.array([c["best_inband_match"] if c["best_inband_match"] is not None else np.nan for c in chan])
print("best in-band match per channel: median %.4f, 90th pct %.4f, max %.4f" % (np.nanmedian(bm), np.nanpercentile(bm, 90), np.nanmax(bm)))
bp = np.array([c["best_match_real"] if real_kernel else c["best_match_pos"] for c in chan])
print("best match (unit-norm kernel; real kernels: phase-maximized) per channel: median %.4f, 90th pct %.4f, max %.4f" % (np.median(bp), np.percentile(bp, 90), bp.max()))
