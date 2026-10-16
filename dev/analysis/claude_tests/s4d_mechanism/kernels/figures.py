"""Figures for kernels.md (static PNG, light surface)."""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

H = kc.HERE
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = LinearSegmentedColormap.from_list("seqblue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
HEAT = LinearSegmentedColormap.from_list("seqblue_full", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.linewidth": 0.8, "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
    "legend.frameon": False, "font.family": "DejaVu Sans",
})
LABEL = {
    "noisy_s4d_d64": "Noisy, S4D 32 tones/channel (d_state 64)",
    "noisy_s4d_d512": "Noisy, S4D 256 tones/channel (d_state 512)",
    "noisy_s4d_d512_band": "Noisy, S4D 256 tones, band start",
    "noisy_free": "Noisy, free 4 s kernels",
    "clean_s4d_layer1": "Clean, S4D 64x32x4, first layer",
    "clean_free": "Clean, free 4 s kernels",
}
S4D = ["noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "clean_s4d_layer1"]
ALL = ["noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "noisy_free", "clean_s4d_layer1", "clean_free"]
summary = json.load(open(os.path.join(H, "summary.json")))
tables = {n: json.load(open(os.path.join(H, f"{n}_table.json"))) for n in ALL}


def modes_inband(n):
    z = np.load(os.path.join(H, f"{n}_modes.npz"))
    inb = z["inband"]
    return np.abs(z["f"][inb]), z["ring"][inb], z["weight"][inb]


def init_inband(n):
    z = np.load(os.path.join(H, f"{n}_init_modes.npz"))
    fa = np.abs(z["f"])
    nyq = 1024 if n.startswith("clean") else 512
    inb = (fa >= 20) & (fa < nyq)
    return fa[inb], z["ring"][inb]


# ------------------------------------------------------------------ 1. ring time vs tau(f)
fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
xx = np.geomspace(1e-3, 1e3, 200)
for ax, n in zip(axs.ravel(), S4D):
    f, ring, w = modes_inband(n)
    o = np.argsort(w)
    tau = kc.tau_of_f(f, 1.2)
    fi, ri = init_inband(n)
    ax.scatter(kc.tau_of_f(fi, 1.2), ri, s=1.5, color=AXIS, linewidths=0, rasterized=True, label="same network at its start (fresh draw)")
    sc = ax.scatter(tau[o], ring[o], c=w[o], s=2, cmap=SEQ, norm=LogNorm(1e-5, 1), linewidths=0, rasterized=True)
    ax.fill_between(xx, xx * (1.2 / 2.75) ** (5 / 3), xx * (1.2 / 0.85) ** (5 / 3), color=ORANGE, alpha=0.18, lw=0,
                    label="ring time = τ(f), chirp mass 0.85-2.75")
    ax.plot(xx, xx, color=ORANGE, lw=1.4, label="ring time = τ(f), 1.2 M☉")
    fl = np.geomspace(20, 1024, 200)
    ax.plot(kc.tau_of_f(fl, 1.2), kc.t_step(fl, 1.2), color=AQUA, lw=1.4, label="in-step time ḟ^(-1/2), 1.2 M☉")
    ax.axvline(kc.MERGE / 2048, color=MUTED, lw=1)
    ax.text(kc.MERGE / 2048 * 1.15, 1.6e-4, "chirp starts in the window\n(τ = 3.75 s)", color=INK2, fontsize=8,
            bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1))
    ax.axhline(4.0, color=MUTED, lw=1)
    ax.text(1.3e-3, 5.0, "kernel length 4 s", color=INK2, fontsize=8, bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-3, 400)
    ax.set_ylim(1e-4, 40)
    ax.set_xlabel("time before merger τ(f) at the resonance frequency, chirp mass 1.2 M☉ (s)")
    ax.set_ylabel("ring time 1/(a·dt·fs) (s)")
    nm = summary[n]["modes"]
    ax.set_title(f"{LABEL[n]}\n{nm['inband_20Hz_to_nyquist']} of {nm['total']} resonances in 20 Hz to Nyquist")
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=4, fontsize=8.5, markerscale=6)
cb = fig.colorbar(sc, ax=axs, shrink=0.6, label="trained resonance's share of its channel's energy")
fig.savefig(os.path.join(H, "fig_ring_vs_hold.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 2. ring time vs frequency
fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
for ax, n in zip(axs.ravel(), S4D):
    f, ring, w = modes_inband(n)
    o = np.argsort(w)
    nyq = 1024 if n.startswith("clean") else 512
    fi, ri = init_inband(n)
    ax.scatter(fi, ri, s=1.5, color=AXIS, linewidths=0, rasterized=True, label="same network at its start (fresh draw)")
    sc = ax.scatter(f[o], ring[o], c=w[o], s=2, cmap=SEQ, norm=LogNorm(1e-5, 1), linewidths=0, rasterized=True)
    fl = np.geomspace(20, nyq, 300)
    ax.fill_between(fl, kc.tau_of_f(fl, 2.75), kc.tau_of_f(fl, 0.85), color=ORANGE, alpha=0.18, lw=0,
                    label="time before merger τ(f), 0.85-2.75 M☉")
    ax.fill_between(fl, kc.t_step(fl, 2.75), kc.t_step(fl, 0.85), color=AQUA, alpha=0.22, lw=0,
                    label="in-step time ḟ^(-1/2), 0.85-2.75 M☉")
    ax.axvspan(20, kc.f_of_tau(kc.MERGE / 2048, 2.75), color=GRID, alpha=0.6, lw=0)
    ax.axvline(kc.f_of_tau(kc.MERGE / 2048, 0.85), color=MUTED, lw=1)
    ax.text(21, 1.5e-4, "no chirp in the\nwindow (any mass)", color=INK2, fontsize=8,
            bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1))
    ax.text(kc.f_of_tau(kc.MERGE / 2048, 0.85) * 1.04, 1.5e-4, "chirp in the window\nfor every mass", color=INK2, fontsize=8,
            bbox=dict(facecolor=SURF, edgecolor="none", alpha=0.85, pad=1))
    ax.axhline(4.0, color=MUTED, lw=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(20, nyq)
    ax.set_ylim(1e-4, 40)
    ax.set_xlabel("resonance frequency (Hz)")
    ax.set_ylabel("ring time 1/(a·dt·fs) (s)")
    ax.set_title(LABEL[n])
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, fontsize=8.5, markerscale=6)
fig.colorbar(sc, ax=axs, shrink=0.6, label="trained resonance's share of its channel's energy")
fig.savefig(os.path.join(H, "fig_ring_vs_freq.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 3. match histograms
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
bins = np.arange(0, 1.0001, 0.025)
for ax, n in zip(axs.ravel(), ALL):
    ch = tables[n]["channels"]
    mi = np.array([r["best_inband_match"] for r in ch], float)
    mu = np.array([r["best_match_real"] if tables[n]["info"]["real_kernel"] else r["best_match_pos"] for r in ch])
    ax.hist(mi, bins=bins, color=BLUE, rwidth=0.85, label="in-band match (kernel restricted to 20 Hz-Nyquist)")
    ax.hist(mu, bins=bins, histtype="step", color=ORANGE, lw=1.4, label="match of the unit-norm kernel")
    b = np.median([r["one_resonance_bound"] for r in ch])
    ax.axvline(b, color=INK2, lw=1.2)
    ax.text(b + 0.01, ax.get_ylim()[1] * 0.92, f"one resonance,\nbest ring time:\nmedian {b:.2f}", color=INK2, fontsize=8, va="top")
    cc = summary[n]["class_counts"]
    ax.text(0.98, 0.55, f"{cc['delay-line piece']} delay-line pieces\n{cc['spectrogram row']} spectrogram rows\n"
            f"{cc['long, not in step']} long, not in step\n{cc['unclear']} unclear",
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_xlim(0, 1)
    ax.set_xlabel("best match per channel")
    ax.set_ylabel("channels")
    ax.set_title(LABEL[n])
h, l = axs[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=2, fontsize=8.5)
fig.savefig(os.path.join(H, "fig_match_hist.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 4. match against the one-resonance bound
fig, axs = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
cols = {"spectrogram row": MUTED, "delay-line piece": BLUE, "long, not in step": ORANGE, "unclear": AQUA}
for ax, n in zip(axs.ravel(), ALL):
    ch = tables[n]["channels"]
    for k, col in cols.items():
        rr = [r for r in ch if r["class"] == k]
        if rr:
            ax.scatter([r["one_resonance_bound"] for r in rr], [r["best_inband_match"] for r in rr], s=10, color=col,
                       edgecolors=SURF, linewidths=0.5, label=f"{k} ({len(rr)})", zorder=3)
    xg = np.linspace(0, 1, 50)
    ax.plot(xg, xg, color=INK2, lw=1)
    ax.plot(xg, np.sqrt(2) * xg, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.33, 0.30, "G = 1", color=INK2, fontsize=8, rotation=0)
    ax.text(0.23, 0.42, "G = 2", color=INK2, fontsize=8)
    ax.set_xlim(0, 0.45)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("best one-resonance match in the channel's band")
    ax.set_ylabel("channel's in-band match")
    ax.set_title(LABEL[n])
    ax.legend(loc="upper left", fontsize=8)
fig.savefig(os.path.join(H, "fig_match_vs_one_resonance.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------------------------ 5. transfer function and group delay, two channels per network
def binned(f, P, gd, width=1.0, lo=20.0, hi=None):
    """|H|^2 summed and group delay |H|^2-weighted in bins of ``width`` Hz (positive f)."""
    hi = hi or f.max()
    edges = np.arange(lo, hi + width, width)
    idx = np.digitize(f, edges) - 1
    ok = (idx >= 0) & (idx < len(edges) - 1) & np.isfinite(gd)
    Pw = np.bincount(idx[ok], weights=P[ok], minlength=len(edges) - 1)
    Gw = np.bincount(idx[ok], weights=(P * gd)[ok], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 0.5 * (edges[1:] + edges[:-1]), Pw, Gw / Pw


fig, axs = plt.subplots(2, 6, figsize=(22, 7.8), constrained_layout=True)
for col, n in enumerate(ALL):
    T = np.load(os.path.join(H, f"{n}_transfer.npz"))
    ch = tables[n]["channels"]
    fs = tables[n]["info"]["fs"]
    mi = np.array([r["best_inband_match"] for r in ch], float)
    picks = [int(np.nanargmax(mi)), int(np.argsort(np.nan_to_num(mi))[len(mi) // 2])]
    x, mcs = kc.templates(rate=int(fs), ifo=0)
    for c, color, tag in zip(picks, (BLUE, ORANGE), ("best", "median")):
        r = ch[c]
        neg = r["inband_side"] == "negative"
        fgrid = T["f"]
        side = (fgrid < 0) if neg else (fgrid > 0)
        fpl = np.abs(fgrid[side])
        P = T["absH"][c][side].astype(float) ** 2
        gdc = T["group_delay_s"][c][side].astype(float)
        fb, Pb, Gb = binned(fpl, P, np.where(np.isfinite(gdc), gdc, 0.0), hi=fs / 2)
        axs[0, col].plot(fb, 10 * np.log10(Pb / Pb.max() + 1e-12), color=color, lw=1.1,
                         label=f"ch {c} ({tag}): in-band match {r['best_inband_match']:.2f}, {r['class']}")
        if tag == "best":
            j = r["best_template"]
            xa = kc.analytic_part(x[j:j + 1])
            km = np.conj(xa[:, ::-1])
            fm, Hk, gdm = kc.transfer(km, fs, 4 * x.shape[-1])
            Pm = np.abs(Hk[0]) ** 2
            pos = fm > 0
            fmb, Pmb, Gmb = binned(fm[pos], Pm[pos], gdm[0][pos], hi=fs / 2)
            lo, hi = r["f10"] / 1.4, r["f90"] * 1.4
            keep = (fb >= lo) & (fb <= hi) & (Pb > 1e-3 * Pb.max())
            axs[1, col].plot(fb[keep], Gb[keep], color=color, lw=1.4, label=f"ch {c}: group delay ({'1 Hz bins' if fs else ''})")
            keepm = (fmb >= lo) & (fmb <= hi) & (Pmb > 1e-3 * Pmb.max())
            off = r["gd_offset_from_chirp_s"] or 0.0
            axs[1, col].plot(fmb[keepm], Gmb[keepm] + off, color=INK2, lw=1.2,
                             label=f"chirp's delay, {mcs[j]:.2f} M☉, shifted {off:+.2f} s")
            axs[1, col].axvspan(r["f10"], r["f90"], color=color, alpha=0.08, lw=0)
            axs[1, col].set_xlim(lo, hi)
            yv = np.concatenate([Gb[keep], Gmb[keepm] + off])
            yv = yv[np.isfinite(yv)]
            if len(yv):
                pad = 0.1 * (yv.max() - yv.min() + 1e-3)
                axs[1, col].set_ylim(yv.min() - pad, yv.max() + pad)
            sl = r["gd_slope_vs_chirp"]
            axs[1, col].set_title(f"ch {c}: slope {sl:.2f}" if sl is not None else f"ch {c}", fontsize=9)
    axs[0, col].set_xscale("log")
    axs[0, col].set_xlim(20, fs / 2)
    axs[0, col].set_ylim(-45, 3)
    axs[0, col].set_title(LABEL[n], fontsize=9)
    axs[0, col].set_ylabel("|H(f)|², 1 Hz bins (dB from peak)")
    axs[0, col].set_xlabel("frequency (Hz)")
    axs[1, col].set_ylabel("group delay (s)")
    axs[1, col].set_xlabel("frequency (Hz); shaded: channel's 10-90% band")
    axs[0, col].legend(loc="lower left", fontsize=7)
    axs[1, col].legend(loc="upper right", fontsize=7)
fig.savefig(os.path.join(H, "fig_transfer_examples.png"), dpi=130)
plt.close(fig)

# ------------------------------------------------------------------ 6. all channels' |H(f)|, sorted by middle frequency
fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
for ax, n in zip(axs.ravel(), ALL):
    T = np.load(os.path.join(H, f"{n}_transfer.npz"))
    ch = tables[n]["channels"]
    fs = tables[n]["info"]["fs"]
    fgrid = T["f"]
    A = T["absH"].astype(float) ** 2
    pos = fgrid >= 0
    fpos = fgrid[pos]
    fold = A[:, pos].copy()
    negpart = A[:, ~pos]
    fneg = -fgrid[~pos]
    # fold negative frequencies onto |f| (same grid spacing)
    idx = np.searchsorted(fpos, fneg)
    ok = idx < len(fpos)
    np.add.at(fold, (slice(None), idx[ok]), negpart[:, ok])
    keep = (fpos >= 20) & (fpos < fs / 2)
    fold = fold[:, keep]
    fpk = fpos[keep]
    fold /= fold.max(1, keepdims=True)
    order = np.argsort([r["f50"] for r in ch])
    im = ax.pcolormesh(fpk, np.arange(len(ch)), 10 * np.log10(fold[order] + 1e-6), cmap=HEAT, vmin=-40, vmax=0,
                       shading="nearest", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlim(20, fs / 2)
    ax.set_xlabel("|frequency| (Hz)")
    ax.set_ylabel("channel (sorted by middle frequency)")
    ax.set_title(LABEL[n])
    ax.grid(False)
fig.colorbar(im, ax=axs, shrink=0.6, label="|H(f)|² (dB from channel peak), both signs of f folded")
fig.savefig(os.path.join(H, "fig_transfer_all_channels.png"), dpi=130)
plt.close(fig)
print("figures written")

# ------------------------------------------------------------------ 7. in-band match against chirp mass, five best channels
fig, axs = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
shades = ["#0d366b", "#1c5cab", "#2a78d6", "#5598e7", "#86b6ef"]
for ax, n in zip(axs.ravel(), ALL):
    M = np.load(os.path.join(H, f"{n}_match.npz"))
    mi = M["inband_match"].astype(float)
    mc = M["chirp_mass"]
    best = np.argsort(np.nan_to_num(np.nanmax(mi, 0)))[::-1][:5]
    for r, c in enumerate(best):
        ax.plot(mc, mi[:, c], color=shades[r], lw=1.3, label=f"channel {c}")
    ax.set_xscale("log")
    ax.set_xticks([0.85, 1.0, 1.2, 1.5, 2.0, 2.75])
    ax.set_xticklabels(["0.85", "1.0", "1.2", "1.5", "2.0", "2.75"])
    ax.set_ylim(0, 1)
    ax.set_xlabel("template chirp mass (M☉)")
    ax.set_ylabel("in-band match")
    ax.set_title(LABEL[n])
    ax.legend(loc="upper left", fontsize=7.5, title="5 best channels", title_fontsize=7.5)
fig.savefig(os.path.join(H, "fig_match_vs_chirp_mass.png"), dpi=150)
plt.close(fig)
print("figure 7 written")
