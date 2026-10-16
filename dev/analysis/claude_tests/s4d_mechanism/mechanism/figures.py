"""Figures for the mechanism tests (static PNG)."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "kernels"))
import kcommon as kc  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"s4d_ck69": "#2a78d6", "free_ck46": "#eb6834", "resnet_ck37": "#1baf7a", "bank": "#eda100"}
NAME = {"s4d_ck69": "clean S4D 64x32x4 (ck_69)", "free_ck46": "free kernels (ck_46)",
        "resnet_ck37": "ResNet (ck_37)", "bank": "coherent template bank"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
                     "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False})
true = np.load(os.path.join(os.path.dirname(HERE), "harness", "data", "clean_chirp_mass.npy")).astype(float)
merger = np.load(os.path.join(os.path.dirname(HERE), "harness", "data", "clean_merger_sample.npy"))


def w1(path):
    return float((np.abs(np.load(os.path.join(HERE, path)) - true) / true < 0.01).mean())


# ------------------------------------------------ 1. T2/T3 with controls
rows = [("clean (no change)", "est_clean_{k}.npy", "pre"),
        ("taper only (control)", "est_taper_{k}.npy", "pre"),
        ("T2 segment shuffle", "est_T2_{k}.npy", "pre"),
        ("T3 segment phase", "est_T3_{k}.npy", "pre"),
        ("ramp at window start only", "est_diag_ramp_start_only_{k}.npy", "post"),
        ("ramp at window end only", "est_diag_ramp_end_only_{k}.npy", "post"),
        ("first 31 ms set to zero", "est_diag_zero_first_64_samples_{k}.npy", "post"),
        ("internal ramps only (control)", "est_diag_internal_ramps_only_{k}.npy", "post"),
        ("T2, internal ramps only", "est_diag_T2_internal_ramps_only_{k}.npy", "post"),
        ("T3, internal ramps only", "est_diag_T3_internal_ramps_only_{k}.npy", "post")]
t2 = dict(np.load(os.path.join(HERE, "manip_T2.npz")))
fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True)
ypos = np.array([0, 1, 2, 3, 4.6, 5.6, 6.6, 7.6, 8.6, 9.6])
for ax, k in zip(axes, COL):
    vals = []
    for (lab, pat, kind), y in zip(rows, ypos):
        v = w1(pat.format(k=k))
        vals.append(v)
        ax.barh(y, v, height=0.62, color=COL[k], alpha=1.0 if kind == "pre" else 0.55)
        ax.text(v + 0.02, y, f"{100 * v:.1f}%", va="center", fontsize=8, color=INK2)
    ax.set_title(NAME[k], color=INK)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("within 1% of the true chirp mass")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(ypos)
axes[0].set_yticklabels([r[0] for r in rows])
axes[0].invert_yaxis()
axes[0].text(-0.02, -0.75, "pre-registered (solid)", transform=axes[0].get_yaxis_transform(), ha="right", fontsize=8, color=INK2)
axes[0].text(-0.02, 4.0, "post hoc (light)", transform=axes[0].get_yaxis_transform(), ha="right", fontsize=8, color=INK2)
fig.suptitle("T2 and T3 controls: accuracy under each change, all 4,000 clean windows (pre-registered T2 numbers use the 3,998 passing the in-step check; differences under 0.03 points)",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_T2T3_controls.png"), dpi=150)
plt.close(fig)

# ------------------------------------------------ 2. T4 peak times
r = json.load(open(os.path.join(HERE, "results_T4.json")))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
fgrid = np.geomspace(20, 160, 200)
tm_med = np.median(merger) / 2048
for ax, (key, title) in zip(axes, [("s4d_ck69", "clean S4D first layer (ck_69), 28 channels"),
                                   ("free_ck46", "free kernels (ck_46), 87 channels")]):
    s = r[key]
    f = np.array(s["channel_freq_Hz"])
    ax.axhspan(-0.1, 0.1, color="#f0efec", zorder=0)
    for mcv, lw in [(0.85, 1.0), (1.8, 1.0), (2.75, 1.0)]:
        tau = kc.tau_of_f(fgrid, mcv)
        ax.plot(fgrid, np.maximum(-tau, -tm_med), color=INK2, lw=lw, zorder=1)
        j = np.argmin(np.abs(np.maximum(-tau, -tm_med) + 1.6))
        ax.text(fgrid[j] * 1.02, -1.6, f"{mcv} M$_\\odot$", fontsize=7, color=INK2, va="bottom")
    ax.scatter(f, s["channel_median_delta_s"], s=34, color=COL[key], edgecolor=SURF, linewidth=1.5, zorder=3,
               label="channel output (as the network computes it)")
    if key == "s4d_ck69":
        s2 = r["s4d_ck69_kernel_only_noD_descriptive"]
        ax.scatter(f, s2["channel_median_delta_s"], s=34, facecolor="none", edgecolor=COL[key], linewidth=1.2,
                   zorder=2, label="kernel only, skip term D removed (post hoc)")
    ax.set_xscale("log")
    ax.set_xticks([20, 30, 50, 100, 150])
    ax.set_xticklabels(["20", "30", "50", "100", "150"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("channel frequency (Hz)")
    ax.set_title(title, color=INK)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.16), fontsize=8, frameon=False, ncol=2)
axes[0].set_ylabel("median peak time minus merger (s)")
axes[0].set_ylim(-4.0, 0.3)
axes[0].text(21, 0.13, "within 0.1 s of merger", fontsize=7, color=INK2)
for ax in axes:
    ax.text(21, -2.2, "grey lines: time the chirp passes f,\nfor three chirp masses (floor = window\nstart at the median merger time)", fontsize=7, color=INK2)
fig.suptitle("T4: where each first-layer channel's envelope peaks, against when the chirp passes its frequency (4,000 clean windows)",
             fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_T4_peak_times.png"), dpi=150, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ 3. T1 sub-windows
t1 = json.load(open(os.path.join(HERE, "results_T1.json")))["variants"]
shades = {0.25: "#86b6ef", 0.5: "#3987e5", 1: "#1c5cab", 2: "#0d366b"}
fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
for ax, k in zip(axes, COL):
    for Ls, c in shades.items():
        items = sorted([(v["end_before_merger_s"], v[k]["within_1pct"]) for v in t1.values()
                        if v["length_s"] == Ls and v["end_before_merger_s"] is not None])
        ax.plot([a for a, _ in items], [b for _, b in items], color=c, lw=2, marker="o", ms=4,
                label=f"{Ls} s kept")
    ax.axhline(t1["full"][k]["within_1pct"], color=INK2, lw=1)
    ax.text(3.05, t1["full"][k]["within_1pct"] + 0.02, "full 4 s, ramped edges", fontsize=7, color=INK2, ha="right")
    ax.set_title(NAME[k], color=INK)
    ax.set_xlabel("kept stretch ends this long before merger (s)")
    ax.set_ylim(0, 1.05)
axes[0].set_ylabel("within 1%")
axes[0].legend(fontsize=7, frameon=False, loc="center right")
fig.suptitle("T1 (descriptive): accuracy when only a ramped stretch of the chirp is kept", fontsize=10, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_T1_subwindows.png"), dpi=150)
plt.close(fig)
print("ok")
