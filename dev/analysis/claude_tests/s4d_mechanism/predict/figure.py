"""Within 1% against SNR, one panel per noisy network: its own trained head, a fresh MLP read-out of
the same channel features, and the 256-template bank's MLP (same in every panel). Test A + test B
windows, binned by SNR; bars are binomial standard errors."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
D, R = HERE / "data", HERE / "results"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # reference palette slots 1-3 (light)
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
NAMES = {"d64": "S4D, 32 tones (d_state 64)", "band": "S4D, 256 tones, band start",
         "d512": "S4D, 256 tones (d_state 512)", "free": "free 4 s kernels"}
EDGES = np.array([4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 35, 50, 100.01])


def lab(which):
    z = np.load(D / which / "labels.npz")
    return z["chirp_mass"].astype(np.float64), z["snr"].astype(np.float64)


mca, snra = lab("testA")
mcb, snrb = lab("testB")
MC, SNR = np.concatenate([mca, mcb]), np.concatenate([snra, snrb])
summ = json.load(open(R / "summary.json"))
match = json.load(open(R / "match_compare.json"))


def curve(tag, method):
    e = np.concatenate([np.load(R / f"est_testA__{tag}__{method}.npy"), np.load(R / f"est_testB__{tag}__{method}.npy")])
    w = np.abs(e - MC) / MC < 0.01
    x, y, s = [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (SNR >= lo) & (SNR < hi)
        p = w[m].mean()
        x.append(np.median(SNR[m]))
        y.append(100 * p)
        s.append(100 * np.sqrt(p * (1 - p) / m.sum()))
    return np.array(x), np.array(y), np.array(s)


plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF})
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
bb = summ["selection"]["bank"]["chosen"]
bx, by, bs = curve("bank", f"{bb}__n512000" if (R / f"est_testA__bank__{bb}__n512000.npy").exists() else bb)
for ax, tag in zip(axes, ["d64", "band", "d512", "free"]):
    ox, oy, os_ = curve(tag, "own")
    best = summ["selection"][tag]["chosen"]
    mx, my, ms = curve(tag, f"{best}__n512000" if (R / f"est_testA__{tag}__{best}__n512000.npy").exists() else best)
    ax.errorbar(bx, by, bs, color=AQUA, lw=2, marker="o", ms=4, capsize=0, label="256-template bank, fresh MLP (512,000 training windows)")
    ax.errorbar(mx, my, ms, color=ORANGE, lw=2, marker="o", ms=4, capsize=0, label="same channels, fresh MLP (512,000 training windows)")
    ax.errorbar(ox, oy, os_, color=BLUE, lw=2, marker="o", ms=4, capsize=0, label="trained network (own head)")
    ax.set_xscale("log")
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.set_xticklabels(["4", "8", "16", "32", "64"])
    ax.set_xlim(4, 100)
    ax.set_ylim(0, 102)
    ax.grid(True, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    m = match[tag]["inband_best_channel_match_median"]
    ax.set_title(f"{NAMES[tag]}\nbest channel in-band match {m:.2f} (median)", fontsize=10)
    ax.set_xlabel("network SNR")
axes[0].set_ylabel("within 1% of true chirp mass (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(R / "fig_accuracy_vs_snr.png", dpi=150)
print("saved", R / "fig_accuracy_vs_snr.png")
