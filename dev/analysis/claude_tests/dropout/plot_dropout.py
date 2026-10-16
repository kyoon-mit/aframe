"""Figures for the 15 October dropout update.

dropout_curves.png: validation within 1% against epoch for the four S4D 64 x 32 x 4 runs
(curves.json from read_wandb.py), plus the dropout-0.2 max run on its training batches.
dropout_same_windows.png: the same weights with dropout off and on, on the cached clean
windows (diag_cpu.json): (a) each run's latest checkpoint, (b) the dropout-0.2 max run by
epoch (epochs 5 and 30 from the 14 October log).
dropout_maxima.png: mean of the 64 maxima with dropout off and on, dropout-0.2 max run.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator, PercentFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EMAX = 60
INK, INK2, MUTED, GRID = "#0f1e33", "#34435a", "#5d6b80", "#e1e5ec"
RED, BLUE, AQUA, GREY = "#e34948", "#1f5596", "#1baf7a", "#a7b0bd"
plt.rcParams.update({
    "font.family": ["IBM Plex Sans", "DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
})

cur = json.load(open(f"{HERE}/curves.json"))["runs"]
diag = json.load(open(f"{HERE}/diag_cpu.json"))


def on_mean(tag, key):
    return sum(o[key] for o in diag[tag]["on"]) / len(diag[tag]["on"])


# ---- figure 1: validation curves ----------------------------------------------------
SER = [
    ("max_p0", "real, maximum over time, dropout 0", BLUE),
    ("energy_max_p02", "complex, energy then maximum, dropout 0.2", AQUA),
    ("mean_p02", "real, mean over time, dropout 0.2", GREY),
    ("max_p02", "real, maximum over time, dropout 0.2", RED),
]
fig, ax = plt.subplots(figsize=(8.6, 5.4))
for tag, label, color in SER:
    r = cur[tag]
    pts = [(e, v) for e, v in zip(r["epoch"], r["val_within_1"]) if e <= EMAX]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, label=label, zorder=3)
    ax.plot([pts[-1][0]], [pts[-1][1]], "o", ms=6, color=color, mec="white", mew=2, zorder=4)
tr = cur["max_p02"]["train_within_1"]
te = sorted(int(k) for k in tr if int(k) <= EMAX)
ax.plot(te, [tr[str(k)] for k in te], color=RED, ls=(0, (4, 3)), lw=1.6, zorder=2,
        label="red run on its training batches, dropout on")
ax.set_xlim(0, EMAX + 1)
ax.set_ylim(0, 1.02)
ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
ax.set_xlabel("epoch (one pass of 50,048 training windows)")
ax.set_ylabel("validation events within 1%")
ax.set_title("S4D 64 × 32 × 4 on the clean signal: validation, dropout off", loc="left")
h, l = ax.get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.6, handlelength=3.0)
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(f"{HERE}/dropout_curves.png", dpi=110)
plt.close(fig)

# ---- figure 2: same weights, same windows ------------------------------------------
p0 = "max_p0_latest" if "max_p0_latest" in diag else "max_p0_last"
RUNS = [("max_p02_last", RED, "real max\ndropout 0.2"), (p0, BLUE, "real max\ndropout 0"),
        ("mean_p02_last", GREY, "real mean\ndropout 0.2"), ("emax_p02_last", AQUA, "complex\nenergy max\ndropout 0.2")]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
for i, (tag, color, name) in enumerate(RUNS):
    r = diag[tag]
    a1.plot([i - 0.2], [r["off"]["within_1pct"]], "o", ms=9, color=color, mec="white", mew=1.5, zorder=3)
    a1.plot([i], [on_mean(tag, "within_1pct")], "o", ms=9, mfc="white", color=color, mew=2, zorder=3)
    a1.plot([i + 0.2], [r["mc"]["within_1pct_mc"]], "D", ms=8, color=color, mec="white", mew=1.5, zorder=3)
a1.set_xticks(range(len(RUNS)))
a1.set_xticklabels([f"{n}\nepoch {diag[t]['epoch']}" for t, _, n in RUNS], fontsize=8.4)
a1.set_xlim(-0.5, len(RUNS) - 0.5)
a1.set_ylim(0, 1.0)
a1.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a1.set_ylabel("cached windows within 1%")
a1.set_title("Latest checkpoint of each run", loc="left")
a1.grid(axis="x", visible=False)
keys = [Line2D([], [], ls="", marker="o", ms=8, color=INK2, mec="white", label="dropout off"),
        Line2D([], [], ls="", marker="o", ms=8, mfc="white", color=INK2, mew=2, label="dropout on"),
        Line2D([], [], ls="", marker="D", ms=7, color=INK2, mec="white", label="average of 16 patterns")]
a1.legend(handles=keys, loc="center", fontsize=8.2, bbox_to_anchor=(0.40, 0.18))

# epochs 5 and 30: 14 October log (EXPERIMENTS.md, 14:10); 63 and 232: diag_cpu.py
ep = [5, 30, 63, 232]
off = [0.147, 0.001, diag["max_p02_ep63"]["off"]["within_1pct"], diag["max_p02_last"]["off"]["within_1pct"]]
on = [0.046, 0.313, on_mean("max_p02_ep63", "within_1pct"), on_mean("max_p02_last", "within_1pct")]
mc = [diag[t]["mc"]["within_1pct_mc"] for t in ("max_p02_ep63", "max_p02_last")]
a2.plot(ep, off, "-o", color=RED, ms=7, mec="white", mew=1.5, label="dropout off (evaluation)")
a2.plot(ep, on, "--o", color=RED, ms=7, mfc="white", mew=1.6, label="dropout on (training mode)")
a2.plot([63, 232], mc, ":D", color=RED, ms=7, mec="white", mew=1.5, label="average of 16 patterns")
a2.set_xscale("log")
a2.xaxis.set_major_locator(FixedLocator(ep))
a2.xaxis.set_minor_locator(NullLocator())
a2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
a2.set_ylim(0, 1.0)
a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a2.set_xlabel("epoch of the checkpoint")
a2.set_title("Real max, dropout 0.2, by epoch", loc="left")
a2.legend(loc="center left", fontsize=8.2, bbox_to_anchor=(0.0, 0.62), handlelength=3.6)
fig.tight_layout()
fig.savefig(f"{HERE}/dropout_same_windows.png", dpi=110)
plt.close(fig)

# ---- figure 3: what the read-out sees --------------------------------------------
f_ep = [30, 63, 232]
f_off = [0.905, diag["max_p02_ep63"]["off"]["mean_feature"], diag["max_p02_last"]["off"]["mean_feature"]]
f_on = [1.184, on_mean("max_p02_ep63", "mean_feature"), on_mean("max_p02_last", "mean_feature")]
fig, ax = plt.subplots(figsize=(5.2, 3.7))
ax.plot(f_ep, f_off, "-o", color=RED, ms=7, mec="white", mew=1.5, label="dropout off")
ax.plot(f_ep, f_on, "--o", color=RED, ms=7, mfc="white", mew=1.6, label="dropout on")
for x, yo, yn in zip(f_ep, f_off, f_on):
    ax.annotate(f"×{yn / yo:.2f}", xy=(x, yn), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8.6, color=INK2)
ax.set_xscale("log")
ax.xaxis.set_major_locator(FixedLocator(f_ep))
ax.xaxis.set_minor_locator(NullLocator())
ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
ax.set_xlim(24, 290)
ax.set_ylim(0, 1.4)
ax.set_xlabel("epoch of the checkpoint")
ax.set_ylabel("mean of the 64 maxima (network units)")
ax.set_title("Real max, dropout 0.2: what the read-out sees", loc="left", fontsize=10)
ax.legend(loc="lower left", fontsize=8.4, handlelength=3.6)
fig.tight_layout()
fig.savefig(f"{HERE}/dropout_maxima.png", dpi=110)
plt.close(fig)
print("ok")
