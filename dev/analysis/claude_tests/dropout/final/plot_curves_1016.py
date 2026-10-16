"""dropout_curves.png for the 16 October dropout update.

Validation within 1% against epoch for the five S4D 64 x 32 x 4 signal-only runs
(curves_1016.json from read_wandb_1016.py), split by read-out: maximum over time
(left) and mean over time (right). Dashed: the dropout-0.2 runs on their own
training batches, with dropout on.

Usage: python plot_curves_1016.py OUT.png
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator, PercentFormatter  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]

# site tokens (docs/assets/css/main.css): ink, ink-2, ink-3, rule, verdict, warn;
# aqua is the colour of the energy read-outs in earlier BNS figures
INK, INK2, INK3, RULE = "#0f1e33", "#34435a", "#5d6b80", "#e1e5ec"
P0, P02, ENERGY = "#1f5596", "#b42318", "#1baf7a"

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 10,
    "axes.edgecolor": INK3, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": RULE, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    "axes.titlecolor": INK, "axes.titlesize": 10.5, "figure.facecolor": "white",
    "axes.facecolor": "white", "savefig.facecolor": "white", "text.color": INK,
})

cur = json.load(open(f"{HERE}/curves_1016.json"))
runs = cur["runs"]


def val(tag):
    r = runs[tag]
    return r["epoch"], r["val_within_1"]


def train(tag):
    tr = runs[tag]["train_within_1"]
    ep = sorted(int(k) for k in tr)
    return ep, [tr[str(e)] for e in ep]


def line(ax, x, y, color, ls="-", lw=1.6, z=3, end=True, halo=False):
    fx = [pe.Stroke(linewidth=lw + 2.2, foreground="white"), pe.Normal()] if halo else None
    ax.plot(x, y, color=color, ls=ls, lw=lw, zorder=z, path_effects=fx)
    if end:
        ax.plot([x[-1]], [y[-1]], "o", ms=5.5, color=color, mec="white", mew=1.5, zorder=z + 1)


def label(ax, x, y, text, va="center", ha="left"):
    ax.annotate(text, xy=(x, y), xytext=(6 if ha == "left" else -6, 0), textcoords="offset points",
                ha=ha, va=va, fontsize=8.4, color=INK2, zorder=6)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.8, 4.7), sharey=True)
XMAX = 300

# left: maximum over time
e, v = val("max_p0")
line(a1, e, v, P0)
label(a1, 150, 0.935, f"real, dropout 0 (to epoch {e[-1]})", ha="center")
e, v = val("energy_max_p02")
line(a1, e, v, ENERGY, z=4, halo=True)
label(a1, e[-1], v[-1] - 0.06, "complex, energy and\nmaximum, dropout 0.2", va="top")
e, v = train("max_p02")
line(a1, e, v, P02, ls=(0, (4, 2.5)), lw=1.4, z=2, end=False)
label(a1, e[-1], v[-1], "real, dropout 0.2,\nits training batches\nwith dropout on", va="center")
e, v = val("max_p02")
line(a1, e, v, P02)
label(a1, 120, 0.235, "real, dropout 0.2", ha="center")
a1.set_title("Maximum over time", loc="left")
a1.set_ylabel("events within 1% of the true chirp mass")

# right: mean over time
e, v = val("mean_p0")
line(a2, e, v, P0)
label(a2, 120, 0.935, f"real, dropout 0 (to epoch {e[-1]})", ha="center")
e, v = train("mean_p02")
line(a2, e, v, P02, ls=(0, (4, 2.5)), lw=1.4, z=2, end=False)
label(a2, e[-1], v[-1] + 0.03, "real, dropout 0.2, its training\nbatches with dropout on", va="bottom")
e, v = val("mean_p02")
line(a2, e, v, P02)
label(a2, e[-1], v[-1] - 0.03, "real, dropout 0.2", va="top")
a2.set_title("Mean over time", loc="left")

for ax in (a1, a2):
    ax.set_xlim(0, XMAX)
    ax.xaxis.set_major_locator(MultipleLocator(50))
    ax.set_xlabel("epoch (50,048 training windows each)")
    ax.set_ylim(0, 1.02)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))

keys = [Line2D([], [], color=P0, lw=2, label="real S4D, dropout 0: validation (dropout off)"),
        Line2D([], [], color=P02, lw=2, label="real S4D, dropout 0.2: validation (dropout off)"),
        Line2D([], [], color=ENERGY, lw=2, label="complex last layer, dropout 0.2: validation (dropout off)"),
        Line2D([], [], color=P02, lw=1.6, ls=(0, (4, 2.5)),
               label="real S4D, dropout 0.2: its training batches (dropout on)")]
fig.legend(handles=keys, loc="lower center", ncol=2, fontsize=8.6, handlelength=3.0,
           columnspacing=2.0)
fig.suptitle("S4D 64 × 32 × 4 on the clean signal: validation within 1% by epoch",
             x=0.01, ha="left", fontsize=11, color=INK)
fig.tight_layout(rect=(0, 0.11, 1, 0.97))
fig.savefig(OUT, dpi=110)
print("wrote", OUT, os.path.getsize(OUT), "bytes; data read at", cur["read_at"])
