"""Validation within 1/2/5/10% against epoch for the three signal-only runs."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.load(open(HERE / "val_within.json"))
RUNS = [("sig_only_reg_free_kernel", "free kernels (256 × 4 s)", "#1baf7a"),
        ("sig_only_reg_s4d_256x256x4", "S4D 256 × 256 × 4, mean pool", "#13386b"),
        ("sig_only_reg_s4d_64x32x4", "S4D 64 × 32 × 4, mean pool", "#2a78d6")]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    "axes.titlecolor": INK, "axes.titlesize": 10,
})
fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), sharey=True)
for i, (ax, tol) in enumerate(zip(axes.flat, (1, 2, 5, 10))):
    for name, label, c in RUNS:
        eps = sorted(int(e) for e in d[name])
        ax.plot(eps, [d[name][str(e)][i] for e in eps], color=c, marker="o", ms=2.5, label=label)
    ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_title(f"within {tol}%"); ax.set_xlabel("epoch"); ax.set_ylabel("fraction of validation events")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=9)
fig.suptitle("Chirp mass from the clean signal (no noise): validation, power law −3 from SNR 4", color=INK)
fig.tight_layout(rect=(0, 0.05, 1, 0.97))
fig.savefig(HERE / "val_within.png", dpi=110)
