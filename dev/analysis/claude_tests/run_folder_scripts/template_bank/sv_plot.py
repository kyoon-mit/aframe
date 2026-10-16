"""Sensitive volume against false-alarm rate, one panel per mass pair, shared y."""
import sys
from pathlib import Path
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1]); runs = [a.split("=", 1) for a in sys.argv[2:]]
C = ["#13386b", "#1baf7a", "#2a78d6", "#8a96a8", "#c0392b"]
LS = ["-", "--", ":", "-.", "-"]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.8, "axes.titlecolor": INK, "axes.titlesize": 10})
ref = h5py.File(Path(runs[0][1]) / "gwtc-3_pipeline_sv.h5", "r")
fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True, sharex=True)
for ax, combo in zip(axes.flat, ("1.4-1.4", "1.6-1.6", "1.8-1.8", "2.0-2.0")):
    far = ref["far"][:]
    for (label, d), c, ls in zip(runs, C, LS):
        sv = h5py.File(Path(d) / "sensitive_volume.h5", "r")[combo]
        ax.errorbar(far, sv["sv"][:], yerr=sv["err"][:], color=c, ls=ls, marker="o", ms=3, capsize=2, label=label)
    ax.plot(far, ref["mbta"][combo]["sv"][:], color="#c0392b", lw=1.4, label="MBTA (O3)")
    ax.plot(far, ref["pycbc_hyperbank"][combo]["sv"][:], color=MUTED, lw=1.2, ls="--", label="PyCBC (O3)")
    m1, m2 = combo.split("-")
    ax.set_title(f"{m1} + {m2} M☉"); ax.set_yscale("log")
    ax.set_xlabel("false-alarm rate (per year)"); ax.set_ylabel("sensitive volume (Gpc³)")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=3, fontsize=8)
fig.suptitle("Sensitive volume over O3b background", color=INK, fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(OUT, dpi=110); print("wrote", OUT)
