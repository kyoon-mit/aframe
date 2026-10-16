"""Comparison plots from the 50k-window tests.

usage: plot_compare.py OUT_DIR LABEL=TEST_DIR [LABEL=TEST_DIR ...]
Per population: within 1/2/5/10% against SNR (2x2, shared y), and predicted
against true chirp mass (median and 1 sigma band, equal axes) for SNR 8-12
and all SNR.
"""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1])
RUNS = [a.split("=", 1) for a in sys.argv[2:]]
C = ["#13386b", "#1baf7a", "#2a78d6", "#8a96a8", "#c0392b", "#7d3c98"]
LS = ["-", "-", "--", ":", "-.", "--"]
INK, MUTED, GRID = "#0F1E33", "#5b6574", "#e3e6ea"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.8,
    "axes.titlecolor": INK, "axes.titlesize": 10,
})
pct = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
TITLE = {"snr4_powerlaw": "power law (index −3), SNR 4 to 50",
         "snr8_powerlaw": "power law (index −3), SNR 8 to 50",
         "snr4_uniform": "uniform SNR 4 to 50"}

for pop, title in TITLE.items():
    data = {}
    for label, d in RUNS:
        f = Path(d) / pop / "param_est_results.csv"
        if f.exists():
            r = pd.read_csv(f)
            data[label] = (r["chirp_mass_true"].to_numpy(), r["chirp_mass_pred"].to_numpy(), r["snr"].to_numpy())
    out = OUT / pop
    out.mkdir(parents=True, exist_ok=True)
    lo = 8 if pop == "snr8_powerlaw" else 4
    edges = np.array([e for e in (4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 35, 50) if e >= lo])
    centers = np.sqrt(edges[:-1] * edges[1:])
    ticks = [t for t in (4, 6, 8, 12, 16, 25, 50) if t >= lo]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
    for ax, tol in zip(axes.flat, (0.01, 0.02, 0.05, 0.10)):
        for (label, (t, p, s)), c, ls in zip(data.items(), C, LS):
            ok = np.abs(p - t) / t < tol
            frac = [ok[(s >= a) & (s < b)].mean() for a, b in zip(edges[:-1], edges[1:])]
            ax.plot(centers, frac, color=c, ls=ls, marker="o", ms=3, label=label)
        ax.set_xscale("log"); ax.set_xticks(ticks, [str(x) for x in ticks]); ax.minorticks_off()
        ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"within {tol:.0%}"); ax.set_xlabel("injected SNR"); ax.set_ylabel("fraction of events")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), fontsize=8)
    fig.suptitle(f"Chirp mass on 50k test windows: {title}", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out / "chirp_mass_compare.png", dpi=110); plt.close(fig)

    bins = np.linspace(0.85, 2.75, 39); mid = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (band, (a, b)) in zip(axes, (("SNR 8 to 12", (8, 12)), ("all SNR", (0, 1e9)))):
        for (label, (t, p, s)), c, ls in zip(data.items(), C, LS):
            sel = (s >= a) & (s < b)
            q = np.array([np.percentile(p[sel & (t >= x) & (t < y)], [16, 50, 84])
                          if (sel & (t >= x) & (t < y)).sum() > 4 else [np.nan] * 3
                          for x, y in zip(bins[:-1], bins[1:])])
            ax.fill_between(mid, q[:, 0], q[:, 2], color=c, alpha=0.12, lw=0)
            ax.plot(mid, q[:, 1], color=c, ls=ls, label=label)
        ax.plot((0.85, 2.75), (0.85, 2.75), color=MUTED, ls=":", lw=1.2)
        ax.set_xlim(0.85, 2.75); ax.set_ylim(0.85, 2.75); ax.set_aspect("equal")
        ax.set_xlabel("true chirp mass (M☉)"); ax.set_ylabel("predicted chirp mass (M☉)")
        ax.set_title(f"{band}: median and 1σ band")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "chirp_mass_pred_vs_true.png", dpi=110); plt.close(fig)
    print("wrote", out)
