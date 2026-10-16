"""Per-population plots for the chirp-kernel network from eval_results.npz."""

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

npz, OUT = np.load(sys.argv[1]), Path(sys.argv[2])
C = ["#13386b", "#2a78d6", "#1baf7a", "#8a96a8"]
LS = ["-", "--", ":", "-."]
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
NAMES = [("trained", "chirp kernels (trained)"), ("mf", "matched filter"),
         ("ml", "epoch-630 network"), ("untrained", "chirp kernels (before training)")]

for pop, title in TITLE.items():
    d = {k: npz[f"{pop}__{k}"] for k in ("true", "snr", "trained", "untrained", "mf", "ml")}
    out = OUT / pop
    out.mkdir(parents=True, exist_ok=True)
    lo_snr = 8 if pop == "snr8_powerlaw" else 4
    edges = np.array([e for e in (4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 25, 35, 50) if e >= lo_snr])
    centers = np.sqrt(edges[:-1] * edges[1:])
    ticks = [t for t in (4, 6, 8, 12, 16, 25, 50) if t >= lo_snr]

    def frac(est, tol):
        ok = np.abs(est - d["true"]) / d["true"] < tol
        return [ok[(d["snr"] >= a) & (d["snr"] < b)].mean() for a, b in zip(edges[:-1], edges[1:])]

    # 1. chirp kernels alone: within 1/2/5/10% against SNR
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for tol, c, ls in zip((0.01, 0.02, 0.05, 0.10), C, LS):
        ax.plot(centers, frac(d["trained"], tol), color=c, ls=ls, marker="o", ms=4, label=f"within {tol:.0%}")
    ax.set_xscale("log"); ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
    ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("injected SNR"); ax.set_ylabel("fraction of events")
    ax.set_title(f"Chirp kernels, chirp mass accuracy: {title}")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "chirp_mass_frac_within_vs_snr.png", dpi=110); plt.close(fig)

    # 2. comparison, 2x2 shared scale
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.4), sharey=True)
    for ax, tol in zip(axes.flat, (0.01, 0.02, 0.05, 0.10)):
        for (k, lab), c, ls in zip(NAMES, C, LS):
            ax.plot(centers, frac(d[k], tol), color=c, ls=ls, marker="o", ms=3, label=lab)
        ax.set_xscale("log"); ax.set_xticks(ticks, [str(t) for t in ticks]); ax.minorticks_off()
        ax.set_ylim(0, 1.02); ax.yaxis.set_major_formatter(pct)
        ax.set_title(f"within {tol:.0%}"); ax.set_xlabel("injected SNR"); ax.set_ylabel("fraction of events")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8)
    fig.suptitle(f"Chirp mass on the same test windows: {title}", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out / "chirp_mass_compare.png", dpi=110); plt.close(fig)

    # 3. predicted against true, median and 1 sigma band, equal axes
    bins = np.linspace(0.85, 2.75, 39); mid = 0.5 * (bins[:-1] + bins[1:])
    bands = [("all SNR", np.ones_like(d["snr"], bool))]
    bands.append(("SNR 8 to 12", (d["snr"] >= 8) & (d["snr"] < 12)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6))
    for ax, (lab_b, sel) in zip(axes, bands):
        for (k, lab), c, ls in zip(NAMES[:3], C, LS):
            q = np.array([np.percentile(d[k][sel & (d["true"] >= a) & (d["true"] < b)], [16, 50, 84])
                          if (sel & (d["true"] >= a) & (d["true"] < b)).sum() > 4 else [np.nan] * 3
                          for a, b in zip(bins[:-1], bins[1:])])
            ax.fill_between(mid, q[:, 0], q[:, 2], color=c, alpha=0.15, lw=0)
            ax.plot(mid, q[:, 1], color=c, ls=ls, label=lab)
        ax.plot((0.85, 2.75), (0.85, 2.75), color=MUTED, ls=":", lw=1.2)
        ax.set_xlim(0.85, 2.75); ax.set_ylim(0.85, 2.75); ax.set_aspect("equal")
        ax.set_xlabel("true chirp mass (M☉)"); ax.set_ylabel("predicted chirp mass (M☉)")
        ax.set_title(f"{lab_b} ({sel.sum()} events): median and 1σ band")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "chirp_mass_pred_vs_true.png", dpi=110); plt.close(fig)

    # 4. relative error distribution of the chirp kernels, by SNR band
    fig, ax = plt.subplots(figsize=(8, 4.4))
    rel = (d["trained"] - d["true"]) / d["true"]
    for (a, b), c, ls in zip([(e, f) for e, f in ((4, 8), (8, 12), (12, 25), (25, 50)) if e >= lo_snr], C, LS):
        sel = (d["snr"] >= a) & (d["snr"] < b)
        if sel.sum():
            ax.hist(100 * rel[sel], bins=np.linspace(-10, 10, 121), density=True, histtype="step",
                    color=c, ls=ls, lw=1.6, label=f"SNR {a} to {b} ({sel.sum()})")
    ax.set_xlabel("relative chirp-mass error (%)"); ax.set_ylabel("density")
    ax.set_title(f"Chirp kernels, error distribution: {title}")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / "chirp_mass_residuals.png", dpi=110); plt.close(fig)
    print("wrote", out)
