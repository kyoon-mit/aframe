"""Render diagnostic PNGs from the dumped CSVs (no GPU, static matplotlib).

Produces static versions of what the interactive notebooks show, one folder per
model under ``diagnostics/``:

    3s_premerger/   score_vs_alignment.png + test_step_*.png (x4)
    merger_4s/      score_vs_alignment.png
    merger_1s/      score_vs_alignment.png

Run:  python diagnostics/make_plots.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss")
DIAG = ROOT / "runs/regression_sv/diag"
OUT = ROOT / "diagnostics"

# model -> (output folder, run dir, trained alignment e, show the merger/OOD line, title)
MODELS = [
    dict(folder="3s_premerger", run="premerger_59-60s_ft", trained_e=-3.0,
         show_merger_line=True, title="Pre-merger 1s (59-60s, id11) — trained e = -3"),
    dict(folder="merger_4s", run="merger_60-64s_ft", trained_e=0.0,
         show_merger_line=False, title="Merger 4s (60-64s) — trained e = 0"),
    dict(folder="merger_1s", run="merger_63-64s_ft", trained_e=0.0,
         show_merger_line=False, title="Merger 1s (63-64s) — trained e = 0"),
]


def plot_score_vs_alignment(csv, outdir, trained_e, show_merger_line, title):
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    for kind, color in [("background", "tab:blue"), ("signal", "tab:red")]:
        sub = df[df.kind == kind]
        for _, g in sub.groupby("event_id"):
            g = g.sort_values("e")
            ax.plot(g.e, g.neg_sigma, color=color, alpha=0.20, lw=1)
        prof = sub.groupby(sub.e.round(3)).neg_sigma.mean()
        ax.plot(prof.index, prof.values, color=color, lw=2.6, label=f"{kind} (mean)")
    ax.axvline(trained_e, color="green", ls="--", lw=1.5, label=f"trained  e={trained_e:g}")
    if show_merger_line:
        ax.axvline(0.0, color="gray", ls=":", lw=1.5, label="merger  e=0 (OOD)")
    ax.set_xlabel("e = kernel right-edge − coalescence [s]   (<0 pre-merger)")
    ax.set_ylabel("detection statistic  −σ   (higher = more confident)")
    ax.set_title(title)
    ax.legend(fontsize=8.5, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "score_vs_alignment.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved", p)


def plot_test_step(csv, outdir, title):
    df = pd.read_csv(csv)

    # 1. inferred vs true, colored by SNR
    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(df.true_chirp_mass, df.fg_chirp_mean, c=df.snr, cmap="viridis",
                    s=14, alpha=0.6, vmax=np.percentile(df.snr, 95))
    lim = [df.true_chirp_mass.min(), df.true_chirp_mass.max()]
    ax.plot(lim, lim, "k--", lw=1, label="perfect")
    ax.set_xlabel(r"true chirp mass [$M_\odot$]")
    ax.set_ylabel(r"inferred chirp mass [$M_\odot$]")
    ax.set_title(f"{title}\ninferred vs true (signal)")
    ax.legend()
    plt.colorbar(sc, label="SNR")
    fig.tight_layout()
    fig.savefig(outdir / "test_step_inferred_vs_true.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. sigma distributions: signal vs noise
    lo = min(df.fg_chirp_sigma.min(), df.bg_chirp_sigma.min())
    hi = max(df.fg_chirp_sigma.quantile(0.99), df.bg_chirp_sigma.quantile(0.99))
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.fg_chirp_sigma, bins, density=True, alpha=0.5, color="tab:red", label="signal")
    ax.hist(df.bg_chirp_sigma, bins, density=True, alpha=0.5, color="tab:blue", label="noise")
    ax.set_xlabel(r"predicted $\sigma$(chirp mass) [$M_\odot$]")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\npredicted uncertainty: signal vs noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "test_step_sigma.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. z-score calibration
    z = df.fg_z.replace([np.inf, -np.inf], np.nan).dropna()
    z = z[np.abs(z) < 10]
    bins = np.linspace(-6, 6, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(z, bins, density=True, alpha=0.6, color="tab:purple",
            label=f"z (mean={z.mean():.2f}, std={z.std():.2f})")
    xs = np.linspace(-6, 6, 200)
    ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "k--", label="unit normal")
    ax.set_xlabel("z = (inferred − true) / σ")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\nuncertainty calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "test_step_zscore.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 4. chirp-mass histograms: true vs inferred (signal vs noise)
    lo = min(df.true_chirp_mass.min(), df.fg_chirp_mean.min(), df.bg_chirp_mean.min())
    hi = max(df.true_chirp_mass.max(), df.fg_chirp_mean.max(), df.bg_chirp_mean.max())
    bins = np.linspace(lo, hi, 60)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df.true_chirp_mass, bins, density=True, histtype="step", lw=2, color="k", label="true")
    ax.hist(df.fg_chirp_mean, bins, density=True, alpha=0.5, color="tab:red", label="inferred (signal)")
    ax.hist(df.bg_chirp_mean, bins, density=True, alpha=0.5, color="tab:blue", label="inferred (noise)")
    ax.set_xlabel(r"chirp mass [$M_\odot$]")
    ax.set_ylabel("density")
    ax.set_title(f"{title}\nchirp mass: true vs inferred")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "test_step_chirpmass_hist.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved 4 test_step PNGs ->", outdir)


def main():
    for m in MODELS:
        outdir = OUT / m["folder"]
        outdir.mkdir(parents=True, exist_ok=True)
        run = DIAG / m["run"]

        score_csv = run / "score_timeseries.csv"
        if score_csv.exists():
            plot_score_vs_alignment(score_csv, outdir, m["trained_e"],
                                    m["show_merger_line"], m["title"])
        else:
            print("MISSING", score_csv)

        test_csv = run / "test_step.csv"
        if test_csv.exists():
            plot_test_step(test_csv, outdir, m["title"])


if __name__ == "__main__":
    main()
