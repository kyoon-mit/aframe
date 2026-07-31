"""CPU re-plot of pred-vs-true from an existing param_est_results.csv.

No model / no GPU: reads the CSV(s) already dumped by PlotParamEstCallback and
redraws the pred-vs-true panel with the band = median prediction +/- the
model's predicted sigma (1sigma, 2sigma) per true bin -- for signal and for
background (noise-only) predictions on the same events.

    python replot_pred_vs_true.py --dir <outdir> [--dist powerlaw]
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CFG = "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/plot_configs.json"  # noqa: E501


def pred_vs_true_ax(ax, t, p, s, lims, edges, title):
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(len(centers), np.nan)
    sig = np.full(len(centers), np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        m = (t >= lo) & (t < hi)
        if m.sum() < 2:
            continue
        med[i] = np.median(p[m])
        if s is not None:
            sig[i] = np.median(s[m])
    ok = ~np.isnan(med)
    if s is not None:
        bok = ok & ~np.isnan(sig)
        ax.fill_between(
            centers[bok],
            (med - sig)[bok],
            (med + sig)[bok],
            color="steelblue",
            alpha=0.35,
            label=r"$1\sigma$",
        )
    ax.plot(centers[ok], med[ok], color="steelblue", lw=1.6, label="Median")
    ax.plot(lims, lims, color="gray", ls=":", lw=1.2)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def one_panel(t, p, s, lims, edges, title, xlab, ylab, out):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    pred_vs_true_ax(ax, t, p, s, lims, edges, title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def frac_within_vs_snr(t, p, snr, tag, out):
    cuts = [0.01, 0.02, 0.05, 0.10]
    cut_colors = ["#3b528b", "#21918c", "#5ec962", "#f4a259"]  # 10% = orange
    a, b = int(np.floor(snr.min())), int(np.ceil(snr.max()))
    bins = np.arange(a, b + 2, 2)
    centers = 0.5 * (bins[:-1] + bins[1:])
    denom = np.where(np.abs(t) > 1e-8, np.abs(t), 1.0)
    rel = np.abs((p - t) / denom)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, cut in enumerate(cuts):
        frac = np.full(len(centers), np.nan)
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:], strict=False)):
            m = (snr >= lo) & (snr < hi)
            if m.sum() > 0:
                frac[i] = 100.0 * (rel[m] < cut).mean()
        ok = ~np.isnan(frac)
        ax.plot(
            centers[ok],
            frac[ok],
            marker="o",
            ms=3,
            color=cut_colors[j],
            label=f"{int(cut * 100)}%",
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel("% within cutoff")
    ax.set_ylim(0, 100)
    ax.set_title(f"SNR {a}-{b}{tag}")
    ax.legend(frameon=False, fontsize=9, title="|rel. err|", loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True, help="dir with the CSVs")
    ap.add_argument("--dist", default="", help="powerlaw/uniform title tag")
    args = ap.parse_args()

    sig = pd.read_csv(os.path.join(args.dir, "param_est_results.csv"))
    t = sig["chirp_mass_true"].to_numpy()
    p = sig["chirp_mass_pred"].to_numpy()
    s = sig["sigma_chirp_mass_pred"].to_numpy()
    snr = sig["snr"].to_numpy() if "snr" in sig else None

    lims = [float(t.min()), float(t.max())]
    edges = np.linspace(lims[0], lims[1], 41)
    blo = int(np.floor(snr.min())) if snr is not None else None
    bhi = int(np.ceil(snr.max())) if snr is not None else None
    blab = f"SNR {blo}-{bhi}" if snr is not None else "all"
    suffix = f"snr{blo}-{bhi}" if snr is not None else "all"
    tag = f" ({args.dist})" if args.dist else ""
    # units-free symbol straight from plot_configs.json (no regex stripping)
    cfg = json.load(open(_CFG))["parameters"]["chirp_mass"]
    sym = cfg["label"]  # $\mathcal{M}_c$
    hat = sym.replace(r"\mathcal{M}", r"\hat{\mathcal{M}}")  # pred symbol

    one_panel(
        t,
        p,
        s,
        lims,
        edges,
        f"{blab}{tag}",
        sym,
        hat,
        os.path.join(args.dir, f"chirp_mass_pred_vs_true_{suffix}.png"),
    )

    # z-score histogram: fixed x -5..5, bin 0.25, ticks every 1, no stats box
    z = (
        sig["chirp_mass_zscore"].to_numpy()
        if "chirp_mass_zscore" in sig
        else (p - t) / (s + 1e-12)
    )
    z = z[np.isfinite(z)]
    zbins = np.arange(-5.0, 5.0 + 0.25, 0.25)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(z, bins=zbins, color="steelblue", alpha=0.6, label="z-score")
    mu, sd = float(np.mean(z)), float(np.std(z))
    xs = np.linspace(-5.0, 5.0, 400)
    bw = 0.25
    ax.plot(
        xs,
        z.size
        * bw
        / (sd * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((xs - mu) / sd) ** 2),
        "r-",
        lw=1.8,
        label=f"fit N({mu:.2f}, {sd:.2f})",
    )
    ax.plot(
        xs,
        z.size * bw / np.sqrt(2 * np.pi) * np.exp(-0.5 * xs**2),
        "k--",
        lw=1.0,
        alpha=0.7,
        label="N(0, 1)",
    )
    ax.set_xlim(-5, 5)
    ax.set_xticks(range(-5, 6))
    ax.set_xlabel(f"{sym} z-score")
    ax.set_ylabel("Counts")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.dir, "chirp_mass_zscore.png"), dpi=140)
    plt.close(fig)
    print("wrote", os.path.join(args.dir, "chirp_mass_zscore.png"))

    if snr is not None:
        frac_within_vs_snr(
            t,
            p,
            snr,
            tag,
            os.path.join(args.dir, "chirp_mass_frac_within_vs_snr.png"),
        )

    bkg_path = os.path.join(args.dir, "param_est_results_bkg.csv")
    if os.path.exists(bkg_path):
        bkg = pd.read_csv(bkg_path)
        pb = bkg["chirp_mass_pred"].to_numpy()
        sb = bkg["sigma_chirp_mass_pred"].to_numpy()
        n = min(len(t), len(pb))
        one_panel(
            t[:n],
            pb[:n],
            sb[:n],
            lims,
            edges,
            f"{blab}{tag} (bkg)",
            sym,
            hat,
            os.path.join(
                args.dir, f"chirp_mass_pred_vs_true_{suffix}_bkg.png"
            ),
        )


if __name__ == "__main__":
    main()
