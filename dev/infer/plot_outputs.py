"""Headless plots from raw_sig.csv / raw_bkg.csv (median line + 68% band).

Fig 1: predicted sigma vs kernel-right-edge-relative-to-merger (sig vs bkg).
Fig 2: chirp-mass relative error (pred-true)/true vs the same (sig only).
Saves PNGs next to the CSVs. Also drops a loud-only (SNR>=15) variant.

    uv run python plot_outputs.py --outdir <dir with raw_sig.csv/raw_bkg.csv>
"""

import argparse
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def band(df, col):
    g = df.groupby("kernel_right_s")[col]
    return g.median(), g.quantile(0.16), g.quantile(0.84)


def sigma_fig(sig, bkg, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for df, name, c in [
        (sig, "injected (sig)", "C0"),
        (bkg, "background (bkg)", "C1"),
    ]:
        med, lo, hi = band(df, "pred_sigma")
        ax.plot(med.index, med.values, color=c, lw=2, label=f"{name} median")
        ax.fill_between(
            med.index,
            lo.values,
            hi.values,
            color=c,
            alpha=0.25,
            label=f"{name} 68%",
        )
    ax.set_xlabel("kernel right edge relative to merger [s]")
    ax.set_ylabel(r"predicted $\sigma_{\mathcal{M}}$")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def relerr_fig(sig, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    med, lo, hi = band(sig, "rel_err")
    ax.plot(med.index, med.values, color="C2", lw=2, label="median")
    ax.fill_between(
        med.index, lo.values, hi.values, color="C2", alpha=0.25, label="68%"
    )
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("kernel right edge relative to merger [s]")
    ax.set_ylabel("chirp-mass relative error (pred - true) / true")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    sig = pd.read_csv(f"{args.outdir}/raw_sig.csv")
    bkg = pd.read_csv(f"{args.outdir}/raw_bkg.csv")
    sig["rel_err"] = (
        sig.pred_chirp_mass - sig.true_chirp_mass_src
    ) / sig.true_chirp_mass_src
    n_all = sig.strain_id.nunique()
    print(f"loaded sig strains={n_all} rows={len(sig)}")

    sigma_fig(
        sig,
        bkg,
        f"sigma vs time-to-merger (all, n={n_all})",
        f"{args.outdir}/sigma_vs_window.png",
    )
    relerr_fig(
        sig,
        f"chirp recovery vs time-to-merger (all, n={n_all})",
        f"{args.outdir}/chirp_relerr_vs_window.png",
    )

    loud_sig = sig[sig.snr >= 15]
    loud_bkg = bkg[bkg.snr >= 15]
    n_loud = loud_sig.strain_id.nunique()
    if n_loud > 5:
        sigma_fig(
            loud_sig,
            loud_bkg,
            f"sigma vs time-to-merger (SNR>=15, n={n_loud})",
            f"{args.outdir}/sigma_vs_window_snr15.png",
        )
        relerr_fig(
            loud_sig,
            f"chirp recovery vs time-to-merger (SNR>=15, n={n_loud})",
            f"{args.outdir}/chirp_relerr_vs_window_snr15.png",
        )


if __name__ == "__main__":
    main()
