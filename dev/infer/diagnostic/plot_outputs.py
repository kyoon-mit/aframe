"""Headless model-output diagnostics.

CSV mode (--outdir, the self-inject dump from save_outputs.py):
  Fig 1: predicted sigma vs kernel-right-edge-relative-to-merger (sig vs bkg).
  Fig 2: chirp-mass relative error (pred-true)/true vs the same (sig only).
  Saves PNGs next to the CSVs. Also drops a loud-only variant.

    uv run python plot_outputs.py --outdir <dir with raw_sig.csv/raw_bkg.csv>

HDF5 mode (--branch, a step-1 SV cache -- raw score straight off the server,
no re-inference): plots the raw model score vs time for the background and
injected streams, with injection coalescence times marked.

    uv run python plot_outputs.py --branch <.../results/branch_0> [--channel 1]
"""

import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SNR_THRESHOLD = 4.0


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


def injection_times(attributes):
    """Coalescence times (relative to t0) of a branch's injections."""
    from ledger.injections import (
        InterferometerResponseSet,
        waveform_class_factory,
    )

    ifos = [
        name.decode() if isinstance(name, bytes) else str(name)
        for name in attributes["ifos"]
    ]
    response_set = waveform_class_factory(
        ifos, InterferometerResponseSet, "ResponseSet"
    )
    t0 = float(attributes["t0"])
    injection_set = response_set.read(
        str(attributes["injection_set_fname"]),
        start=t0,
        end=t0 + float(attributes["duration"]),
        shifts=list(attributes["shifts"]),
    )
    if len(injection_set) == 0:
        return np.array([])
    return np.asarray(injection_set.injection_time) - t0


def _pick(series, channel):
    """Select one channel of a possibly-2-D score stream."""
    if series is None or series.ndim == 1:
        return series
    return series[:, 0 if channel is None else channel]


def collect_injection_windows(branch_dir, channel, span):
    """Slice a +/- span/2 score window around every injection in a branch.

    Aligns each injection at merger = 0 and returns (offsets, fg, bg) where
    offsets is the shared relative-time axis (s) and fg/bg are (n_inj, n_off)
    score windows from the injected and background streams. Windows that fall
    off either end of the stream are dropped.
    """
    with h5py.File(os.path.join(branch_dir, "timeseries.hdf5"), "r") as f:
        attributes = dict(f.attrs)
        background = _pick(np.asarray(f["background_ts"]), channel)
        foreground = (
            _pick(np.asarray(f["foreground_ts"]), channel)
            if "foreground_ts" in f
            else None
        )
    if foreground is None:
        return None
    sample_rate = float(attributes["inference_sampling_rate"])
    half = int(round((span / 2) * sample_rate))
    offsets = np.arange(-half, half + 1) / sample_rate
    coalescences = injection_times(attributes)
    centers = np.round(coalescences * sample_rate).astype(int)

    fg_windows, bg_windows = [], []
    for center in centers:
        left, right = center - half, center + half + 1
        if left < 0 or right > len(background):
            continue
        fg_windows.append(foreground[left:right])
        bg_windows.append(background[left:right])
    if not fg_windows:
        return None
    return offsets, np.stack(fg_windows), np.stack(bg_windows)


def score_vs_time_fig(branch_dirs, channel, span, out):
    """Median score +/- 68% band vs time-to-merger, injected vs background.

    Aggregates the +/- span/2 windows around every injection across all the
    given branch dirs (align at merger = 0), then draws the median line and the
    16-84% band for the injected and background streams -- the same summary
    style as the sigma-vs-window figure, but for the raw SV-cache score.
    """
    offsets = None
    fg_all, bg_all = [], []
    for branch_dir in branch_dirs:
        result = collect_injection_windows(branch_dir, channel, span)
        if result is None:
            continue
        offsets, fg_windows, bg_windows = result
        fg_all.append(fg_windows)
        bg_all.append(bg_windows)
    if offsets is None:
        raise SystemExit("no injection windows collected")
    foreground = np.concatenate(fg_all)
    background = np.concatenate(bg_all)
    n_inj = len(foreground)

    fig, ax = plt.subplots(figsize=(9, 5))
    for data, name, color in [
        (foreground, "injected", "C0"),
        (background, "background", "C1"),
    ]:
        median = np.median(data, axis=0)
        lo = np.quantile(data, 0.16, axis=0)
        hi = np.quantile(data, 0.84, axis=0)
        ax.plot(offsets, median, color=color, lw=2, label=f"{name} median")
        ax.fill_between(
            offsets, lo, hi, color=color, alpha=0.25, label=f"{name} 68%"
        )
    ax.axvline(0, color="k", lw=0.8, ls="--", label="merger")
    ax.set_xlabel("time relative to merger [s]")
    ax.set_ylabel("raw model score")
    ax.set_title(f"score vs time-to-merger (n={n_inj} injections)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}  (n={n_inj})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--outdir", help="CSV mode: dir with raw_sig/bkg.csv")
    ap.add_argument("--branch", help="HDF5 mode: one SV branch dir")
    ap.add_argument(
        "--results",
        help="HDF5 mode: results dir -- aggregate all branch_* for a "
        "smoother band",
    )
    ap.add_argument(
        "--channel",
        type=int,
        default=None,
        help="HDF5 multi-channel cache: column to plot (0=mass, 1=sigma)",
    )
    ap.add_argument(
        "--span",
        type=float,
        default=32.0,
        help="HDF5 mode: total window span around merger, in seconds",
    )
    ap.add_argument("--out", help="HDF5 mode: output png path")
    args = ap.parse_args()

    if args.branch or args.results:
        if args.results:
            import glob

            branch_dirs = sorted(
                os.path.dirname(path)
                for path in glob.glob(
                    f"{args.results}/branch_*/timeseries.hdf5"
                )
            )
            default_out = os.path.join(args.results, "score_vs_merger.png")
        else:
            branch_dirs = [args.branch]
            default_out = os.path.join(args.branch, "score_vs_merger.png")
        score_vs_time_fig(
            branch_dirs, args.channel, args.span, args.out or default_out
        )
        return

    if not args.outdir:
        ap.error("give --results/--branch (HDF5) or --outdir (CSV)")
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

    loud_sig = sig[sig.snr >= SNR_THRESHOLD]
    loud_bkg = bkg[bkg.snr >= SNR_THRESHOLD]
    n_loud = loud_sig.strain_id.nunique()
    if n_loud > 5:
        sigma_fig(
            loud_sig,
            loud_bkg,
            f"sigma vs time-to-merger (SNR>={SNR_THRESHOLD}, n={n_loud})",
            f"{args.outdir}/sigma_vs_window_snr{SNR_THRESHOLD}.png",
        )
        relerr_fig(
            loud_sig,
            f"chirp recovery vs time-to-merger "
            f"(SNR>={SNR_THRESHOLD}, n={n_loud})",
            f"{args.outdir}/chirp_relerr_vs_window_snr{SNR_THRESHOLD}.png",
        )


if __name__ == "__main__":
    main()
