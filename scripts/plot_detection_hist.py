"""Overlaid detection-score histograms: foreground (signal) vs background (noise).

Reads an SV run's background.hdf5 / foreground.hdf5 and plots the detection
statistic (= -sigma, higher = more confident) as density-normalized overlaid
histograms: background blue, foreground red (both alpha-filled). A loud-injection
subset (SNR>=15) is overlaid as a dark-red step so you can see whether *detectable*
signals separate from noise even when the faint majority does not.

Usage:
    python plot_detection_hist.py --background bg.hdf5 --foreground fg.hdf5 \
        --output /path/to/detection_score_hist.png [--title "..."] [--snr-cut 15]
"""
import argparse
import time

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read(path, key):
    # retry: the file may be mid-rewrite by a running job
    for _ in range(10):
        try:
            with h5py.File(path, "r") as f:
                return f[f"parameters/{key}"][:]
        except Exception:
            time.sleep(2)
    raise RuntimeError(f"could not read {key} from {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--background", required=True)
    ap.add_argument("--foreground", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--snr-cut", type=float, default=15.0)
    args = ap.parse_args()

    bg = _read(args.background, "detection_statistic")
    fg = _read(args.foreground, "detection_statistic")
    snr = _read(args.foreground, "snr")

    # drop the "missed injection" sentinel (-1e30) assigned by windowed recovery
    # when no trigger falls in the window — these have no meaningful score.
    SENT = -1e29
    bg = bg[bg > SENT]
    keep = fg > SENT
    n_missed = int((~keep).sum())
    fg = fg[keep]
    snr = snr[keep]
    loud = fg[snr >= args.snr_cut]

    lo = float(min(bg.min(), fg.min()))
    hi = float(max(bg.max(), fg.max()))
    bins = np.linspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(bg, bins=bins, density=True, color="#1f77b4", alpha=0.5,
            label=f"background / noise  (n={len(bg):,})")
    ax.hist(fg, bins=bins, density=True, color="#d62728", alpha=0.5,
            label=f"foreground / injections  (n={len(fg):,}; {n_missed:,} missed)")
    if len(loud):
        ax.hist(loud, bins=bins, density=True, histtype="step", color="#7a0000",
                lw=2, label=f"injections SNR≥{args.snr_cut:.0f}  (n={len(loud):,})")
    ax.axvline(bg.max(), color="k", ls="--", lw=1, alpha=0.7,
               label=f"loudest background ({bg.max():.3f})")

    ax.set_yscale("log")
    ax.set_xlabel("detection statistic  (−σ, higher = more confident)", fontsize=11)
    ax.set_ylabel("density", fontsize=11)
    ax.set_title(args.title or "Detection score: signal vs background", fontsize=11)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=140, bbox_inches="tight")
    print(f"saved {args.output}")
    print(f"  bg: med={np.median(bg):.3f} max={bg.max():.3f}")
    print(f"  fg(all): med={np.median(fg):.3f}   fg(SNR>={args.snr_cut:.0f}): "
          f"med={np.median(loud):.3f} max={loud.max():.3f}" if len(loud) else "")


if __name__ == "__main__":
    main()
