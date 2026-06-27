"""Per-run foreground/background detection-statistic distributions.

For every aframe inference run under ``--runs-root`` this plots the histogram
of background vs foreground detection statistics, the reverse-cumulative
foreground count, and the number of foregrounds recovered above the loudest
background event (the FPR=0 detection count).  One PNG per run is written into
the run's own directory.

By default the statistics are taken from the *best* integration method's
regenerated ``background.hdf5`` / ``foreground.hdf5`` (the method highlighted
by ``compare_sv_reference.py``), so this figure and the SV comparison describe
the same detection statistic.  Use ``--source run`` to instead read the run's
top-level ``results_aframe/{background,foreground}.hdf5``.

Run inside the plots project environment, e.g.::

    uv run --no-sync --directory projects/plots python fg_bg_distributions.py
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from compare_sv_reference import (
    DEFAULT_RUNS_ROOT,
    SV_SUBDIR,
    _combo_keys,
    pick_best_method,
)


def _detection_statistic(path):
    with h5py.File(path, "r") as f:
        return f["parameters"]["detection_statistic"][:]


def resolve_sources(run_dir, source):
    """Return ``(label, background_path, foreground_path)`` or ``None``."""
    results = run_dir / "results_aframe"
    if source == "run":
        bg, fg = results / "background.hdf5", results / "foreground.hdf5"
        return ("run", bg, fg) if bg.exists() and fg.exists() else None

    # source == "best": use the best integration method's ledgers
    sv_dir = run_dir / SV_SUBDIR
    if not sv_dir.is_dir():
        return None
    combo_keys = _combo_keys(next(sv_dir.glob("*/sensitive_volume.h5")))
    best = pick_best_method(sv_dir, combo_keys)
    if best is None:
        return None
    name = best[0]
    bg = sv_dir / name / "background.hdf5"
    fg = sv_dir / name / "foreground.hdf5"
    return (name, bg, fg) if bg.exists() and fg.exists() else None


def plot_run(run_dir, source, out_name, force):
    out_path = run_dir / out_name
    if out_path.exists() and not force:
        logging.info(
            "skip %s -- %s exists (use --force)", run_dir.name, out_path
        )
        return
    srcs = resolve_sources(run_dir, source)
    if srcs is None:
        logging.info("skip %s -- no %s ledgers found", run_dir.name, source)
        return
    label, bg_path, fg_path = srcs

    bkg_d = _detection_statistic(bg_path)
    fg_d = _detection_statistic(fg_path)
    # injections that were never recovered carry -inf; drop them so they don't
    # collapse the histogram range.
    fg_d = fg_d[np.isfinite(fg_d)]
    if bkg_d.size == 0 or fg_d.size == 0:
        logging.warning("skip %s -- empty statistics", run_dir.name)
        return

    # The science is in the loudest events (right tail), which a linear x-axis
    # crushes into a single overlapping bin.  Re-bin in distance-from-the-
    # loudest-event ``u = max - score`` on a log scale so the top scores get
    # the finest resolution, and draw the score axis reversed (loudest on the
    # right, like the linear plot) with ticks relabelled back to raw score.
    top = float(np.concatenate([bkg_d, fg_d]).max())
    u_bkg = top - bkg_d
    u_fg = top - fg_d
    u_all = np.concatenate([u_bkg, u_fg])
    positive = u_all[u_all > 0]
    if positive.size == 0:
        logging.warning("skip %s -- all scores identical", run_dir.name)
        return
    bins = np.logspace(np.log10(positive.min()), np.log10(u_all.max()), 60)

    # the loudest event(s) sit at u == 0; fold them into the first (highest-
    # score) bin so they still register on the log axis.
    def clip(u):
        return np.clip(u, bins[0], bins[-1])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(
        clip(u_bkg),
        histtype="step",
        bins=bins,
        color="royalblue",
        linestyle="--",
        label="background",
    )
    ax.hist(
        clip(u_fg),
        histtype="step",
        bins=bins,
        color="royalblue",
        label="foreground",
    )
    # reverse-cumulative foreground count N(score >= t) == #{u <= u_t}, which
    # is the forward cumulative in u (cumulative=1).
    ax.hist(
        clip(u_fg),
        histtype="step",
        bins=bins,
        color="darkorange",
        linestyle="-.",
        label="foreground (reverse cumsum)",
        cumulative=1,
    )

    max_bkg = float(bkg_d.max())
    num_detected = int((fg_d >= max_bkg).sum())
    u_thresh = clip(np.array([top - max_bkg]))[0]
    ax.plot(
        u_thresh,
        num_detected,
        "ro",
        label=f"Detected at FPR=0: {num_detected}",
    )
    ax.annotate(
        f"#{num_detected}",
        (u_thresh, num_detected),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        color="red",
        fontsize=8,
    )

    ax.set_xscale("log")
    ax.invert_xaxis()  # loudest (smallest u, highest score) on the right
    ticks = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), 6)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda u, _: f"{top - u:.3g}"))
    ax.minorticks_off()
    ax.set_xlabel("Score (log-spaced toward loudest event)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{run_dir.name} ({label}): foreground vs background detection scores"
    )
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logging.info(
        "%s -> %s (source=%s, FPR=0 detections=%d)",
        run_dir.name,
        out_path,
        label,
        num_detected,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument(
        "--source",
        choices=["best", "run"],
        default="best",
        help="which ledgers to use: best integration method, or the run's "
        "top-level results_aframe ledgers",
    )
    ap.add_argument("--run", nargs="*", default=None)
    ap.add_argument(
        "--out-name", default="detection_stat_fg_bg_distributions.png"
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs = sorted(p for p in args.runs_root.iterdir() if p.is_dir())
    if args.run:
        runs = [r for r in runs if r.name in args.run]
        if not runs:
            raise SystemExit(f"no runs matched {args.run}")

    for run_dir in runs:
        plot_run(run_dir, args.source, args.out_name, args.force)


if __name__ == "__main__":
    main()
