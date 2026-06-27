"""Per-model SV comparison against the reference run.

For every aframe inference run under ``--runs-root`` this script picks the
*best* integration method (the one with the highest sensitive volume averaged
across the FAR curve, normalised per mass combo so the heavier-mass / larger-
volume combos don't dominate) and overlays its SV-vs-FAR curve against the
reference run (``--reference``).  One grid figure per run is written into the
run's own directory.

The per-method SV curves are read from
``<run>/results_aframe/plots/sv/<method>/sensitive_volume.h5`` (produced by
``scoring_sv.py``); runs without that folder are skipped.

Run inside the plots project environment, e.g.::

    uv run --no-sync --directory projects/plots python compare_sv_reference.py
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from plots.legacy.matplotlib_tools import make_grid, plot_err_bands

# default location of the results tree relative to the repo root
DEFAULT_RUNS_ROOT = Path("runs/aframe_results/runs")
DEFAULT_REFERENCE = Path(
    "runs/aframe_results/reference/aframe-decimator-sv.h5"
)
SV_SUBDIR = Path("results_aframe/plots/sv")


def _combo_keys(sv_path):
    """Mass-combo group names in an SV file (everything that isn't a curve)."""
    with h5py.File(sv_path, "r") as f:
        return [
            k
            for k in f.keys()
            if isinstance(f[k], h5py.Group) and "sv" in f[k]
        ]


def read_sv(path, combo_keys):
    """Load ``fars`` + per-combo ``sv``/``err`` from an SV h5 file."""
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        sv = {k: f[k]["sv"][:] for k in combo_keys}
        err = {k: f[k]["err"][:] for k in combo_keys}
    return fars, sv, err


def pick_best_method(sv_dir, combo_keys):
    """Return ``(name, fars, sv, err)`` for the best method in ``sv_dir``.

    "Best" = highest mean sensitive volume across the whole FAR curve,
    normalised per combo against the best method in that combo, then averaged
    over combos.  Mean-over-FAR is used (rather than the single most-stringent
    FAR point) because the smallest-FAR SV is frequently 0 for every method and
    so cannot discriminate between them.
    """
    methods = {}
    for d in sorted(p for p in sv_dir.iterdir() if p.is_dir()):
        f = d / "sensitive_volume.h5"
        if f.exists():
            methods[d.name] = read_sv(f, combo_keys)
    if not methods:
        return None

    mean_sv = {
        name: np.array([np.mean(sv[k]) for k in combo_keys])
        for name, (_, sv, _) in methods.items()
    }
    best_per_combo = np.max(np.stack(list(mean_sv.values())), axis=0)
    best_per_combo[best_per_combo == 0] = 1.0
    ranking = sorted(
        methods,
        key=lambda n: np.mean(mean_sv[n] / best_per_combo),
        reverse=True,
    )
    best = ranking[0]
    return (best, *methods[best])


def plot_run(run_dir, reference, out_name, force):
    """Build the best-method-vs-reference grid for one run."""
    sv_dir = run_dir / SV_SUBDIR
    if not sv_dir.is_dir():
        logging.info("skip %s -- no SV folder (%s)", run_dir.name, sv_dir)
        return
    out_path = run_dir / out_name
    if out_path.exists() and not force:
        logging.info(
            "skip %s -- %s exists (use --force)", run_dir.name, out_path
        )
        return

    combo_keys = _combo_keys(next(sv_dir.glob("*/sensitive_volume.h5")))
    best = pick_best_method(sv_dir, combo_keys)
    if best is None:
        logging.warning("skip %s -- no per-method SV data", run_dir.name)
        return
    name, fars, sv, err = best
    logging.info("%s: best integration method = %s", run_dir.name, name)

    ref_fars, ref_sv, ref_err = read_sv(reference, combo_keys)

    mass_combos = [tuple(float(x) for x in k.split("-")) for k in combo_keys]
    fig, axes = make_grid(mass_combos)
    for i, ax in enumerate(axes):
        k = combo_keys[i]
        ax.plot(
            ref_fars,
            ref_sv[k],
            color="0.4",
            linestyle="--",
            linewidth=1.5,
            label="reference" if i == 0 else None,
            zorder=3,
        )
        plot_err_bands(
            ax, ref_fars, ref_sv[k], ref_err[k], color="0.4", alpha=0.15
        )
        ax.plot(
            fars,
            sv[k],
            color="#d62728",
            linewidth=2.0,
            label=f"{name} (best)" if i == 0 else None,
            zorder=5,
        )
        plot_err_bands(ax, fars, sv[k], err[k], color="#d62728", alpha=0.2)

    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle(f"{run_dir.name}: best method vs reference", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    logging.info("%s -> %s", run_dir.name, out_path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument(
        "--run",
        nargs="*",
        default=None,
        help="only process these run names (default: all under runs-root)",
    )
    ap.add_argument(
        "--out-name",
        default="best_vs_reference.png",
        help="output filename, written inside each run directory",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    if not args.reference.exists():
        raise SystemExit(f"reference not found: {args.reference}")

    runs = sorted(p for p in args.runs_root.iterdir() if p.is_dir())
    if args.run:
        runs = [r for r in runs if r.name in args.run]
        if not runs:
            raise SystemExit(f"no runs matched {args.run}")

    for run_dir in runs:
        plot_run(run_dir, args.reference, args.out_name, args.force)


if __name__ == "__main__":
    main()
