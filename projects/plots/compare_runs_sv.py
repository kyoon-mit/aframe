"""Compare each run's best integration method against a reference SV curve.

For every run under ``--runs-dir`` that has per-method sensitive-volume output
(``results_aframe/plots/sv/<method>/sensitive_volume.h5``, produced by
``scoring_sv.py``), this picks the *top* integration method -- the one with the
highest sensitive volume at the left (most stringent / smallest FAR) edge,
normalised per mass combo so the heavier-mass panels don't dominate -- and
plots its SV-vs-FAR curve.  All runs are overlaid together with the reference
run, one panel per mass combo.  Error bands are omitted.

Run inside the plots project environment, e.g.::

    uv run --no-sync --directory projects/plots python compare_runs_sv.py
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

MASS_COMBOS = ["1.4-1.4", "1.5-1.5", "2.0-2.0", "2.3-2.3"]

DEFAULT_RUNS_DIR = "/home/barmstrong/aframe_official/runs/aframe_results/runs"
DEFAULT_REFERENCE = (
    "/home/barmstrong/aframe_official/runs/aframe_results/reference/"
    "aframe-decimator-sv.h5"
)
DEFAULT_OUTPUT = (
    "/home/barmstrong/aframe_official/runs/aframe_results/"
    "top_methods_vs_reference.png"
)


def read_sv(path, combo_keys):
    """Load ``fars`` + per-combo ``sv`` from a sensitive_volume.h5 file."""
    with h5py.File(path, "r") as f:
        fars = f["fars"][:]
        sv = {k: f[k]["sv"][:] for k in combo_keys}
    return fars, sv


def top_method(sv_dir, combo_keys):
    """Return (method_name, (fars, sv), n_methods) for the best method in a
    run's ``plots/sv`` directory, or ``None`` if no method data is present."""
    methods = {}
    for md in sorted(p for p in sv_dir.iterdir() if p.is_dir()):
        h5 = md / "sensitive_volume.h5"
        if h5.exists():
            methods[md.name] = read_sv(h5, combo_keys)
    if not methods:
        return None

    # left-edge SV, normalised per combo so all combos count equally
    left = {
        n: np.array([sv[k][0] for k in combo_keys])
        for n, (_, sv) in methods.items()
    }
    best = np.max(np.stack(list(left.values())), axis=0)
    best[best == 0] = 1.0
    ranking = sorted(
        methods, key=lambda n: np.mean(left[n] / best), reverse=True
    )
    top = ranking[0]
    return top, methods[top], len(methods)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="include runs that have not finished all integration methods "
        "(top method is then chosen among the methods available so far)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs_dir = Path(args.runs_dir)

    # gather the top method of every run that has any SV output
    runs = {}
    for run in sorted(runs_dir.iterdir()):
        sv_dir = run / "results_aframe" / "plots" / "sv"
        if not sv_dir.is_dir():
            continue
        result = top_method(sv_dir, MASS_COMBOS)
        if result is None:
            continue
        runs[run.name] = result

    if not runs:
        raise SystemExit(f"no per-method SV output found under {runs_dir}")

    # treat the most methods seen as "complete"; optionally drop partial runs
    n_complete = max(r[2] for r in runs.values())
    if not args.allow_partial:
        partial = [n for n, r in runs.items() if r[2] < n_complete]
        for n in partial:
            logging.info(
                "skipping partial run %s (%d/%d methods); use --allow-partial "
                "to include",
                n,
                runs[n][2],
                n_complete,
            )
            del runs[n]

    logging.info(
        "comparing %d runs against reference %s", len(runs), args.reference
    )
    ref_fars, ref_sv = read_sv(args.reference, MASS_COMBOS)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    axes = axes.flatten()

    names = sorted(runs)
    cmap = plt.get_cmap("tab20", max(len(names), 1))
    colors = {name: cmap(i) for i, name in enumerate(names)}

    for i, (ax, key) in enumerate(zip(axes, MASS_COMBOS, strict=False)):
        ax.set_xscale("log")
        ax.set_title(f"Log Normal $m_1$-$m_2$ = {key}")
        if i % 2 == 0:
            ax.set_ylabel("Sensitive Volume [Gpc$^3$]")
        if i >= 2:
            ax.set_xlabel("False Alarm Rate [yr$^{-1}$]")

        for name in names:
            top, (fars, sv), _ = runs[name]
            label = f"{name} [{top}]" if i == 0 else None
            ax.plot(
                fars,
                sv[key],
                linewidth=1.5,
                color=colors[name],
                alpha=0.9,
                label=label,
            )

        # reference on top
        ax.plot(
            ref_fars,
            ref_sv[key],
            linewidth=2.5,
            color="k",
            linestyle="--",
            zorder=10,
            label="reference (decimator)" if i == 0 else None,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    # put the reference first in the legend
    order = sorted(
        range(len(labels)), key=lambda j: not labels[j].startswith("reference")
    )
    fig.legend(
        [handles[j] for j in order],
        [labels[j] for j in order],
        loc="lower center",
        ncol=3,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info("wrote %s", out)

    # also report which method won for each run
    for name in names:
        top, _, n = runs[name]
        logging.info("  %-28s top=%s (%d methods)", name, top, n)


if __name__ == "__main__":
    main()
