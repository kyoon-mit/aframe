"""Step 5: rank all integration variants by sensitive volume.

Reads every ``aggregated/<tag>/sensitive_volume.h5`` (written by step 4) and
prints a table of the 1.5-1.5 Msun sensitive volume at a reference FAR and at
the FAR floor, ranked best-first. Also writes ``aggregated/summary.txt``.

    uv run python step5_summary.py --aggregated <.../aggregated> [--far 100]
"""

import argparse
import glob
import os

import h5py
import numpy as np

DEFAULT_COMBO = "1.4-1.4"


def pick_combo(h5_file, requested):
    """Return the requested mass combo, or the first available if absent.

    Different runs bin masses differently (e.g. 1.5-1.5 vs 1.4/1.6/1.8/2.0),
    so fall back to whatever this file actually has rather than KeyError.
    """
    if requested in h5_file:
        return requested
    combos = sorted(
        key
        for key in h5_file
        if isinstance(h5_file[key], h5py.Group) and "sv" in h5_file[key]
    )
    if not combos:
        raise SystemExit("no mass-combo groups in sensitive_volume.h5")
    return combos[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aggregated", required=True)
    parser.add_argument("--far", type=float, default=100.0, help="ref /yr")
    parser.add_argument("--combo", default=DEFAULT_COMBO, help="mass combo")
    args = parser.parse_args()

    rows = []
    combo = None
    for h5_path in sorted(
        glob.glob(f"{args.aggregated}/*/sensitive_volume.h5")
    ):
        tag = os.path.basename(os.path.dirname(h5_path))
        with h5py.File(h5_path, "r") as h5_file:
            combo = pick_combo(h5_file, args.combo)
            fars_per_year = h5_file["fars"][:]  # already per year
            volumes = h5_file[combo]["sv"][:]
            errors = h5_file[combo]["err"][:]
        order = np.argsort(fars_per_year)
        fars_per_year = fars_per_year[order]
        volumes = volumes[order]
        errors = errors[order]
        at_ref = float(np.interp(args.far, fars_per_year, volumes))
        rows.append(
            {
                "tag": tag,
                "sv_ref": at_ref,
                "sv_floor": float(volumes[0]),
                "err_floor": float(errors[0]),
                "floor": float(fars_per_year[0]),
            }
        )

    rows.sort(key=lambda row: row["sv_ref"], reverse=True)
    lines = [
        f"{combo} Msun sensitive volume [Gpc^3], ranked by SV @ "
        f"{args.far:g}/yr",
        f"{'tag':<14} {'SV@ref':>10} {'SV@floor':>10} {'floor[/yr]':>10}",
    ]
    for row in rows:
        lines.append(
            f"{row['tag']:<14} {row['sv_ref']:>10.3e} "
            f"{row['sv_floor']:>10.3e} {row['floor']:>10.1f}"
        )
    report = "\n".join(lines)
    print(report)
    summary_path = os.path.join(args.aggregated, "summary.txt")
    with open(summary_path, "w") as summary_file:
        summary_file.write(report + "\n")
    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
