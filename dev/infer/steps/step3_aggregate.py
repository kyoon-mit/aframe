"""Aggregate per-branch infer outputs into one background + foreground ledger.

Combines results/branch_*/{background,foreground}_<tag>.hdf5 into a separate
per-tag directory (default <results>/../aggregated/<tag>/) so the aggregated
files never sit among the 250 branch dirs. Tb (total background livetime,
which sets the FAR floor 1/Tb) is summed explicitly across branches, since
ledger aggregation tracks row counts, not the Tb metadata.
"""

import argparse
import glob
import os

from ledger.events import EventSet, RecoveredInjectionSet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results", required=True, help="dir of branch_* subdirs"
    )
    parser.add_argument(
        "--tag",
        default="boxcar",
        help="integration tag from step 2 (e.g. boxcar, gaussian4)",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="where to write aggregated files "
        "(default: <results>/../aggregated/<tag>)",
    )
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.normpath(args.results)),
        "aggregated",
        args.tag,
    )
    os.makedirs(outdir, exist_ok=True)

    bg_files = sorted(
        glob.glob(f"{args.results}/branch_*/background_{args.tag}.hdf5")
    )
    fg_files = sorted(
        glob.glob(f"{args.results}/branch_*/foreground_{args.tag}.hdf5")
    )
    if not bg_files:
        raise SystemExit(f"no branch background files under {args.results}")

    total_tb = sum(EventSet.read(f).Tb for f in bg_files)
    # drop empty foreground branches (no injections at that shift)
    fg_nonempty = [
        f for f in fg_files if len(RecoveredInjectionSet.read(f)) > 0
    ]

    bg_out = os.path.join(outdir, "background.hdf5")
    fg_out = os.path.join(outdir, "foreground.hdf5")
    EventSet.aggregate(bg_files, bg_out, clean=False)
    RecoveredInjectionSet.aggregate(fg_nonempty, fg_out, clean=False)

    # restore the summed Tb on the aggregated background
    background = EventSet.read(bg_out)
    background.Tb = total_tb
    background.write(bg_out)

    foreground = RecoveredInjectionSet.read(fg_out)
    far_floor = EventSet.read(bg_out).min_far
    print(f"branches: {len(bg_files)} bg, {len(fg_nonempty)} with foreground")
    print(
        f"background events: {len(background)}  Tb: {total_tb:.0f}s "
        f"({total_tb / (60 * 60 * 24 * 365.25):.3f} yr)"
    )
    print(f"recovered injections: {len(foreground)}")
    print(f"FAR floor (min_far): {far_floor:.3f} /yr")
    print(f"wrote {bg_out}\n      {fg_out}")


if __name__ == "__main__":
    main()
