"""Aggregate per-branch infer outputs into one background + foreground ledger.

Combines results/branch_*/{background,foreground}.hdf5 into
results/{background,foreground}.hdf5. Tb (total background livetime, which sets
the FAR floor 1/Tb) is summed explicitly across branches, since ledger
aggregation tracks row counts, not the Tb metadata.
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
    args = parser.parse_args()

    bg_files = sorted(glob.glob(f"{args.results}/branch_*/background.hdf5"))
    fg_files = sorted(glob.glob(f"{args.results}/branch_*/foreground.hdf5"))
    if not bg_files:
        raise SystemExit(f"no branch background files under {args.results}")

    total_tb = sum(EventSet.read(f).Tb for f in bg_files)
    # drop empty foreground branches (no injections at that shift)
    fg_nonempty = [
        f for f in fg_files if len(RecoveredInjectionSet.read(f)) > 0
    ]

    bg_out = os.path.join(args.results, "background.hdf5")
    fg_out = os.path.join(args.results, "foreground.hdf5")
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
