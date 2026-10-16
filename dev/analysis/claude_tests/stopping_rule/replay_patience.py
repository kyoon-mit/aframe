"""Replay plain early stopping on the local training histories.

For each run in dev/runs/claude_tests with at least --min-epochs validation
epochs of val/within_1pct_chirp_mass, stop at the first epoch whose best is
--patience epochs old, and report the shortfall: best of the full history
minus best before the stop, in percentage points. Signal-only runs are the
sig_only_* runs; the rest are noisy.

    python replay_patience.py [--min-epochs 60] [--patience 20 30 40 50 60 80 100 120 160]
"""

import argparse
import glob
import importlib.util
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../runs/claude_tests")
KEY = "val/within_1pct_chirp_mass"


def load_rows():
    spec = importlib.util.spec_from_file_location("warp_table", os.path.join(R, "warp_table.py"))
    src = open(spec.origin).read().split("mode = sys.argv[1]")[0]
    ns = {}
    exec(compile(src, spec.origin, "exec"), ns)
    return ns["rows"]


def shortfall(values, patience):
    best, best_at = -1.0, 0
    for i, v in enumerate(values):
        if v > best:
            best, best_at = v, i
        if i - best_at >= patience:
            return 100 * (max(values) - best)
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, nargs="+", default=[20, 30, 40, 50, 60, 80, 100, 120, 160])
    args = parser.parse_args()
    rows = load_rows()
    histories = {}
    for d in sorted(glob.glob(os.path.join(R, "*/wandb"))):
        name = os.path.basename(os.path.dirname(d))
        h = [r[KEY] for r in rows(name) if KEY in r]
        if len(h) >= args.min_epochs:
            histories[name] = h
    sig = {k: v for k, v in histories.items() if k.startswith("sig_only_")}
    noisy = {k: v for k, v in histories.items() if not k.startswith("sig_only_")}
    print(f"{len(histories)} histories with >= {args.min_epochs} validation epochs: {len(sig)} signal-only, {len(noisy)} noisy")
    for p in args.patience:
        s = max((shortfall(v, p) for v in sig.values()), default=0.0)
        n = [shortfall(v, p) for v in noisy.values()]
        print(f"patience {p:3d}: largest shortfall signal-only {s:.2f} points; noisy {max(n, default=0):.2f} points, "
              f"{sum(x > 0.5 for x in n)} of {len(n)} noisy runs above 0.5")


if __name__ == "__main__":
    main()
