"""CLI: train and/or evaluate the learned scorers per model.

scorer train    --runs <name> [<name> ...]
scorer evaluate --runs <name> [<name> ...]
scorer run      --runs <name> [<name> ...]   # train then evaluate
"""

import argparse
import logging
from pathlib import Path

DEFAULT_RUNS_DIR = "/home/barmstrong/aframe_official/runs/aframe_results/runs"
DEFAULT_OUT = "/home/barmstrong/aframe_official/runs/aframe_results/scorer"
DEFAULT_REFERENCE = (
    "/home/barmstrong/aframe_official/runs/aframe_results/reference/"
    "aframe-decimator-sv.h5"
)
DEFAULT_REJECTED = (
    "/fast/barmstrong/ligoss/data/bns/aframe_test/rejected-parameters.hdf5"
)
PROTOTYPE = ["s4d_merger_1s", "linoss_4s_merger"]


def add_common(ap):
    ap.add_argument(
        "--runs", nargs="*", default=PROTOTYPE, help="run names to process"
    )
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--train-frac", type=float, default=0.6)
    # window + run geometry
    ap.add_argument(
        "--pre", type=float, default=4.0, help="window seconds before centre"
    )
    ap.add_argument(
        "--post", type=float, default=4.0, help="window seconds after centre"
    )
    ap.add_argument("--rate", type=float, default=16.0)
    ap.add_argument("--fduration", type=float, default=2.0)
    ap.add_argument("--cluster-window", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"]
    )


def add_train(ap):
    ap.add_argument(
        "--objectives",
        nargs="*",
        default=["classify", "rank", "snr", "detect"],
        choices=["classify", "rank", "snr", "detect"],
        help="training objectives to fit per model",
    )
    ap.add_argument(
        "--snr-floor",
        type=float,
        default=12.0,
        help="min SNR for training positives",
    )
    ap.add_argument("--max-pos", type=int, default=15000)
    ap.add_argument("--max-neg", type=int, default=30000)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--kernel", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--margin", type=float, default=1.0, help="rank-loss margin"
    )
    ap.add_argument(
        "--target-fpr",
        type=float,
        default=0.02,
        help="detect objective: background quantile used as the threshold",
    )


def add_eval(ap):
    ap.add_argument(
        "--methods",
        nargs="*",
        default=[
            "raw",
            "box:0.125",
            "box:0.25",
            "box:0.5",
            "box:1.0",
            "gauss:0.25",
            "tri:0.25",
            "median:0.1875",
            "max:0.25",
        ],
        help="integration-method specs to evaluate (see pipeline.make_method)",
    )
    ap.add_argument(
        "--cluster-windows",
        nargs="*",
        type=float,
        default=[],
        help="clustering-window sweep (s); held at --cluster-sweep-method",
    )
    ap.add_argument(
        "--cluster-sweep-method",
        default="box:0.25",
        help="integration method held fixed during the cluster-window sweep",
    )
    ap.add_argument(
        "--skip-learned",
        action="store_true",
        help="evaluate only the integration methods, skip the trained CNNs",
    )
    ap.add_argument(
        "--baseline-win",
        type=float,
        default=1.0,
        help="boxcar window used to propose candidates for the feature scorer",
    )
    ap.add_argument(
        "--stride", type=int, default=4, help="CNN sliding stride (samples)"
    )
    ap.add_argument("--max-segments", type=int, default=None)
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--reference", default=DEFAULT_REFERENCE)
    ap.add_argument("--rejected-params", default=DEFAULT_REJECTED)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("train", "evaluate", "run"):
        sp = sub.add_parser(cmd)
        add_common(sp)
        if cmd in ("train", "run"):
            add_train(sp)
        if cmd in ("evaluate", "run"):
            add_eval(sp)
    cp = sub.add_parser(
        "combine", help="overlay one method's SV across models"
    )
    add_common(cp)
    cp.add_argument(
        "--method",
        default="tri:0.25",
        help="integration method whose SV curve to overlay per model",
    )
    cp.add_argument(
        "--out-plot",
        default=None,
        help="output png (default: <out>/combined_<method>.png)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs_dir = Path(args.runs_dir)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # imported here so `train` doesn't pull the SV/plotting stack unnecessarily
    if args.cmd in ("train", "run"):
        from .train import train_model
    if args.cmd in ("evaluate", "run"):
        from .evaluate import evaluate_model

    if args.cmd == "combine":
        from .combine import combine_models

        out_plot = (
            Path(args.out_plot)
            if args.out_plot
            else (out_root / f"combined_{args.method.replace(':', '_')}.png")
        )
        combine_models(args.runs, out_root, args.method, out_plot)
        return

    for name in args.runs:
        run_dir = runs_dir / name / "results_aframe"
        if not (run_dir / "timeseries.hdf5").exists():
            logging.warning("skipping %s: no timeseries.hdf5", name)
            continue
        logging.info("=== %s ===", name)
        if args.cmd in ("train", "run"):
            train_model(run_dir, name, args, out_root)
        if args.cmd in ("evaluate", "run"):
            evaluate_model(run_dir, name, args, out_root / name, out_root)


if __name__ == "__main__":
    main()
