"""Plot the raw network response around the loudest signals for each model.

For every run (model) that has ``results_aframe/timeseries.hdf5`` +
``foreground.hdf5``, this picks the few loudest injected signals below a given
SNR cap and plots the *raw* model output (the network statistic time series,
before any integration/clustering) in a window leading up to and shortly after
each signal.  This shows what the response actually looks like, including each
model's baseline and scale.

The same injection set is shared across runs, so the selected signals are the
same for every model and the per-model figures are directly comparable.

A raw sample ``j`` of a segment maps to time ``t0 - fduration/2 + j/rate``
(the inverse of the offset applied in ``infer.postprocess`` /
``scoring_sv.Postprocessor``), so ``t=0`` in each panel is the true injection
time.

Run inside the plots project environment, e.g.::

    uv run --no-sync --directory projects/plots python plot_raw_response.py
"""

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_RUNS_DIR = "/home/barmstrong/aframe_official/runs/aframe_results/runs"
DEFAULT_OUTPUT_DIR = (
    "/home/barmstrong/aframe_official/runs/aframe_results/raw_responses"
)


def parse_key(key):
    t0_str, shift_str = key.split("_", 1)
    shift = tuple(int(x) for x in shift_str.strip("[]").split())
    return float(t0_str), shift


def build_segment_index(ts_group, dataset):
    """Map each timeslide shift -> sorted list of (t0, n_samples, key) for the
    segments whose ``dataset`` ('foreground' or 'background') is populated."""
    index = {}
    for key in ts_group.keys():
        t0, shift = parse_key(key)
        n = ts_group[key][dataset].shape[0]
        if n == 0:
            continue
        index.setdefault(shift, []).append((t0, n, key))
    for shift in index:
        index[shift].sort()
    return index


def find_segment(index, shift, t, rate):
    for t0, n, key in index.get(shift, []):
        if t0 <= t < t0 + n / rate:
            return t0, n, key
    return None


def extract_windows(ts_group, dataset, index, targets, args, pre, post):
    """Pull the raw ``dataset`` response in a [-pre, +post] sample window
    around each target's ``time`` (centred so t=0 is that time). ``targets`` is
    iterable of dicts with at least 'time' and 'shift'; up to ``n_signals``
    plottable windows are returned with 't'/'y' added."""
    picked = []
    for tgt in targets:
        seg = find_segment(index, tgt["shift"], tgt["time"], args.rate)
        if seg is None:
            continue
        t0, n, key = seg
        j = int(round((tgt["time"] - t0 + args.fduration / 2) * args.rate))
        lo, hi = max(0, j - pre), min(n, j + post)
        if hi - lo < (pre + post) // 2:
            continue  # too close to a segment edge to be useful
        y = ts_group[key][dataset][lo:hi]
        tt = (np.arange(lo, hi) - j) / args.rate
        picked.append({**tgt, "t": tt, "y": y})
        if len(picked) == args.n_signals:
            break
    return picked


def plot_model(run_dir, name, args):  # noqa: C901
    fg_path = run_dir / "foreground.hdf5"
    bg_path = run_dir / "background.hdf5"
    ts_path = run_dir / "timeseries.hdf5"
    if not (fg_path.exists() and ts_path.exists()):
        logging.info("skipping %s: missing foreground/timeseries", name)
        return

    with h5py.File(fg_path, "r") as f:
        p = f["parameters"]
        snr = p["snr"][:]
        inj_time = p["injection_time"][:]
        fg_shift = p["shift"][:].astype(int)
        det_time = p["detection_time"][:]

    # loudest signals below the SNR cap (highest SNR first)
    eligible = np.flatnonzero(snr < args.max_snr)
    sig_order = eligible[np.argsort(snr[eligible])[::-1]]

    def sig_targets():
        for i in sig_order:
            yield {
                "shift": tuple(fg_shift[i]),
                "time": float(inj_time[i]),
                "snr": float(snr[i]),
                "det_offset": float(det_time[i] - inj_time[i]),
            }

    # loudest misclassifications == loudest background false alarms (highest
    # detection statistic first)
    bg_targets = None
    if bg_path.exists():
        with h5py.File(bg_path, "r") as f:
            bp = f["parameters"]
            bg_stat = bp["detection_statistic"][:]
            bg_time = bp["detection_time"][:]
            bg_shift = bp["shift"][:].astype(int)
        bg_order = np.argsort(bg_stat)[::-1]

        def bg_targets():  # noqa: F811
            for i in bg_order:
                yield {
                    "shift": tuple(bg_shift[i]),
                    "time": float(bg_time[i]),
                    "stat": float(bg_stat[i]),
                }
    else:
        logging.info("%s: no background.hdf5, plotting signals only", name)

    pre = int(args.pre * args.rate)
    post = int(args.post * args.rate)

    with h5py.File(ts_path, "r") as f:
        ts = f["timeseries"]
        fg_index = build_segment_index(ts, "foreground")
        sig = extract_windows(
            ts, "foreground", fg_index, sig_targets(), args, pre, post
        )
        mis = []
        if bg_targets is not None:
            bg_index = build_segment_index(ts, "background")
            mis = extract_windows(
                ts, "background", bg_index, bg_targets(), args, pre, post
            )

    if not sig and not mis:
        logging.warning("%s: no plottable windows found", name)
        return

    # some models emit the log() of the response; exp() recovers the actual
    # output.  Only transform genuine log-probabilities, i.e. output <= 0
    # everywhere so exp() lands in [0, 1].  Models with positive output would
    # exp() to values > 1 and are left as-is.  Decide once across both rows.
    all_y = np.concatenate([w["y"] for w in sig + mis])
    finite = all_y[np.isfinite(all_y)]
    ymax = finite.max() if finite.size else np.nan
    exp_in_unit = finite.size > 0 and ymax <= 1e-6
    if args.log_transform == "on":
        is_log = True
    elif args.log_transform == "off":
        is_log = False
    else:  # auto
        is_log = exp_in_unit
    if is_log:
        logging.info(
            "%s: log-space output (max=%.3g) -> exp() into [0,1]", name, ymax
        )
        for w in sig + mis:
            w["y"] = np.exp(w["y"])

    rows = [("signals", sig), ("misclassifications", mis)]
    rows = [r for r in rows if r[1]]  # drop an empty row (e.g. no background)
    ncols = max(len(picks) for _, picks in rows)

    fig, axes = plt.subplots(
        len(rows),
        ncols,
        figsize=(4.2 * ncols, 3.4 * len(rows)),
        squeeze=False,
        sharey=True,
        sharex=True,
    )
    for r, (rlabel, picks) in enumerate(rows):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(picks):
                ax.axis("off")
                continue
            w = picks[c]
            ax.plot(w["t"], w["y"], linewidth=1.0, color="#1f77b4")
            ax.axvline(
                0.0, color="k", linestyle="--", linewidth=0.8, label="event"
            )
            if rlabel == "signals":
                if w["t"][0] <= w["det_offset"] <= w["t"][-1]:
                    ax.axvline(
                        w["det_offset"],
                        color="r",
                        linestyle=":",
                        linewidth=0.8,
                        label="recovered",
                    )
                ax.set_title(f"SNR = {w['snr']:.1f}")
            else:
                ax.set_title(f"det stat = {w['stat']:.3g}")
            ax.grid(alpha=0.2)
            if r == len(rows) - 1:
                ax.set_xlabel("time relative to event [s]")
        axes[r][0].set_ylabel(
            f"{rlabel}\n" + ("exp(output)" if is_log else "network output")
        )
        axes[r][0].legend(fontsize=7, loc="upper left")

    transformed = "  [exp-transformed]" if is_log else ""
    fig.suptitle(
        f"{name} — raw response near loudest signals (top) and "
        f"misclassifications (bottom), SNR < {args.max_snr:g}{transformed}"
    )
    fig.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logging.info(
        "%s -> %s (%d signals, %d misclassifications)",
        name,
        out,
        len(sig),
        len(mis),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=DEFAULT_RUNS_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument(
        "--runs", nargs="*", default=None, help="subset of run names"
    )
    ap.add_argument(
        "--n-signals", type=int, default=6, help="signals per model"
    )
    ap.add_argument("--max-snr", type=float, default=15.0)
    ap.add_argument(
        "--pre", type=float, default=8.0, help="seconds before injection"
    )
    ap.add_argument(
        "--post", type=float, default=4.0, help="seconds after injection"
    )
    ap.add_argument(
        "--rate", type=float, default=16.0, help="inference sampling rate"
    )
    ap.add_argument("--fduration", type=float, default=2.0)
    ap.add_argument(
        "--log-transform",
        choices=["auto", "on", "off"],
        default="auto",
        help="exp() the response for log-space models; 'auto' detects them as "
        "mostly-negative output",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    runs_dir = Path(args.runs_dir)
    names = args.runs or [
        d.name for d in sorted(runs_dir.iterdir()) if d.is_dir()
    ]
    for name in names:
        run_dir = runs_dir / name / "results_aframe"
        if not run_dir.is_dir():
            continue
        plot_model(run_dir, name, args)


if __name__ == "__main__":
    main()
