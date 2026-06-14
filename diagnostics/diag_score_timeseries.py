"""Diagnostic 1 (GPU dump): the model's score *as a function of kernel alignment*.

For a handful of signal events (loud injections) and background spots (no signal),
this slides the model's kernel across the merger and records the score at every
alignment. The result is one score-vs-time curve per event:

  - a SIGNAL curve should dip in sigma (rise in the statistic -sigma) as the kernel
    lines up with the signal, peaking near the trained pre-merger offset;
  - a BACKGROUND curve should stay flat and uncertain.

This script only *produces the data* (a CSV); plot it interactively with
``notebooks/diag_score_timeseries.ipynb`` (plotly, no GPU needed).

The x-axis is ``e`` = (kernel right-edge time) - (coalescence). So ``e = 0`` means
the kernel ends exactly at the merger; ``e < 0`` is pre-merger. A pre-merger model
should peak at negative e (e.g. ~-3 for the id11 3 s-before model). Background
spots use a random reference time, so their curves are flat noise.

Usage
-----
    python scripts/diag_score_timeseries.py \
        --config projects/train/configs/regression_infer_premerger_1s_id11_1wk_intg.yaml \
        --output runs/.../diag/score_timeseries.csv \
        --n-signal 20 --n-background 20 --e-before 8 --e-after 2 --snr-min 15
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from diag_common import (
    Scorer, load_infer_config, load_segment, list_background_files, windows_for_edges,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-signal", type=int, default=20)
    ap.add_argument("--n-background", type=int, default=20)
    ap.add_argument("--e-before", type=float, default=8.0,
                    help="seconds of kernel-edge BEFORE the merger to scan")
    ap.add_argument("--e-after", type=float, default=2.0,
                    help="seconds of kernel-edge AFTER the merger to scan")
    ap.add_argument("--snr-min", type=float, default=15.0,
                    help="only show injections at least this loud")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-powerlaw", nargs=3, type=float, metavar=("MIN", "MAX", "ALPHA"),
                    default=[4.0, 50.0, -3.0],
                    help="rescale injection SNR to PowerLaw(min,max,alpha) (training prior)")
    ap.add_argument("--no-snr-rescale", action="store_true",
                    help="use the injection set's native SNRs instead of the powerlaw")
    args = ap.parse_args()

    cfg = load_infer_config(args.config)
    scorer = Scorer(cfg, device=cfg.get("device", "cuda"))
    geom = scorer.geom
    rng = np.random.default_rng(args.seed)
    snr_pl = None if args.no_snr_rescale else tuple(args.snr_powerlaw)

    # kernel-edge alignments to scan, at the inference cadence
    e_values = np.arange(-args.e_before, args.e_after + 1e-9, 1.0 / geom.inference_sampling_rate)

    rows = []  # (event_id, kind, snr, e, neg_sigma, chirp_mean, chirp_sigma)
    n_sig, n_bg = 0, 0
    # keep events clear of segment edges: need the full window plus the scan span
    edge = args.e_before + geom.right_edge_offset + geom.sample_length + 2.0

    for bg_fname in list_background_files(cfg):
        if n_sig >= args.n_signal and n_bg >= args.n_background:
            break
        seg = load_segment(cfg, bg_fname, snr_powerlaw=snr_pl)
        if seg.foreground is None:
            continue
        inj = seg.injection_set
        seg_end = seg.t0 + seg.background.shape[1] / geom.raw_sample_rate

        # ---- signal events: the loudest injections in this segment ----------
        if n_sig < args.n_signal:
            for idx in np.argsort(inj.snr)[::-1]:
                if n_sig >= args.n_signal or inj.snr[idx] < args.snr_min:
                    break
                coal = float(inj.injection_time[idx])
                if not (seg.t0 + edge < coal < seg_end - edge):
                    continue
                wins, e_kept = windows_for_edges(seg.foreground, seg.t0, geom, coal, e_values)
                if wins is None:
                    continue
                mean, sigma = scorer.score_batched(wins, args.batch_size)
                eid = f"signal_{n_sig:02d}"
                for e, m, s in zip(e_kept, mean, sigma):
                    rows.append((eid, "signal", float(inj.snr[idx]),
                                 float(e), float(-s), float(m), float(s)))
                n_sig += 1

        # ---- background events: random spots, far from any injection --------
        if n_bg < args.n_background:
            inj_times = np.asarray(inj.injection_time)
            attempts = 0
            while n_bg < args.n_background and attempts < 200:
                attempts += 1
                center = rng.uniform(seg.t0 + edge, seg_end - edge)
                if np.any(np.abs(inj_times - center) < args.e_before + args.e_after + 5.0):
                    continue
                wins, e_kept = windows_for_edges(seg.background, seg.t0, geom, center, e_values)
                if wins is None:
                    continue
                mean, sigma = scorer.score_batched(wins, args.batch_size)
                eid = f"background_{n_bg:02d}"
                for e, m, s in zip(e_kept, mean, sigma):
                    rows.append((eid, "background", np.nan,
                                 float(e), float(-s), float(m), float(s)))
                n_bg += 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event_id", "kind", "snr", "e",
                    "neg_sigma", "chirp_mean", "chirp_sigma"])
        w.writerows(rows)

    print(f"wrote {len(rows)} rows for {n_sig} signal + {n_bg} "
          f"background events -> {args.output}")


if __name__ == "__main__":
    main()
