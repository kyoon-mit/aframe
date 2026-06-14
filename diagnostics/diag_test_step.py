"""Diagnostic 2 (GPU dump): the model's "test step" on real injections.

No time slides: for every injection we place the kernel exactly where the model
was TRAINED to fire and score it ONCE, on the signal and on the identical
signal-free noise. This is the offline analogue of the model's ``test_step``.

The placement is the kernel alignment ``e`` = (kernel right-edge) − (coalescence):

  - ``--fix-e`` sets it directly. Use the model's trained pre-merger offset, which
    is **minus the training ``window_offset``**: e.g. the id11 / 59-60s model was
    trained with ``window_offset: 3.0`` (kernel [coal−4, coal−3]) so ``--fix-e -3``.
    A merger model (``window_offset 0``) uses ``--fix-e 0``.
  - If ``--fix-e`` is omitted, we instead scan ``[-e-before, e-after]`` and take the
    most-confident foreground alignment. **Beware:** for a pre-merger model the loud
    late-inspiral/merger entering the short kernel near ``e≈0`` is out-of-distribution
    and produces *overconfident wrong* outputs, so the argmin can pick garbage. Prefer
    ``--fix-e`` for this model.

The background is scored at the same window, so each injection yields a matched
signal/noise pair on identical noise. Writes one CSV row per injection: true chirp
mass, both predictions, both sigmas, SNR, the alignment ``best_e``, and the z-score.
Plot with ``notebooks/diag_test_step.ipynb``.

Usage
-----
    python scripts/diag_test_step.py \
        --config projects/train/configs/regression_infer_premerger_1s_id11_1wk_intg.yaml \
        --output runs/.../diag/test_step.csv \
        --fix-e -3.0 --max-injections 2000
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
    ap.add_argument("--fix-e", type=float, default=None,
                    help="fixed kernel alignment = -(training window_offset); "
                         "e.g. -3.0 for id11/59-60s. If set, no scan (recommended).")
    ap.add_argument("--e-before", type=float, default=6.0,
                    help="(scan mode only) seconds of kernel-edge before merger to scan")
    ap.add_argument("--e-after", type=float, default=2.0,
                    help="(scan mode only) seconds of kernel-edge after merger to scan")
    ap.add_argument("--e-step", type=float, default=0.25,
                    help="(scan mode only) alignment scan step (s)")
    ap.add_argument("--max-injections", type=int, default=0,
                    help="stop after this many scored injections (0 = all)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--snr-powerlaw", nargs=3, type=float, metavar=("MIN", "MAX", "ALPHA"),
                    default=[4.0, 50.0, -3.0],
                    help="rescale injection SNR to PowerLaw(min,max,alpha) (training prior)")
    ap.add_argument("--no-snr-rescale", action="store_true",
                    help="use the injection set's native SNRs instead of the powerlaw")
    args = ap.parse_args()

    cfg = load_infer_config(args.config)
    scorer = Scorer(cfg, device=cfg.get("device", "cuda"))
    geom = scorer.geom
    snr_pl = None if args.no_snr_rescale else tuple(args.snr_powerlaw)
    if args.fix_e is not None:
        e_values = np.array([args.fix_e])
    else:
        e_values = np.arange(-args.e_before, args.e_after + 1e-9, args.e_step)

    rows = []
    for bg_fname in list_background_files(cfg):
        if args.max_injections and len(rows) >= args.max_injections:
            break
        seg = load_segment(cfg, bg_fname, snr_powerlaw=snr_pl)
        if seg.foreground is None:
            continue
        inj = seg.injection_set
        chirp = np.asarray(inj.chirp_mass)

        for i in range(len(inj)):
            coal = float(inj.injection_time[i])
            fg_wins, e_kept = windows_for_edges(seg.foreground, seg.t0, geom, coal, e_values)
            if fg_wins is None:
                continue
            bg_wins, _ = windows_for_edges(seg.background, seg.t0, geom, coal, e_values)

            fg_mean, fg_sigma = scorer.score_batched(fg_wins, args.batch_size)
            bg_mean, bg_sigma = scorer.score_batched(bg_wins, args.batch_size)

            k = int(np.argmin(fg_sigma))  # most-confident foreground alignment
            true = float(chirp[i])
            fm, fs = float(fg_mean[k]), float(fg_sigma[k])
            z = (fm - true) / fs if fs > 0 else np.nan
            rows.append((
                float(inj.injection_time[i]), true, float(inj.snr[i]),
                fm, fs, float(bg_mean[k]), float(bg_sigma[k]),
                float(e_kept[k]), z,
            ))
            if args.max_injections and len(rows) >= args.max_injections:
                break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "injection_time", "true_chirp_mass", "snr",
            "fg_chirp_mean", "fg_chirp_sigma",
            "bg_chirp_mean", "bg_chirp_sigma", "best_e", "fg_z",
        ])
        w.writerows(rows)

    print(f"wrote {len(rows)} injections -> {args.output}")


if __name__ == "__main__":
    main()
