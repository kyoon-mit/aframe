"""Build the per-model labelled window dataset for training.

Positives are windows of the network response at injections that are loud
enough to actually leave a mark (``snr >= snr_floor``); labelling the hopeless
quiet injections as signal would just be label noise.  Negatives are windows
around background triggers, with the *loudest* background heavily over-sampled
-- those hard negatives are what set the sensitive volume at low FAR.

Only segments in the training half of the time split are used here.
"""

import logging

import h5py
import numpy as np

from .core import (
    build_segment_index,
    find_segment,
    normalize,
    robust_stats,
    sample_for_time,
    time_split,
)


def _extract(ts_group, dataset, index, shift, t, pre, post, rate, fduration):
    """Centred [-pre, +post] sample window around GPS time ``t``; None at an
    edge / missing segment."""
    seg = find_segment(index, shift, t, rate)
    if seg is None:
        return None
    t0, n, key = seg
    j = sample_for_time(t, t0, rate, fduration)
    lo, hi = j - pre, j + post
    if lo < 0 or hi > n:
        return None
    return ts_group[key][dataset][lo:hi]


def build_training_set(run_dir, args, neg_mode="mixed"):
    """Return ``(X, y, snr_target, stats)``.

    X is ``(N, L)`` standardized windows, y is 0/1, snr_target is the injection
    SNR for positives and 0 for negatives, and stats is the per-model
    ``(center, scale)`` standardization.

    ``neg_mode='mixed'`` samples half the negatives from the loudest background
    and half at random; ``neg_mode='hard'`` uses only the loudest background
    (for tail-focused / ranking training)."""
    fg_path = run_dir / "foreground.hdf5"
    bg_path = run_dir / "background.hdf5"
    ts_path = run_dir / "timeseries.hdf5"

    with h5py.File(fg_path, "r") as f:
        p = f["parameters"]
        snr = p["snr"][:]
        inj_time = p["injection_time"][:]
        inj_shift = p["shift"][:].astype(int)
    with h5py.File(bg_path, "r") as f:
        p = f["parameters"]
        bg_stat = p["detection_statistic"][:]
        bg_time = p["detection_time"][:]
        bg_shift = p["shift"][:].astype(int)

    pre = int(args.pre * args.rate)
    post = int(args.post * args.rate)

    pos, neg = [], []
    with h5py.File(ts_path, "r") as f:
        ts = f["timeseries"]
        split = time_split(ts, "background", args.train_frac)
        fg_index = build_segment_index(ts, "foreground")
        bg_index = build_segment_index(ts, "background")

        def in_train(index, shift, t):
            seg = find_segment(index, shift, t, args.rate)
            return seg is not None and seg[2] in split.train

        # ---- positives: loud injections in the training segments ----
        pos_snr = []
        pos_order = np.flatnonzero(snr >= args.snr_floor)
        pos_order = pos_order[np.argsort(snr[pos_order])[::-1]]
        for i in pos_order:
            if len(pos) >= args.max_pos:
                break
            sh = tuple(inj_shift[i])
            if not in_train(fg_index, sh, inj_time[i]):
                continue
            w = _extract(
                ts,
                "foreground",
                fg_index,
                sh,
                float(inj_time[i]),
                pre,
                post,
                args.rate,
                args.fduration,
            )
            if w is not None:
                pos.append(w)
                pos_snr.append(float(snr[i]))

        # ---- negatives: background triggers ----
        # 'mixed'  : top-K loudest plus a random sample of the rest
        # 'hard'   : only the loudest background (tail-focused training)
        bg_order = np.argsort(bg_stat)[::-1]
        if neg_mode == "hard":
            cand = bg_order
        else:
            n_hard = min(args.max_neg // 2, len(bg_order))
            rng = np.random.default_rng(args.seed)
            rest = bg_order[n_hard:]
            rng.shuffle(rest)
            cand = np.concatenate([bg_order[:n_hard], rest])
        for i in cand:
            if len(neg) >= args.max_neg:
                break
            sh = tuple(bg_shift[i])
            if not in_train(bg_index, sh, bg_time[i]):
                continue
            w = _extract(
                ts,
                "background",
                bg_index,
                sh,
                float(bg_time[i]),
                pre,
                post,
                args.rate,
                args.fduration,
            )
            if w is not None:
                neg.append(w)

    if not pos or not neg:
        raise RuntimeError(
            f"insufficient training windows (pos={len(pos)}, neg={len(neg)})"
        )

    # per-model standardization from the background (negative) windows
    stats = robust_stats(np.concatenate(neg))
    X = np.stack([normalize(w, stats) for w in pos + neg]).astype(np.float32)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(
        np.float32
    )
    snr_target = np.concatenate(
        [np.array(pos_snr), np.zeros(len(neg))]
    ).astype(np.float32)
    logging.info(
        "training windows (%s neg): %d pos (snr>=%.1f) / %d neg, L=%d, "
        "stats=%.3g/%.3g",
        neg_mode,
        len(pos),
        args.snr_floor,
        len(neg),
        X.shape[1],
        stats[0],
        stats[1],
    )
    return X, y, snr_target, stats
