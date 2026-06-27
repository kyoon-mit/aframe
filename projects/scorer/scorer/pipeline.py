"""Turn a per-segment score into background/foreground ledgers and sensitive
volume -- the same path the integration methods take, so learned scorers and
baselines are compared on identical footing.

Everything here operates on a chosen set of segment keys (the held-out test
split), and the foreground is subset to the injections that fall in those
segments so the comparison is self-consistent across methods.
"""

import hashlib
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from ledger.events import EventSet, RecoveredInjectionSet
from plots.legacy import main as legacy_main
from plots.legacy.main import main as calc_sensitive_volume
from priors.priors import end_o3_ratesandpops_bns

# The GWTC-3 reference curves inside ``calc_sensitive_volume`` depend only on
# the mass combos and FAR thresholds, which are identical across the methods
# we evaluate (same held-out Tb).  Cache them so we pay the slow pipeline once
# instead of once per method -- same trick as scoring_sv.
_gwtc3 = legacy_main.gwtc3_pipeline_sv
_GWTC3_CACHE = {}


def _cached_gwtc3(
    mass_combos,
    injection_file,
    detection_criterion,
    detection_thresholds,
    output_dir,
    **kw,
):
    thr = np.asarray(detection_thresholds)
    key = (
        detection_criterion,
        tuple(map(tuple, mass_combos)),
        thr.shape,
        hashlib.sha1(thr.tobytes()).hexdigest(),
    )
    if key not in _GWTC3_CACHE:
        _GWTC3_CACHE[key] = _gwtc3(
            mass_combos=mass_combos,
            injection_file=injection_file,
            detection_criterion=detection_criterion,
            detection_thresholds=detection_thresholds,
            output_dir=output_dir,
            **kw,
        )
    return _GWTC3_CACHE[key]


legacy_main.gwtc3_pipeline_sv = _cached_gwtc3

from .core import (  # noqa: E402
    FDURATION,
    IFOS,
    MASS_COMBOS,
    PSD_LENGTH,
    RATE,
    ZEROLAG_SHIFT,
    build_segment_index,
    find_segment,
    parse_key,
)


# --------------------------------------------------------------------------- #
# clustering + recovery (faithful to scoring_sv / infer.postprocess)
# --------------------------------------------------------------------------- #
def cluster(score, t0_eff, shift, args, cluster_window=None):
    cw_seconds = (
        args.cluster_window if cluster_window is None else cluster_window
    )
    cw = int(args.rate * cw_seconds)
    window_size = int(cw // 2)
    i = int(np.argmax(score[:window_size])) if window_size else 0
    events, times = [], []
    while i < len(score):
        val = score[i]
        window = score[i + 1 : i + 1 + window_size]
        if (val < window).any():
            i += int(np.argmax(window)) + 1
        else:
            events.append(val)
            times.append(t0_eff + i / args.rate)
            i += window_size + 1
    Tb = len(score) / args.rate
    events = np.array(events)
    times = np.array(times)
    shifts = np.ones((len(events), len(shift))) * shift
    return EventSet(events, times, shifts, Tb)


def nearest_recover(ev_times, ev_stats, inj_times):
    if ev_times.size == 0:
        return np.full(inj_times.shape, -np.inf), inj_times.copy()
    order = np.argsort(ev_times)
    et, es = ev_times[order], ev_stats[order]
    pos = np.clip(np.searchsorted(et, inj_times), 1, len(et) - 1)
    left, right = pos - 1, pos
    take_left = np.abs(inj_times - et[left]) <= np.abs(et[right] - inj_times)
    choice = np.where(take_left, left, right)
    return es[choice], et[choice]


# --------------------------------------------------------------------------- #
# per-segment event producers
# --------------------------------------------------------------------------- #
def boxcar(win_seconds, rate=RATE):
    """The production mean-integration; also used to propose candidates for the
    feature scorer.  Returns (score_fn, lag_seconds).  This is a *trailing*
    mean, so its output lags the signal peak by ``win_seconds``."""
    size = int(rate * win_seconds) + 1
    kernel = np.ones(size) / size

    def fn(y):
        return np.convolve(y, kernel, mode="full")[: -size + 1]

    return fn, win_seconds


def make_method(spec, rate=RATE):
    """Parse an integration-method spec into ``(score_fn, lag_seconds)``.

    Specs (``w`` in seconds):

    * ``raw``       -- identity, no integration (the per-sample response).
    * ``box:w``     -- trailing boxcar mean (the production method); lag ``w``.
    * ``gauss:w``   -- centred Gaussian, sigma ``w``; lag 0.  Soft-tapered.
    * ``tri:w``     -- centred triangular (Bartlett) window, half-width ``w``.
    * ``median:w``  -- centred rolling median; kills lone-sample glitches while
                       keeping a real peak.  lag 0.
    * ``max:w``     -- centred rolling max; peak-preserving (tends to lift the
                       background -- a useful control).  lag 0.
    * ``ewma:w``    -- causal exponential moving average, timescale ``w``; lag
                       ``w``.
    """
    from scipy import ndimage, signal

    if spec == "raw":
        return (lambda y: np.asarray(y, dtype=np.float64)), 0.0

    kind, _, val = spec.partition(":")
    w = float(val) if val else 0.0

    if kind == "box":
        return boxcar(w, rate)

    if kind == "gauss":
        sigma = max(rate * w, 1e-6)
        return (
            lambda y: ndimage.gaussian_filter1d(
                np.asarray(y, dtype=np.float64), sigma, mode="nearest"
            )
        ), 0.0

    if kind == "tri":
        size = 2 * int(rate * w) + 1
        kernel = np.bartlett(size)
        kernel /= kernel.sum()
        return (
            lambda y: np.convolve(
                np.asarray(y, dtype=np.float64), kernel, mode="same"
            )
        ), 0.0

    if kind == "median":
        size = max(2 * int(rate * w) + 1, 1)
        return (
            lambda y: ndimage.median_filter(
                np.asarray(y, dtype=np.float64), size=size, mode="nearest"
            )
        ), 0.0

    if kind == "max":
        size = max(2 * int(rate * w) + 1, 1)
        return (
            lambda y: ndimage.maximum_filter1d(
                np.asarray(y, dtype=np.float64), size=size, mode="nearest"
            )
        ), 0.0

    if kind == "ewma":
        alpha = 1.0 / (rate * w + 1.0)
        return (
            lambda y: signal.lfilter(
                [alpha], [1.0, -(1.0 - alpha)], np.asarray(y, dtype=np.float64)
            )
        ), w

    raise ValueError(f"unknown integration method spec: {spec!r}")


def dense_producer(score_fn, lag, args, cluster_window=None):
    """per-segment function: drop burn-in, score densely, cluster."""
    offset = int(PSD_LENGTH * args.rate)

    def produce(y_raw, t0, shift):
        if y_raw.size <= offset:
            return EventSet()
        y = y_raw[offset:]
        s = np.asarray(score_fn(y))[: len(y)]
        t0_eff = t0 + PSD_LENGTH - FDURATION / 2 - lag
        return cluster(s, t0_eff, shift, args, cluster_window)

    return produce


def candidate_producer(base_score_fn, base_lag, window_prob_fn, args):
    """per-segment function for the feature scorer: cluster on the boxcar to
    propose candidates, then re-score each candidate window with the classifier
    and use that probability as the statistic."""
    offset = int(PSD_LENGTH * args.rate)
    pre = int(args.pre * args.rate)
    post = int(args.post * args.rate)

    def produce(y_raw, t0, shift):
        if y_raw.size <= offset:
            return EventSet()
        y = y_raw[offset:]
        s = np.asarray(base_score_fn(y))[: len(y)]
        t0_eff = t0 + PSD_LENGTH - FDURATION / 2 - base_lag
        base = cluster(s, t0_eff, shift, args)
        if len(base) == 0:
            return base
        # window sample index of each candidate within the post-offset array
        idx = np.round((base.detection_time - t0_eff) * args.rate).astype(int)
        windows, keep = [], []
        for k, j in enumerate(idx):
            lo, hi = j - pre, j + post
            if lo < 0 or hi > len(y):
                continue
            windows.append(y[lo:hi])
            keep.append(k)
        if not windows:
            return EventSet()
        probs = window_prob_fn(np.stack(windows))
        keep = np.array(keep)
        shifts = np.ones((len(keep), len(shift))) * shift
        return EventSet(
            np.asarray(probs), base.detection_time[keep], shifts, base.Tb
        )

    return produce


# --------------------------------------------------------------------------- #
# injection truth restricted to the test segments
# --------------------------------------------------------------------------- #
def test_injections(fg_path, ts_path, test_keys, args):
    with h5py.File(fg_path, "r") as f:
        inj_time = f["parameters"]["injection_time"][:]
        inj_shift = f["parameters"]["shift"][:].astype(int)
    with h5py.File(ts_path, "r") as f:
        fg_index = build_segment_index(f["timeseries"], "foreground")

    mask = np.zeros(len(inj_time), dtype=bool)
    for i in range(len(inj_time)):
        seg = find_segment(
            fg_index, tuple(inj_shift[i]), inj_time[i], args.rate
        )
        if seg is not None and seg[2] in test_keys:
            mask[i] = True

    times = inj_time[mask]
    shifts = inj_shift[mask]
    by_shift = {}
    for sh in np.unique(shifts, axis=0):
        by_shift[tuple(sh)] = np.flatnonzero((shifts == sh).all(axis=1))
    return mask, times, by_shift


# --------------------------------------------------------------------------- #
# full evaluation of one per-segment producer
# --------------------------------------------------------------------------- #
def evaluate(run_dir, test_keys, produce, inj, args, out_dir, rejected_params):
    """Run ``produce`` over the test segments, build background/foreground and
    compute sensitive volume.  ``inj`` is the output of
    :func:`test_injections`.  Returns the path to ``sensitive_volume.h5``."""
    fg_path = run_dir / "foreground.hdf5"
    ts_path = run_dir / "timeseries.hdf5"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    test_mask, inj_times, by_shift = inj
    bg_stats, bg_times, bg_shifts = [], [], []
    Tb_total = 0.0
    fg_by_shift = {sh: ([], []) for sh in by_shift}

    with h5py.File(ts_path, "r") as f:
        ts = f["timeseries"]
        keys = [k for k in ts.keys() if k in test_keys]
        for key in tqdm(keys, desc=out_dir.name, leave=False):
            t0, shift = parse_key(key)
            grp = ts[key]
            # background -> FAR (zero-lag excluded)
            if shift != ZEROLAG_SHIFT and grp["background"].shape[0]:
                ev = produce(grp["background"][:], t0, shift)
                if len(ev):
                    bg_stats.append(ev.detection_statistic)
                    bg_times.append(ev.detection_time)
                    bg_shifts.append(ev.shift)
                Tb_total += ev.Tb
            # foreground -> recovery
            if shift in fg_by_shift and grp["foreground"].shape[0]:
                ev = produce(grp["foreground"][:], t0, shift)
                if len(ev):
                    fg_by_shift[shift][0].append(ev.detection_time)
                    fg_by_shift[shift][1].append(ev.detection_statistic)

    if not bg_stats:
        raise RuntimeError(f"no background events produced in {out_dir}")
    background = EventSet(
        detection_statistic=np.concatenate(bg_stats),
        detection_time=np.concatenate(bg_times),
        shift=np.concatenate(bg_shifts),
        Tb=Tb_total,
    )
    background.write(out_dir / "background.hdf5")

    # recover the test injections
    recovered_stat = np.full(len(inj_times), -np.inf)
    recovered_time = inj_times.copy()
    for sh, idx in by_shift.items():
        times_list, stats_list = fg_by_shift[sh]
        if not times_list:
            continue
        stat, time = nearest_recover(
            np.concatenate(times_list),
            np.concatenate(stats_list),
            inj_times[idx],
        )
        recovered_stat[idx] = stat
        recovered_time[idx] = time

    # write a foreground ledger holding only the test injections
    fg = RecoveredInjectionSet.read(fg_path)[test_mask]
    fg.detection_statistic = recovered_stat
    fg.detection_time = recovered_time
    fg.Tb = Tb_total
    fg_out = out_dir / "foreground.hdf5"
    fg.write(fg_out)

    try:
        calc_sensitive_volume(
            background=out_dir / "background.hdf5",
            foreground=fg_out,
            rejected_params=Path(rejected_params),
            ifos=IFOS,
            mass_combos=MASS_COMBOS,
            source_prior=end_o3_ratesandpops_bns,
            output_dir=out_dir,
            dt=args.dt,
            backend="matplotlib",
        )
    except (
        Exception
    ) as exc:  # e.g. too little held-out livetime for any FAR bin
        logging.warning(
            "SV failed for %s (Tb=%.0fs): %s", out_dir.name, Tb_total, exc
        )
        return None
    logging.info(
        "evaluated %s (Tb=%.0fs, %d bg events)",
        out_dir.name,
        Tb_total,
        len(background),
    )
    return out_dir / "sensitive_volume.h5"
