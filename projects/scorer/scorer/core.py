"""Segment bookkeeping shared across the scorer package.

The conventions here mirror ``scoring_sv.Postprocessor`` /
``infer.postprocess`` so that anything we build lands on the same time axis as
the production pipeline:

* a segment's raw array index ``j`` maps to GPS time
  ``t0 - fduration/2 + j/rate``
* the first ``psd_length * rate`` samples are PSD burn-in and are dropped
  before scoring/clustering, after which sample ``i`` maps to
  ``t0 + psd_length - fduration/2 + i/rate``
"""

from dataclasses import dataclass

import numpy as np

# Run parameters (defaults from the infer condor config for these runs).
RATE = 16.0  # inference_sampling_rate [Hz]
PSD_LENGTH = 64.0  # seconds sliced off the front of every segment
FDURATION = 2.0  # whitening filter length [s]
CLUSTER_WINDOW = 8.0  # clustering window [s]
ZEROLAG_SHIFT = (0, 0)  # un-shifted analysis, excluded from the background

MASS_COMBOS = [[1.4, 1.4], [1.5, 1.5], [2.0, 2.0], [2.3, 2.3]]
IFOS = ["H1", "L1"]


def parse_key(key):
    """``'1241443783.0_[0 1]'`` -> (t0, shift tuple)."""
    t0_str, shift_str = key.split("_", 1)
    shift = tuple(int(x) for x in shift_str.strip("[]").split())
    return float(t0_str), shift


def build_segment_index(ts_group, dataset):
    """Map each timeslide shift -> sorted list of ``(t0, n_samples, key)`` for
    the segments whose ``dataset`` ('foreground'/'background') is populated."""
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


def sample_for_time(t, t0, rate=RATE, fduration=FDURATION):
    """Raw-array sample index whose centre is GPS time ``t``."""
    return int(round((t - t0 + fduration / 2) * rate))


@dataclass
class SplitKeys:
    """Segment keys assigned to the train/test halves of a model's output."""

    train: set
    test: set


def time_split(ts_group, dataset, train_frac):
    """Split a model's segments into train/test by GPS time so there is no
    temporal leakage.  All timeslide shifts of a given ``t0`` go to the same
    side of the split."""
    index = build_segment_index(ts_group, dataset)
    t0s = sorted({t0 for segs in index.values() for (t0, _, _) in segs})
    if not t0s:
        return SplitKeys(set(), set())
    cut = t0s[int(len(t0s) * train_frac)] if len(t0s) > 1 else t0s[0] + 1
    train, test = set(), set()
    for segs in index.values():
        for t0, _, key in segs:
            (train if t0 < cut else test).add(key)
    return SplitKeys(train, test)


def normalize(arr, stats):
    """Standardize a raw response array with stored ``(center, scale)``."""
    center, scale = stats
    out = (np.asarray(arr, dtype=np.float64) - center) / scale
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def robust_stats(values):
    """Median / scaled-MAD, a per-model standardization robust to the wildly
    different output ranges across models (log-prob, positive, ~-800, ...)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    center = float(np.median(v))
    mad = float(np.median(np.abs(v - center)))
    scale = 1.4826 * mad if mad > 0 else (float(np.std(v)) or 1.0)
    return center, scale
