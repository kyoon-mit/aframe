"""Hand-crafted window features for the approach-2 classifier.

These summarise the shape we saw in the raw-response plots -- how high the
response gets, how much area it accumulates, how wide the elevated plateau is,
and how sharply it rises -- i.e. exactly the things a boxcar mean throws away.
Inputs are already standardized (see ``core.normalize``).
"""

import numpy as np

FEATURE_NAMES = [
    "max",
    "mean",
    "std",
    "ptp",
    "area",
    "energy",
    "frac_above_0",
    "frac_above_2",
    "argmax_rel",
    "max_slope",
    "p50",
    "p90",
    "p99",
]


def window_features(W):
    """``(N, L)`` standardized windows -> ``(N, F)`` feature matrix."""
    W = np.asarray(W, dtype=np.float64)
    n, L = W.shape
    diffs = np.diff(W, axis=1)
    feats = np.stack(
        [
            W.max(axis=1),
            W.mean(axis=1),
            W.std(axis=1),
            W.max(axis=1) - W.min(axis=1),
            W.sum(axis=1) / L,
            (W**2).sum(axis=1) / L,
            (W > 0).mean(axis=1),
            (W > 2).mean(axis=1),
            (W.argmax(axis=1) / L) - 0.5,
            np.abs(diffs).max(axis=1),
            np.percentile(W, 50, axis=1),
            np.percentile(W, 90, axis=1),
            np.percentile(W, 99, axis=1),
        ],
        axis=1,
    )
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
