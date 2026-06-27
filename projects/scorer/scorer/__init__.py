"""Learned replacement for the aframe integration/scoring step.

The detection pipeline turns a network's raw output time series into a
detection statistic by *integrating* it (a boxcar mean, EMA, ...) and then
*clustering* the result into independent triggers.  This package keeps the
clustering (it is what makes the false-alarm rate meaningful) but replaces the
hand-designed integration with a small learned scorer, trained per model on a
held-out-by-time split of that model's own output.

Two scorers are provided:

* ``cnn``      -- a tiny 1-D CNN over a window of the raw response (approach 1)
* ``features`` -- a logistic-regression / gradient-boosting classifier on
                  hand-crafted window features (approach 2)

Both are evaluated by feeding their score through the *same* clustering and
sensitive-volume machinery as the integration methods, on held-out segments,
with the integration baselines recomputed on the same segments for a fair
comparison.
"""
