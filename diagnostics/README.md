# Regression-model diagnostics

Offline checks for the S4D chirp-mass **regression** model, independent of the
sensitive-volume pipeline. Each diagnostic is a **GPU dump script** (writes a CSV)
plus a **plotting notebook** (no GPU). They share `diag_common.py`, which loads the
model and preprocessing exactly as `regression_infer.py` does, so the numbers match
the SV pipeline.

| diagnostic | script → notebook | answers |
|---|---|---|
| **1. Score vs kernel alignment** | `diag_score_timeseries.py` → `diag_score_timeseries.ipynb` | *Does the model localize the signal in time, and does the statistic separate signal from noise?* |
| **2. Offline "test step"** | `diag_test_step.py` → `diag_test_step.ipynb` | *At the trained placement, how accurate is the chirp mass, how well-calibrated is σ, does signal separate from noise?* |

> Notebooks are git-ignored repo-wide (`**/*.ipynb`); they live here but are not committed.

---

## The one concept you need: kernel alignment `e`

Everything is indexed by **`e` = (kernel right-edge time) − (coalescence time)**:

- `e = 0` → the kernel ends exactly at the merger.
- `e < 0` → **pre-merger** (kernel ends before the merger).

A window of length `psd_length + fduration + kernel_length` is fed in; PSD is
estimated on the first `psd_length`, whitening trims `fduration/2` from each side,
so the kernel the model actually sees ends `RIGHT_EDGE_OFF = psd_length +
fduration/2 + kernel_length` seconds after the window start. To place a kernel at a
chosen `e`, the window starts at `coalescence + e − RIGHT_EDGE_OFF`.

**This is the only correct placement convention.** Do **not** use the SV recovery
`window_offset` (e.g. 25.5 s) to place a kernel — that is a separate bookkeeping
quantity for matching events to injections, and using it puts the kernel tens of
seconds in the wrong place.

### The trained alignment

A model fires best at the alignment it was trained on: **`e = −(training
window_offset)`**.

| model | training `window_offset` | use `--fix-e` |
|---|---|---|
| id11 / 59-60s (pre-merger) | 3.0 | **−3.0** |
| 58-59s | 4.0 | −4.0 |
| 57-58s | 5.0 | −5.0 |
| merger | 0.0 | 0.0 |

### ⚠ Do not pick the most-confident alignment for a pre-merger model

The model's **confidence keeps growing toward the merger** (the late inspiral is
louder), so the smallest-σ alignment is near `e ≈ 0`, where the merger enters the
short kernel — that is **out-of-distribution**, and the model is **overconfident but
wrong** (e.g. true 1.9 → predicted 0.9 at σ ≈ 0.02). Chirp-mass *accuracy* is only
trustworthy at the trained `e`. That is why Diagnostic 2 fixes `e` instead of taking
`argmin σ`.

---

## Running

Use the project venv on a GPU node (gpu_test, 40 GB is plenty). `CFG` is any
`regression_infer` YAML; `RUN/diag/` is where the notebooks expect the CSVs.

```bash
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/projects/train
PY=.venv/bin/python
D=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/diagnostics
CFG=configs/regression_infer_premerger_1s_id11_1wk_intg.yaml
RUN=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_sv/premerger_1s_id11_1wk_intg

# 1. score vs alignment (scan e), loud injections + background spots
$PY $D/diag_score_timeseries.py --config $CFG --output $RUN/diag/score_timeseries.csv \
    --n-signal 20 --n-background 20 --e-before 8 --e-after 2 --snr-min 12

# 2. offline test step at the TRAINED alignment (id11 -> e = -3)
$PY $D/diag_test_step.py --config $CFG --output $RUN/diag/test_step.csv \
    --fix-e -3.0 --max-injections 2000
```

Then open the matching notebook and point its `CSV =` (and `TRAINED_E`) at the run.

---

## What "good" looks like (id11, verified)

- **Diagnostic 1:** red (signal) `−σ` rises above flat blue (background) across the
  pre-merger range `e ∈ [−4, −1]`; a second dip at `e ≈ 0` is the OOD merger artifact.
- **Diagnostic 2 at `e = −3`, SNR ≥ 25:** corr(true, inferred) ≈ 0.87, ~86 % within
  10 %, fg σ ≈ 0.05 vs bg σ ≈ 0.36. Faint injections converge to noise, as expected.

---

## Implementation notes

- **Detection statistic** is `−σ` (small σ = confident). Model outputs are
  de-normalized to physical units (`mean·y_std + y_mean`, `σ·y_std`); the chirp-mass
  target is **detector-frame** (no redshift correction).
- **`PsdEstimator` batch-of-2 quirk:** a batch with exactly 2 elements is treated as
  a `[psd_source, data]` pair and collapsed to 1 output. `Scorer.score` pads size-2
  batches to 3 and drops the extra, so independent windows are always scored 1:1.
