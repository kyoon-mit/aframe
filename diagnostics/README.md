# Regression-model diagnostics

Offline checks for the S4D chirp-mass **regression** model, independent of the
sensitive-volume pipeline. Each diagnostic is a **GPU dump script** (writes a CSV)
plus a **plotting notebook** (no GPU). They share `diag_common.py`, which loads the
model and preprocessing exactly as `regression_infer.py` does, so the numbers match
the SV pipeline.

| diagnostic | script | answers |
|---|---|---|
| **1. Score vs kernel alignment** | `diag_score_timeseries.py` | *Does the model localize the signal in time, and does the statistic separate signal from noise?* |
| **2. Offline "test step"** | `diag_test_step.py` | *At the trained placement, how accurate is the chirp mass, how well-calibrated is σ, does signal separate from noise?* |

The `.py` scripts are **generic** (driven entirely by a config + flags). The
**notebooks are per-model** (each hard-codes its CSV path and `TRAINED_E`):

| model | config | diagnostic 1 notebook | diagnostic 2 notebook |
|---|---|---|---|
| pre-merger 1 s (59-60s, id11) | `regression_infer_diag_premerger_59-60s_ft.yaml` | `diag_score_timeseries_premerger_59-60s.ipynb` | `diag_test_step_premerger_59-60s.ipynb` |
| merger 4 s (60-64s) | `regression_infer_diag_merger_60-64s_ft.yaml` | `diag_score_timeseries_merger_60-64s.ipynb` | — |
| merger 1 s (63-64s) | `regression_infer_diag_merger_63-64s_ft.yaml` | `diag_score_timeseries_merger_63-64s.ipynb` | — |

All three configs use the **fine-tuned (`_ft`) checkpoints** and copy their geometry
verbatim from the matching training config in `configs/{premerger,merger}/`.

> Notebooks are git-ignored repo-wide (`**/*.ipynb`); they live here but are not committed.

## Injection SNR: powerlaw, matching training

By default both scripts **rescale every injection's SNR to `PowerLaw(4, 50, α=−3)`**
(`--snr-powerlaw MIN MAX ALPHA`), the same prior the models were trained/validated on
(`train.augmentations.SnrSampler`). The stored injection set carries a broad
astrophysical SNR range (~0.2–73); rescaling makes the diagnostics reflect the
model's actual operating regime. SNR is linear in signal amplitude (fixed noise), so
each per-IFO response is scaled by `target/stored`. Pass `--no-snr-rescale` to use the
set's native SNRs instead.

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
| 59-60s / id11 (pre-merger 1 s) | 3.0 | **−3.0** |
| 58-59s | 4.0 | −4.0 |
| 57-58s | 5.0 | −5.0 |
| merger 4 s (60-64s) / merger 1 s (63-64s) | 0.0 | 0.0 |

### ⚠ Do not pick the most-confident alignment for a pre-merger model

The model's **confidence keeps growing toward the merger** (the late inspiral is
louder), so the smallest-σ alignment is near `e ≈ 0`, where the merger enters the
short kernel — that is **out-of-distribution**, and the model is **overconfident but
wrong** (e.g. true 1.9 → predicted 0.9 at σ ≈ 0.02). Chirp-mass *accuracy* is only
trustworthy at the trained `e`. That is why Diagnostic 2 fixes `e` instead of taking
`argmin σ`.

---

## Running

Use the project venv on a GPU node (gpu_test, 40 GB is plenty). Outputs go under
`runs/regression_sv/diag/<model>/`, which is where the notebooks read their CSVs.
SNR is rescaled to the powerlaw prior by default. The committed launcher
`runs/regression_sv/diag/regen_all.sbatch` runs all of the below in one job.

```bash
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/projects/train
PY=.venv/bin/python
D=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/diagnostics
C=configs ; O=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_sv/diag

# pre-merger 1 s (59-60s): scan + test step at the trained alignment e = -3
$PY $D/diag_score_timeseries.py --config $C/regression_infer_diag_premerger_59-60s_ft.yaml \
    --output $O/premerger_59-60s_ft/score_timeseries.csv --e-before 8 --e-after 2 --snr-min 12
$PY $D/diag_test_step.py --config $C/regression_infer_diag_premerger_59-60s_ft.yaml \
    --output $O/premerger_59-60s_ft/test_step.csv --fix-e -3.0 --max-injections 2000

# merger 4 s (60-64s) and merger 1 s (63-64s): scan around e = 0
$PY $D/diag_score_timeseries.py --config $C/regression_infer_diag_merger_60-64s_ft.yaml \
    --output $O/merger_60-64s_ft/score_timeseries.csv --e-before 5 --e-after 4 --snr-min 12
$PY $D/diag_score_timeseries.py --config $C/regression_infer_diag_merger_63-64s_ft.yaml \
    --output $O/merger_63-64s_ft/score_timeseries.csv --e-before 5 --e-after 4 --snr-min 12
```

Then open the matching notebook (its `CSV =` and `TRAINED_E` already point at the run).

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
