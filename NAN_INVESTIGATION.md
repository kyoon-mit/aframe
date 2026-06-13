# NaN Investigation — `chirp_mass_snr_4_50_59-60s_d64_s64_l4_id8`

**Date:** 2026-06-01
**Run:** `kyoon-mit-massachusetts-institute-of-technology/aframe/chirp_mass_snr_4_50_59-60s_d64_s64_l4_id8`
**Branch / commit:** `kyoon-dev` @ `7c795ad` (working tree clean — no committed code modified during this investigation)

---

## TL;DR

The training loss is `NaN` because the **waveform generator (`ml4gw.waveforms.IMRPhenomPv2`) returns an all-NaN waveform for near-equal-mass precessing systems** (mass ratio `q = m2/m1 ≳ 0.999`, i.e. `|m1−m2|/m1 ≲ 1e-3`). The prior in the live config samples `mass_ratio ~ Uniform(0.4, 1.0)`, so it draws these lethal samples at a rate of **~1.25×10⁻⁴ per injection** (~1 in 8,000). A **single** NaN waveform poisons the model weights permanently (NaN loss → NaN grads → NaN weights), which is why the loss is NaN for **every step of every epoch** afterward.

It is **not** a gradient explosion, **not** the `target/snr` SNR-rescale division, **not** bad background data, and **not** the model/loss code. Gradient clipping does nothing. The background HDF5 files are clean.

**Fix (config only):** cap the prior at `mass_ratio: high: 0.99` (was `1.0`). This removes 100% of the observed NaN samples, has negligible effect on the chirp-mass regression target, and requires **no change to committed code**.

---

## 1. Symptom

- `val/mse/out_0 = nan` at the very first validation; `train/gaussnll = nan` throughout.
- EarlyStopping repeatedly fires (`Monitored metric val/mse/out_0 = nan is not finite`) but `min_epochs=100` keeps the (dead) run alive.
- In the afternoon run `slurm_17913273` (same current code+config), training was **healthy** — `train/loss_step` fell `1.37 → 0.5` over the first 81 steps — then went **NaN at step 82** and stayed NaN for all 39 epochs that followed.

That "healthy, then suddenly NaN at one step, then NaN forever" shape is the key clue: a one-off poisoning event, not a gradual divergence.

## 2. Root cause

The on-the-fly injection pipeline (`RegressionTimeDomainDataset._inject_from_generator`) does:

```
prior sample → IMRPhenomPv2 waveform → slice (pre-merger window) → project
            → SNR rescale → inject into background → whiten → (X, y)
```

A stage-by-stage finiteness probe (`debug_stage.py`) showed the **first** non-finite tensor is the **waveform itself**:

```
[LOCALIZE] >>> first NON-FINITE at stage 'waveform_generator' (train step 74) <<<
hc = shape=(256, 16384) finite=False n_nonfinite=16384   # entire waveform NaN
hp = shape=(256, 16384) finite=False n_nonfinite=16384
first_bad_params = {chirp_mass: 1.408, mass_ratio: 0.99966,
                    mass_1: 1.61765, mass_2: 1.61711,  # ← essentially equal masses
                    a_1: 0.074, a_2: 0.166, tilt_1: 0.79, tilt_2: 1.22, ...}
```

Everything downstream (`response → compute_network_snr → rescaled response → whitener → loss`) is just the NaN propagating.

A controlled `mass_ratio` sweep on that exact sample (`debug_waveform.py`):

| `mass_ratio` | `m1, m2` (`|m1−m2|`) | waveform |
|---|---|---|
| 0.99966 | 1.61765, 1.61711 (5.4e-4) | **NaN** |
| 0.9999  | (1.6e-4) | **NaN** |
| 0.999   | (1.6e-3) | finite |
| 0.995 / 0.99 / 0.95 / 0.8 / 0.5 | … | finite |

So `IMRPhenomPv2` produces NaN once the two masses are within ~0.1% of each other — the classic **equal-mass precessing singularity** (the precession-angle / spin-rotation math divides by the mass difference). The non-precessing path (`IMRPhenomD` / aligned spins) does not have this.

Monte-Carlo over **40,000** real prior draws:

```
nan_waveforms = 5 / 40000   (frac = 1.25e-4)
mass_ratio of NaN samples ∈ [0.99960, 0.99990]   (ALL q > 0.999)
prior draws with q > 0.99  : 1.64%
prior draws with q > 0.999 : 0.18%
```

Every NaN is a near-equal-mass sample. None occur away from `q→1`.

## 3. Why one bad batch makes *every* epoch NaN

This answers "if it's a bad batch, why does it continue across all epochs?"

1. One injection has a NaN waveform → `loss = NaN`.
2. `loss.backward()` writes `NaN` into **every** parameter gradient.
3. `optimizer.step()` writes `NaN` into the **weights** (AdamW moment buffers also go NaN).
4. From then on every forward pass — on any batch, good or bad, any later epoch — is `NaN`.

The lethal sample only has to appear **once**. There is no NaN-skip / weight-reset, so the model never recovers. With `batch_size=256` and rate 1.25e-4, the expected wait is ~1 NaN per ~31 steps — consistent with the observed death at step 75–82.

## 4. Why it didn't happen before / what changed

The working pre-merger runs (val/mse ≈ 0.03–0.045, **2026-05-25/26**) trained to **Epoch 75 with zero NaN** (`grep -c nan output.log = 0`). At a 1.25e-4 NaN rate that is impossible unless their prior **excluded q≈1**. Those runs:
- used **O3b** background (launched via `regression_s4d.slurm`), and
- used the same config *path* `configs/ai4gw/chirp_mass_1s_d64_s64_l4.yaml`, but that file is **untracked / git-ignored and has been edited since** — the live version now samples `mass_ratio` all the way to `1.0` with precessing `IMRPhenomPv2`.

So the regression was introduced by **editing the (untracked) config's `mass_ratio` prior up to 1.0** (and/or switching to the precessing approximant), not by any committed-code change.

For completeness, the only committed-code change since the last-known-good commit `cf0a327` (`7c795ad`, May 29) was the spread-penalty formula in `model/regression.py` (`softplus(...)` → `(var_pred−var_target)²`). This is **not** the NaN cause — the NaN is purely data-side and appears before the model runs; `gradient_clip_val=1.0` changed nothing. Config deltas vs. the working runs (`sample_rate` 512→256, `batch_size` 128→256, `lambda_spread`) are also not the cause; `batch_size` only changes *how fast* the lethal sample is drawn.

## 5. The fix

`projects/train/configs/ai4gw/chirp_mass_1s_d64_s64_l4.yaml` — cap the mass-ratio prior below the singularity:

```yaml
mass_ratio:
  class_path: torch.distributions.Uniform
  init_args:
    low: 0.4
    high: 0.99      # was 1.0
    validate_args: false
```

Rationale:
- The singularity is at `q > ~0.999`; `0.99` sits comfortably below it with margin against marginal/edge-case waveforms.
- It removes only the upper **1.64%** of mass-ratio space (all near-equal-mass).
- The model regresses **chirp mass**, which is sampled independently; capping `q` leaves the chirp-mass target distribution essentially unchanged. Physics impact on the task: negligible.
- **No committed code is touched** (the config is untracked).

## 6. Verification

**Before fix** (`mass_ratio` up to 1.0), `debug_nan.py` on gpu_test:
- NaN at **step 75**; with `gradient_clip_val=1.0`, NaN at the **identical step 75** (byte-identical trajectory) → confirms not gradient-driven.

**After fix** (`mass_ratio: high: 0.99`), `debug_nan.py --steps 800` on gpu_test (`slurm/verify_fix.slurm`, job 18102413):
- `RESULT: completed 800 steps with FINITE loss (no NaN).`
- That is a full epoch + ~160k injections (where the unfixed config reliably died by step ~80).
- Loss **decreased 1.85 → 0.73** as the warmup schedule ramped (epoch-interval warmup ⇒ slow epoch 0, then it falls) — i.e. the model both stops NaN-ing **and** learns.
- `n_zero=n_tiny=n_nan=0` for intrinsic SNR every step; `Xfinite=True` every step.

**End-to-end** real training (`slurm/chirp_mass_59-60s_d64_s64_l4.slurm`, fresh wandb run `…_id9`, job 18106191): launched with the fixed config to confirm the full production path (online logging, checkpointing, validation, full callback stack).
- Cleared the step-75/82 death point and trained cleanly. `train/gaussnll` **decreased every epoch**: `1.45 → 0.66 → 0.39 → −0.07` (the broken `id8` showed `train/gaussnll = nan.0` here).
- **First validation finite:** `val/gaussnll = −0.307` (negative ⇒ well-calibrated predicted variance). The broken run showed `val/gaussnll = nan.0` / `val/mse = nan`.
- **Zero** `EarlyStopping: nan is not finite` signals (the broken run emitted one at every validation) ⇒ the monitored `val/mse/out_0` is finite too.
- Validation draws its fixed waveform set from the **same** (now-capped) prior, so it is protected too.
- Job cancelled after the first validation to free the gpu_test node; launch the full run on your usual partition (e.g. gpu_h200 via `regression_s4d.slurm`) with the fixed config.

Conclusion: the NaN is fully resolved by the one-line prior cap; the model trains **and** validates with finite, improving metrics.

## 7. Secondary observations (not the NaN, but worth a look)

- **Large whitened inputs.** Even on finite steps, `max|X|` of the whitened batch reached ~300–1500 (expected O(1–10)). Background files are clean, so this is whitening of real O3 noise transients / PSD mismatch. The model's per-sample `X/X.std()` normalization keeps it finite, but very large outliers are not ideal. Consider clamping/▒winsorizing whitened inputs if training is noisy.
- **`f_ref` mismatch.** `train/conversion.py::precessing_to_lalsimulation_parameters` hard-codes `f_ref = 40` (there is a `TODO` about it) while the config sets `f_ref: 50.0`. Harmless for the NaN, but the spins are converted at a different reference frequency than the waveform is generated — worth reconciling.
- **No NaN guard in the injection pipeline.** A single NaN sample is unrecoverable. If you want defense-in-depth beyond capping the prior, a guard in `_inject_from_generator` that drops/replaces non-finite waveforms would make training robust to *any* future approximant edge case. (This would modify committed code, so it is left as a recommendation, not applied.)

## 8. Reproduction / diagnostic artifacts (untracked)

| file | purpose |
|---|---|
| `projects/train/debug_nan.py` | instrumented real-`fit` loop: per-step loss, grad-norm, intrinsic-SNR, max\|X\|; stops at first NaN. Also tests `--clip`. |
| `projects/train/debug_stage.py` | monkeypatches each pipeline stage to find the first non-finite one. |
| `projects/train/debug_waveform.py` | `mass_ratio` sweep + Monte-Carlo NaN-rate characterization of the generator. |
| `projects/train/debug_background.py` | CPU sanity check of the O3a background HDF5 files (found clean). |
| `slurm/debug_nan.slurm`, `slurm/debug_stage.slurm`, `slurm/debug_waveform.slurm`, `slurm/verify_fix.slurm` | gpu_test submission wrappers. |

### Key diagnostic evidence (gpu_test, A100 MIG 3g.20gb)

- `debug_nan.py` (current config): NaN at **step 75**; with `gradient_clip_val=1.0`: **identical** NaN at step 75 ⇒ not gradient-driven. `n_zero=n_tiny=0` for intrinsic SNR ⇒ not the `target/snr` division.
- `debug_stage.py`: first non-finite stage = **`waveform_generator`**, sample `mass_ratio=0.99966`.
- `debug_waveform.py`: NaN iff `q ≳ 0.999`; MC rate 1.25e-4, all NaNs at `q>0.999`.
- `debug_background.py`: all 24 O3a files `nan=0 inf=0 zerofrac=0`, no gaps.
