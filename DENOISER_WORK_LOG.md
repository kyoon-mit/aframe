# Denoiser work log

A record of the denoiser investigation on the local RTX PRO 6000 machine,
from the first four-way loss scan through the frozen-denoiser classifier
runs. Written to be readable by someone picking this up cold, including the
mistakes, because several of them cost real time and are easy to repeat.

Branch: `local-reg-dev-latest`. Runs log to the `DENOISER-SCAN` and
`CLAUDE-TESTS` projects on Weights and Biases.

---

## Summary

The task is a denoiser that maps whitened noisy strain `(B, 2, 8192)` to the
clean injected waveform, feeding a downstream detection classifier.

Five findings, in the order they were established:

1. **The original loss was measuring the wrong thing.** Four variants of
   `ScheduledMixtureLoss` looked healthy by their loss curves but had a
   correlation with the target of 0.007, with 98 percent of the predicted
   power sitting at the leading edge of the analysis window. The magnitude
   spectrum discards phase, so the objective could not tell a chirp from an
   impulse, and an impulse has the smooth broad spectrum that had been read
   as a sign of a clean fit.

2. **Separating shape from gain fixed it.** `ShapeGainLoss` raised the
   correlation from 0.007 to 0.34, about fifty times, and dropped the
   leading-edge power fraction from 0.98 to 0.016 against a target value of
   0.036.

3. **The amplitude deficit is not a defect.** Measured gain tracks measured
   correlation, both near 0.2, which is the minimum-variance optimum
   `c* = rho`. A model that recovers a fifth of the waveform shape correctly
   emits a fifth of the amplitude. Forcing the gain to one without raising
   the correlation would only add confident structure the data does not
   support.

4. **Capacity is not the bottleneck.** A ladder of 67k, 200k and 400k
   parameters all plateaued near rho 0.21.

5. **The two loss terms are wildly out of balance, and the imbalance moves
   during a run.** The spectral gradient exceeded the time-domain gradient by
   a factor of about 2400 at the trained amplitude, so `alpha: 0.5` weighted
   the two terms evenly by value while the time term supplied roughly 0.04
   percent of the update. The ratio scales as amplitude to the minus two, so
   the SNR curriculum moves it by two orders of magnitude with no change to
   the configuration.

The current work is the fix for point 5 and a first detection statistic
built on top of a frozen denoiser.

---

## Phase 1: the four-way scan and what it hid

The starting point was `ScheduledMixtureLoss`, a convex mixture of a
time-domain mean-square error and a frequency-domain term:

```
L_time = mean over samples of (pred - target)^2
L_spec = mean over bins of (log_b(|FFT(pred)| + eps) - log_b(|FFT(target)| + eps))^2
L      = alpha * L_time + (1 - alpha) * L_spec
```

Four variants scanned `density` (whether each row is divided by its own
error against a zero prediction) against `log_floor` (`eps` above).

The visible symptom was that some runs produced a strikingly smooth
frequency-domain prediction, free of the bin-to-bin jitter the target
carries, and simultaneously about an order of magnitude too small in
amplitude. The smoothness looked like a partial success worth preserving.

Diagnostics on the trained checkpoints said otherwise:

| variant | rho | power in first 5% of window |
|---|---|---|
| density true, floor 1e-3 | 0.056 | 1.1% |
| density false, floor 1e-3 | 0.007 | 98.2% |
| target itself | - | 3.6% |

A cross-correlation scan over all lags ruled out a simple timing offset: the
peak is at zero lag, so the low correlation is real. The model was emitting a
short transient at the window edge. A smooth broad spectrum is exactly what
an impulse looks like in the frequency domain, so smoothness and smallness
were two views of one failure, not a good property beside a fixable one.

Two explanations offered along the way were tested and refuted:

- That the effect was Wiener shrinkage of a good fit. The measured
  correlation is too low for that reading.
- That whitening leaks broadband power at the window edges. Applying a Tukey
  taper moved the 99-percent power containment from 556 Hz to 559 Hz, no
  effect.

---

## Phase 2: ShapeGainLoss

The replacement supervises four properties separately:

```
L = (1 - rho) + lambda_g * (log c*)^2 + lambda_t * MSE + lambda_b * bkg
```

- `rho` is the normalised inner product, scale-free so shrinking cannot
  reduce it, and phase-sensitive so a misplaced impulse scores near zero.
- `c*` is the least-squares rescaling of the prediction onto the target, so
  the gain term is zero at the correct amplitude and indifferent to shape.
- `bkg` drives noise-only rows to zero, since shape and gain are undefined
  when the target is identically zero.

Unit tests rank the failure modes correctly: perfect 0.0, half amplitude
0.27, zeros 103, edge impulse 104. The mode the old loss preferred is now the
worst.

Results after roughly fifty epochs:

| variant | best rho | epoch | gain | early power |
|---|---|---|---|---|
| learned term weights | 0.344 | 11 | 0.27 | 0.016 |
| lambda_g 1.0 | 0.338 | 6 | 0.34 | 0.031 |
| shape dominated | 0.324 | 15 | 0.29 | 0.017 |
| lambda_g 0.3 | 0.303 | 11 | 0.40 | 0.031 |
| previous scan, best of four | 0.007 | - | - | 0.982 |

Two implementation defects surfaced here:

- `configure_optimizers` walked only the architecture's parameters, so a loss
  carrying its own parameters (the learned weights) would never have been
  updated. It looked like it worked.
- Checkpoints were selected on validation loss, the quantity already shown to
  mislead. Selection moved to `val/rho`.

### A reading error worth recording

The apparent decline in rho after epoch ten looked like divergence. It was
mostly the SNR curriculum: the sampler ramps its floor from 20 down to 4, so
the target amplitude falls by a factor of four and rho falls with it.
Dividing by the target amplitude gives a curve that is flat or rising. But
after the curriculum finishes, the normalised curve does keep falling, so
part of the decline is real degradation at SNR 4. Both statements are needed;
the first alone was wrong.

---

## Phase 3: capacity, splines, approximate recovery

Three directions were tried and mostly closed off.

**Capacity.** 67k, 200k and 400k parameters all plateaued near rho 0.21, so
model size is not the limit. The 1.19M variant ran out of memory.

**Spline output basis.** The idea was to expand the output on cubic B-splines
so that a one-sample edge impulse becomes inexpressible. The first attempt
used a knot spacing of 16 samples, which caps the representable frequency at
64 Hz while the chirp sweeps past 300 Hz, and rho collapsed to 0.007. That
was a design error, not a failure of the idea. Rebuilt at stride 4 (256 Hz)
and stride 2 (512 Hz).

**Approximate recovery.** Since exact recovery is unreachable at SNR 4, an
`ApproximateRecoveryLoss` weights the shape term by the target's local energy
envelope and divides out the realised gain before comparing shape. Its unit
tests behave as designed (a two-fold shrink costs 0.024 against a perfect
0.0), but it has not yet been run to a conclusion.

---

## Phase 4: the gradient imbalance

This is the current thread and the most useful measurement in the log.

A probe supplied on another machine predicted that the two terms' gradients
scale in opposite directions: the time term as amplitude, the log-magnitude
spectral term as one over amplitude, because differentiating a logarithm
contributes a factor of one over the magnitude. Their ratio therefore moves
as amplitude squared. The probe reproduced exactly here (78262.8 at amplitude
0.01, 0.0008 at amplitude 100, crossover near 2.6).

To measure it on real batches rather than synthetic ones,
`term_gradient_norms` differentiates each stashed loss term separately with
respect to the prediction. The prediction is the quantity the terms share and
it costs one backward pass through the loss alone, not through the network.

Measured on a real run:

| epoch | grad time | grad spectral | ratio | target rms |
|---|---|---|---|---|
| 0 | 1.04e-03 | 2.18e-02 | 21 | 0.285 |
| 39 | 1.85e-04 | 4.17e-01 | 2255 | 0.061 |
| 325 | 1.70e-04 | 4.10e-01 | 2406 | 0.067 |

So `alpha: 0.5` gives the time term about 0.04 percent of the update, and the
curriculum moves the balance point by a factor of 115 during the ramp.

`log_floor` turned out to matter for the same reason. The floor caps the
per-bin spectral gradient at one over the floor, so a smaller floor lets bins
containing no signal dominate:

| log_floor | ratio | rho |
|---|---|---|
| 1e0 | 45 | 0.096 |
| 1e-3 | 2440 | 0.044 |
| 1e-9 | 4789 | 0.029 |

Monotone, and floor 1e0 is best without any other change.

### Two ways to move alpha

`snr_tracking` predicts the ratio from the waveform amplitude using a fitted
constant, `ratio = 14 / target_rms^2`. It works in the sense that alpha
follows the curriculum, but the constant is data-dependent (18 on synthetic
data, 138 on real) and the run reached its alpha ceiling of 0.9999 while the
ratio was still 5041.

`grad_tracking` replaces the model with the measurement. The unweighted
per-term gradient norms are already logged each epoch, and their ratio is
exactly the factor alpha has to cancel:

```
alpha * g_time = (1 - alpha) * g_spec
=> alpha = ratio / (1 + ratio)
```

Nothing is calibrated in advance, and it is self-correcting because
`term_gradient_norms` differentiates the terms before alpha is applied, so
the measurement does not depend on the current alpha.

With floor 1e0 and `grad_tracking` together the ratio sits between 13 and 18
rather than 2440.

### The decay_steps bug

`decay_steps` in the SNR sampler counts optimiser steps, not epochs. When
`batches_per_epoch` was cut from 782 to 16 for a faster diagnostic, the
inherited value of 31280 silently became 1955 epochs, more than the 800 the
run allowed. The SNR floor sat at 18.4 and never approached 4.

That invalidated a claim made at the time. The run reporting rho 0.460 was
training at SNR 17.5, not at SNR 4. With `decay_steps` corrected to 640 the
same configuration reaches rho 0.089 at SNR 4. Lower number, harder task,
and the two are not comparable.

Any parameter counted in steps has to be restated whenever
`batches_per_epoch` changes. Parameters counted in epochs
(`warmup_epochs`, `T_0`, `max_epochs`, `every_n_epochs`) do not, since the
scheduler interval is `epoch`.

---

## Phase 5: detection

`DenoisedClassification` already existed, along with `TimeSlideAUROC`, whose
`max_fpr` default of 1e-3 is set in the existing configs. Two pieces were
added:

- `LoadDenoiserWeights`, which initialises the denoiser from a standalone
  `Denoiser` checkpoint. The pure denoiser task stores its modules under
  `model.model.*` while the joint architecture keeps them under
  `model.model.denoiser.*`, so the state dict is remapped by prefix. With
  `freeze` set, the parameters get `requires_grad=False` and the module is
  held in eval mode so dropout stays off.
- `ClassificationTimeDomainS4DenoiseClassifyS4`, which pairs the frozen
  denoiser with a second S4D stack instead of a convolutional ResNet. A chirp
  is a long sweep whose phase evolves across the whole window, so a
  recurrence carrying state can use evidence a fixed receptive field cannot.

The freeze is verified by `train/loss_denoise`, which stays at about 15700
with no trend across epochs. It is not perfectly constant because training
batches are resampled each epoch; stationary is the correct expectation, not
flat.

Results so far, target AUROC greater than 0.525 at FPR 1e-3:

| run | head | best AUROC |
|---|---|---|
| `den_frozen_resnet_bce_v2` | ResNet1D [2,2,2] | 0.518 |
| `den_frozen_s4_bce` | S4D 64/64/4 | 0.506 (early) |

Both are below target and both are using a weak denoiser, the `alpha_fix`
checkpoint at rho 0.15, because the better denoiser was still training.

---

## Diagnostics and infrastructure built

**Loss-independent metrics.** Loss values are not comparable between
objectives, and a falling loss accompanied a worsening reconstruction in the
first scan. Three metrics are now logged for every variant:

- `rho`, the normalised inner product on rows carrying signal. Scale-free and
  phase-sensitive, so neither a shrunken copy nor a misplaced impulse scores
  well. This is the primary figure of merit.
- `gain`, the realised amplitude ratio. One means correct amplitude.
- `early_frac`, the fraction of predicted power in the first five percent of
  the window. A direct probe of the edge-impulse failure; the target's own
  value is 0.036 and the broken runs reached 0.98.

**Per-term gradient norms.** `grad/<term>`, sampled every `log_grad_every`
batches and averaged per epoch. A single batch is far too noisy to read: the
spectral gradient norm varies by more than its own size between batches.
Terms are logged separately rather than as a ratio, because the ratio of
averages is not the average of ratios and leaving them separate keeps the
choice of summary open.

**Fixed reference events.** The evolution callback used to cache whichever
training batch it saw first, so the plotted events differed between runs and
could not be compared. `build_plot_events.py` now writes a curated file:

| # | name | SNR | note |
|---|---|---|---|
| 1 | snr50 | 50 | quiet background |
| 2 | snr20 | 20 | same waveform |
| 3 | snr4 | 4 | at threshold |
| 4 | glitch | 4 | on the loudest glitch found |
| 5 | background | 0 | noise only, zero target |
| 6 | long | 8 | 32 s, -30 to +2 s, separate figure |

The background is timeslid so no coincident astrophysical signal survives.
The file records `merger_index`, so the plotter no longer derives the time
origin from an argmax; that used to drift between rows and was undefined on a
background row, where it silently fell back to a 0-to-4 s axis with a
different meaning on the same figure.

`ReferenceEventCallback` plots that file. Note that the classifier configs
still use the older `DenoiserEvolutionCallback` and should be switched.

---

## Mistakes worth not repeating

**Re-deriving geometry instead of reading it.** The reference event window was
wrong four times running because the placement was reconstructed from the
config rather than copied from `slice_waveforms` and `build_val_batches`.
Training crops a span and random-crops the kernel within it, so the merger
lands uniformly between 3.5 and 4.0 seconds; validation view 0 pins it at
3.5. Both follow from `left_pad`, not `right_pad`, and `right_pad_size`
carries a `filter_size // 2` term that is easy to drop.

**Stacking jobs on one GPU.** Five concurrent runs held both cards at 97
percent for days, and GPU 0 subsequently fell off the PCIe bus
(`current_link_width` reading 63, an all-ones value). It needed a physical
reboot. The cards sit in the mid 80s even at idle and are two degrees from
their slowdown point under load, so three jobs is the practical limit.

**Killing the wrong process.** A `pkill -f` pattern matched the issuing shell
more than once, and a kill aimed at one job took out an unrelated run whose
dataloader workers shared the process group. Kill by process group id read
from `ps`, never by a loose pattern.

**Changing a config on a live wandb run id.** Editing then reverting a
config while the id was already registered produced a `ConfigError` and
killed the run. A changed configuration needs a new run id.

---

## Current state

Three jobs running:

| run | GPU | what |
|---|---|---|
| `no_norm_1e0_floor_alpha_track_v2` | 1 | floor 1e0 plus `grad_tracking`, `decay_steps` fixed, now genuinely at SNR 4 |
| `den_frozen_resnet_bce_v2` | 0 | frozen denoiser, ResNet head |
| `den_frozen_s4_bce` | 0 | frozen denoiser, S4D head |

Open questions:

- Whether `grad_tracking` holds the balance for a full run, and whether the
  feedback loop between alpha and the measured ratio settles.
- Whether background rows can be learned now. The earlier attempt used
  `waveform_prob 0.5` at floor 1e-3 and failed badly (rho 0.007, ratio
  62000), plausibly because a zero target pins every spectral bin at the
  floor. Floor 1e0 removes that specific blow-up, so the combination of
  `waveform_prob 0.5`, floor 1e0 and `grad_tracking` has never been tried.
- Whether a better denoiser lifts AUROC past 0.525. Both classifiers are
  currently fed the rho 0.15 checkpoint.
- Whether rho at SNR 4 can exceed the low values seen so far, or whether it
  is bounded by the signal-to-noise ratio itself. `SNRWeightedMSELoss` exists
  and is unused; it weights each row by `(ref / (snr + ref))^gamma` to shift
  capacity onto quiet rows.
