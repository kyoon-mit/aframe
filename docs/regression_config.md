# Regression Training Config Reference

This document explains every field in the `configs/ai4gw/chirp_mass_*.yaml` files
used to train BNS chirp-mass regression models.

---

## Window Geometry

Understanding how the model sees the gravitational-wave data is the most
important concept before touching any config value.

### Relevant parameters

> **Two different `right_pad` parameters exist** — one on the waveform generator,
> one on the data loader. They operate at completely different stages and should
> not be confused.

| Parameter | Location | Default | Effect |
|---|---|---|---|
| `waveform_sampler.duration` | CBCGenerator | — | Total length (s) of the generated waveform file |
| `waveform_sampler.right_pad` *(waveform-level)* | CBCGenerator | 0.0 | **Sets coalescence time.** Appends this many seconds of silence after the merger in the waveform file. `coal_time = duration − right_pad`. Has nothing to do with what the model sees — it just positions the merger within the file. |
| `waveform_sampler.window_offset` | CBCGenerator | 0.0 | Shifts the windowing anchor backwards from coalescence. `anchor = coal_time − window_offset`. Use this to look at the inspiral before the merger. |
| `kernel_length` | data | — | Width (s) of the whitened window the model receives |
| `left_pad` *(window-level)* | data | 0.0 | **Controls jitter.** The anchor must sit ≥ `left_pad` s from the left edge of the model window. `jitter = kernel_length − left_pad − right_pad`. |
| `right_pad` *(window-level)* | data | 0.0 | **Shifts window rightward, and reduces jitter.** Functionally similar to decreasing `window_offset` by the same amount — both move the window later relative to coalescence. The difference: `data.right_pad` also shrinks the jitter budget (`jitter = kernel_length − left_pad − right_pad`), while `window_offset` does not. Prefer `window_offset` for positioning; leave this at 0. |
| `fduration` | data | — | Whitening filter length (s); consumes `fduration/2` samples from each edge |
| `sample_rate` | data | — | Output sample rate (Hz) after resampling |

### Exact formulas

```
# Physical coalescence time in the waveform
coal_time = waveform_sampler.duration − waveform_sampler.right_pad

# Windowing anchor (the sample that slice_waveforms centres around)
anchor_time = coal_time − waveform_sampler.window_offset

# Whitened window seen by the model (with jitter j ∈ [0, jitter_max])
window_start = anchor_time − kernel_length + data.right_pad + j
window_end   = anchor_time + data.right_pad + j

# Maximum jitter (seconds)
jitter_max = kernel_length − left_pad − right_pad − 0   (≈ left_pad slack)
```

In practice with `data.right_pad = 0`:

```
window = [anchor_time − kernel_length + j,  anchor_time + j]
       = [coal − window_offset − kernel_length + j,
          coal − window_offset + j]
```

### Worked example: `chirp_mass_4s_d64_s64_l4.yaml`

```
duration        = 64 s
right_pad       = 1.0 s  →  coal at duration − right_pad = 64 − 1 = 63 s
window_offset   = 2.0 s  →  anchor at coal − window_offset = 63 − 2 = 61 s
kernel_length   = 2.0 s
left_pad        = 1.5 s  →  jitter_max = 2.0 − 1.5 = 0.5 s
data.right_pad  = 0.0 s

Model sees: [61 − 2 + j, 61 + j]  =  [59 + j, 61 + j]  s
         →  [59.0, 61.0]  to  [59.5, 61.5]  s  (coal is 2–4 s away)
```

### How to change the window

**Set coalescence time** — there is no single `coalescence_time` parameter.
It is implicitly `coal_time = duration − right_pad`. To place coal at time T in a
waveform of length D, set `right_pad = D − T`. Example: coal at 63 s in 64 s
waveform → `right_pad = 1.0`. `right_pad` also serves as the post-merger silence
buffer needed for whitening (keep ≥ 0.5 s).

**Move the window earlier/later relative to coal** — change `window_offset`.
Larger `window_offset` pushes the window further into the inspiral.

**Widen/narrow the window** — change `kernel_length`. This is the model input length.

**Change jitter amount** — change `left_pad`. The jitter budget is
`kernel_length − left_pad − right_pad`. Set `left_pad = kernel_length − right_pad`
for zero jitter (exact, but this breaks `sample_kernels`; leave ≥ 1 sample slack).

**Change waveform length** — change `duration`. Longer waveforms contain more
inspiral, which can help IMRPhenomPv2 produce accurate low-frequency content.
Keep `duration ≥ coal_time + right_pad` (i.e. `duration ≥ coal + 1 s` minimum).

**Constraint**: `kernel_length + fduration > left_pad + right_pad + fduration`
simplifies to `kernel_length > left_pad + right_pad`. Violating this raises
`ValueError: Kernel size cannot be less than total padding`.

---

## Data Section

```yaml
data:
  class_path: train.data.regression.RegressionTimeDomainDataset
  init_args:
```

### Background / input data

| Field | Description |
|---|---|
| `background_dir` | Path to directory containing background HDF5 files. Can be flat (files directly inside) or aframe layout (`background/` subfolder). |
| `ifos` | List of interferometers, e.g. `[H1, L1]`. |
| `sample_rate` | Target sample rate (Hz). Background is resampled from its native rate. |

### Preprocessing

| Field | Description |
|---|---|
| `kernel_length` | Duration (s) of the whitened input window the model receives. |
| `fduration` | Length (s) of the time-domain whitening filter. Consumes `fduration/2` seconds from each edge of the raw window. |
| `psd_length` | Seconds of background used to estimate the PSD for whitening. Longer = more accurate PSD but more I/O per batch. 20–64 s typical. |
| `fftlength` | FFT length (s) for Welch PSD estimation. Usually 2 s. |
| `highpass` | High-pass corner frequency (Hz). Removes low-frequency noise. |
| `lowpass` | Low-pass corner frequency (Hz). `null` = no low-pass. |
| `left_pad` | **Margin constraint, not zero-padding.** The windowing anchor must sit at least `left_pad` seconds from the **left** (earlier) edge of the model's whitened window. Increasing it guarantees more pre-anchor content but shrinks the jitter budget. `jitter = kernel_length − left_pad − right_pad`. |
| `right_pad` | **Margin constraint, not zero-padding.** The anchor must sit at least `right_pad` seconds from the **right** (later) edge of the whitened window. A non-zero value shifts the window forward in time so the model sees `right_pad` seconds of post-anchor content. Use 0 for inspiral-only windows (anchor = trailing edge). |

### Training loop

| Field | Description |
|---|---|
| `batch_size` | Samples per training batch. |
| `batches_per_epoch` | Training batches per epoch. Total training samples = `batch_size × batches_per_epoch`. |
| `waveform_prob` | Fraction of batch samples that contain an injected signal (1.0 = all signal, 0.0 = pure noise). |
| `max_num_workers` | Maximum CPU worker processes for data loading. More workers → less dataloader stalling. Match to `--cpus-per-task` in SLURM. |
| `prefetch_factor` | Batches to prefetch per worker. Higher reduces stalls at cost of memory. |
| `persistent_workers` | Keep worker processes alive between epochs (avoids re-init overhead). Usually `true`. |
| `target_parameters` | Which physical parameters to regress. E.g. `[chirp_mass]`. |
| `n_val_waveforms` | Number of waveforms used for validation. |

### Extrinsic sky parameters

These are sampled independently of the waveform prior:

| Field | Description |
|---|---|
| `dec` | Declination distribution. `ml4gw.distributions.Cosine` = isotropic. |
| `psi` | Polarization angle. Uniform [0, π]. |
| `phi` | Right ascension. Uniform [0, 2π]. |

### SNR curriculum (`snr_sampler`)

```yaml
snr_sampler:
  class_path: train.augmentations.SnrSampler
  init_args:
    max_min_snr: 30.0    # starting minimum SNR (high = easy signals)
    min_min_snr: 4.0     # ending minimum SNR (low = hard signals)
    max_snr: 50.0        # maximum SNR cap
    alpha: -3.0          # power-law exponent (negative = more low-SNR; alpha=0 illegal)
    decay_steps: 100000  # training steps over which min SNR decays from max to min
```

`SnrSampler` implements curriculum learning: the minimum SNR starts at
`max_min_snr` and linearly decays to `min_min_snr` over `decay_steps` optimizer
steps. This helps the network first learn on bright signals then generalize.

**Important**: `alpha = 0` is not supported by `ml4gw.PowerLaw` (it equals a
Uniform distribution and raises an error). Use `alpha = -0.01` for a near-flat
distribution.

---

## Waveform Sampler (`waveform_sampler`)

```yaml
waveform_sampler:
  class_path: train.data.waveforms.generator.cbc.CBCGenerator
  init_args:
```

| Field | Description |
|---|---|
| `ifos` | Detector list; must match `data.ifos`. |
| `sample_rate` | Sample rate for waveform generation. Usually matches or is lower than `data.sample_rate`. |
| `approximant` | Waveform model class path, e.g. `ml4gw.waveforms.IMRPhenomPv2`. |
| `duration` | Total waveform length (s). Should be long enough that `f_min` content is generated (BNS at 20 Hz needs ~60 s). |
| `right_pad` | Seconds of silence appended after coalescence. Coal time = `duration − right_pad`. |
| `window_offset` | Additional seconds to shift signal_idx before coal for windowing (see Window Geometry). |
| `f_min` | Minimum frequency (Hz) at which waveform content is generated. Typical: 20 Hz. |
| `f_ref` | Reference frequency (Hz) for spin/phase definitions. Typical: 50 Hz. |
| `val_waveform_file` | Path to an HDF5 file of pre-generated validation waveforms. `null` = generate on-the-fly. |

### Intrinsic prior (`training_prior`)

Each parameter under `priors` is a `torch.distributions` or `ml4gw.distributions`
object:

| Parameter | Typical distribution | Notes |
|---|---|---|
| `chirp_mass` | `Uniform(0.87, 2.22)` | BNS range in M☉ |
| `mass_ratio` | `Uniform(0.4, 1.0)` | q = m2/m1 ≤ 1 |
| `a_1`, `a_2` | `Uniform(0, 0.4)` | Spin magnitudes |
| `tilt_1`, `tilt_2` | `Sine(0, π)` | Isotropic spin tilts |
| `phi_12`, `phi_jl` | `Uniform(0, 2π)` | Spin azimuthal angles |
| `inclination` | `Sine(0, π)` | Orbital inclination |
| `phic` | `Uniform(0, 2π)` | Coalescence phase |
| `distance` | `UniformComovingVolume(10, 500)` | Luminosity distance in Mpc |

---

## Model Section

```yaml
model:
  class_path: train.model.regression.LitS4DGaussianNLL
  init_args:
```

### Architecture

| Field | Description |
|---|---|
| `d_input` | Number of input channels (= number of IFOs, typically 2). |
| `d_output` | Output size = 2 × number of regression targets (mean + variance per target). |
| `d_model` | S4D hidden state dimension. Larger = more capacity, more memory. |
| `d_state` | S4D state size per layer. |
| `n_layers` | Number of S4D blocks. |
| `dropout` | Dropout probability applied between layers. |

### Normalization

| Field | Description |
|---|---|
| `normalize_input` | If `true`, each input channel is divided by its per-sample std before the model. Helps with varying signal amplitudes. |
| `y_mean` | List of per-target means used to normalize training labels. Center of the prior. |
| `y_std` | List of per-target stds used to normalize training labels. Roughly the prior std. |

The model trains in normalized label space and automatically un-normalizes
outputs at inference. Set `y_mean` and `y_std` to match the center and width of
your `chirp_mass` prior:

```
y_mean = (low + high) / 2
y_std  ≈ (high - low) / sqrt(12)   # std of uniform distribution
```

For `Uniform(0.87, 2.22)`: `y_mean = 1.545`, `y_std ≈ 0.387`.
(The config uses `y_mean=1.2, y_std=0.35` which is slightly off-center but fine.)

### Loss

| Field | Description |
|---|---|
| `beta_nll` | Weight on the NLL loss term. Combined loss = `beta_nll × NLL + (1 − beta_nll) × MSE`. |
| `lambda_spread` | Penalty weight on predicted variance to discourage overconfident or underconfident estimates. |

### Learning rate schedule

| Field | Description |
|---|---|
| `lr` | Initial peak learning rate. |
| `base_lr` | S4D-specific learning rate for state-space parameters (often lower than `lr`). |
| `weight_decay` | L2 regularization. |
| `lr_scheduler` | `WarmupCosineAnnealingWarmRestarts`: cosine restarts with linear warmup. |
| `warmup_epochs` | Epochs to linearly ramp LR from `warmup_start_factor × lr` to `lr`. |
| `T_0` | Epochs for the first cosine restart cycle. Doubles each restart with `T_mult`. |
| `T_mult` | Multiplier for cycle length after each restart. |
| `eta_min` | Minimum LR at trough of each cosine cycle. |

---

## Trainer Section

| Field | Description |
|---|---|
| `max_epochs` | Hard training epoch limit. |
| `min_epochs` | Minimum epochs before early stopping can trigger. |
| `check_val_every_n_epoch` | Validation frequency. |
| `precision` | Mixed precision: `null` = float32, `"16-mixed"` = AMP. |

### Early stopping

```yaml
- class_path: lightning.pytorch.callbacks.EarlyStopping
  init_args:
    monitor: val/mse/out_0   # metric to watch
    patience: 20             # epochs without improvement before stopping
```

### Model checkpointing

Two checkpoints are saved:
- Best `val/gaussnll` (primary quality metric)
- Best `val/mse/out_0` (best raw MSE on the first output variable)

---

## Quick-start: common changes

**Change target window** (e.g. 2 s ending 3 s before merger):
```yaml
data:
  kernel_length: 2.0
  left_pad: 1.5      # 0.5 s jitter
  right_pad: 0.0

waveform_sampler:
  window_offset: 3.0  # anchor 3 s before coal
  right_pad: 1.0      # 1 s post-coal buffer
  duration: 64.0      # coal at 63 s
```

**Change model size** (smaller / faster):
```yaml
d_model: 32
d_state: 32
n_layers: 2
```

**Use higher sample rate** (captures higher frequencies):
```yaml
data:
  sample_rate: 512   # also update waveform_sampler.sample_rate if needed
```

**Near-flat SNR distribution** (instead of power law):
```yaml
snr_sampler:
  alpha: -0.01   # NOTE: alpha=0.0 is forbidden by ml4gw.PowerLaw
```
