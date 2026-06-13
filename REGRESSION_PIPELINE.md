# Regression Pipeline: GW Parameter Estimation with aframe + LinOSS / S4D

## Table of Contents

1. [How the original aframe works](#1-how-the-original-aframe-works)
2. [How the regression implementation works](#2-how-the-regression-implementation-works)
3. [Data formats](#3-data-formats)
4. [Training](#4-training)
5. [Inference](#5-inference)
6. [Sensitive volume calculation](#6-sensitive-volume-calculation)
7. [Config reference](#7-config-reference)

---

## 1. How the original aframe works

### 1.1 High-level architecture

Original aframe is a **binary classifier** that answers: "does this 1-second window of whitened strain contain a CBC signal?" It outputs a single real-valued logit per window. Higher logit = more signal-like. This logit is the detection statistic that feeds the FAR/SV calculation.

### 1.2 Training data pipeline

```
HDF5 background files  (raw strain, per segment)
        │
        ▼  Hdf5TimeSeriesDataset  ──────────────────────────────
        │  yields (batch, n_ifos, sample_length × SR) windows   │
        │  = psd_length + kernel_length + fduration seconds     │
        │                                                        │
HDF5 waveform files   (cross/plus polarisations)               │
        │                                                        │
        ▼  Hdf5WaveformLoader + ChunkedWaveformDataset          │
        │  yields (chunk, 2, L_wf) polarisation tensors        │
        │                                                        │
        └─────────────────────────────────────────────────────▶ ZippedDataset
                                                                  │
                                                on_after_batch_transfer (GPU)
                                                                  │
                                           ┌──────────────────────┤
                                           │ PsdEstimator         │
                                           │  split window into   │
                                           │  PSD segment +       │
                                           │  kernel segment      │
                                           │                      │
                                           │ SignalInverter /     │
                                           │ SignalReverser       │
                                           │  random augmentation │
                                           │                      │
                                           │ WaveformProjector    │
                                           │  project hc,hp onto  │
                                           │  H1/L1 at random     │
                                           │  sky location, SNR   │
                                           │                      │
                                           │ Whiten               │
                                           │  FIR filter with     │
                                           │  estimated PSD       │
                                           │                      │
                                           └──▶ (X_whitened, y_binary)
                                                X: (B, n_ifos, L)
                                                y: (B, 1) ∈ {0,1}
```

**Key numbers** (typical BNS run):
- `sample_rate` = 2048 Hz
- `psd_length` = 64 s  → PSD estimated from 64 s of data preceding each window
- `kernel_length` = 1.0 s → model sees 2048 samples
- `fduration` = 0.5 s → whitening filter; 0.25 s cropped from each edge
- `waveform_prob` = 0.5 → 50 % of batches contain an injection

### 1.3 Model architecture

```
SupervisedArchitecture  (base class in architectures/supervised.py)
      │
      └─ forward(x: (B, n_ifos, L)) → logit: (B, 1)
         loss: BinaryCrossEntropyWithLogits
```

The concrete model (e.g. `SupervisedS4DModel`) pairs the S4D SSM backbone with
`SupervisedArchitecture`. An `AframeBase` Lightning module wraps it and adds
AUROC-based validation via timeslides.

### 1.4 Inference (Triton-based)

```
Background HDF5 files
        │
        ▼  Sequence  (infer/data.py)
        │  • reads raw strain in stride-sized chunks
        │  • applies IFO time shifts (timeslides)
        │  • injects InterferometerResponseSet waveforms at zero-lag
        │  • yields (x_bg, x_inj) of shape (n_ifos, batch×stride)
        │
        ▼  InferenceClient  (hermes/Triton)
        │  • sends chunks as streaming sequence requests
        │  • Triton model ensemble does whitening + model forward
        │  • returns scalar logit per stride → score timeseries
        │
        ▼  Postprocessor  (infer/postprocess.py)
        │  • integrate():  boxcar convolution over integration_window_length
        │  • cluster():    sliding-window local-max peak picking
        │  → EventSet (background)  +  RecoveredInjectionSet (foreground)
```

The Triton coupling exists because the **classifier is stateful/streaming**: it
processes one stride (~128 ms) at a time and the model (CNN or SSM) maintains
hidden state across strides so it can "look back" efficiently. Triton's sequence
model protocol manages the hidden state between requests.

### 1.5 Sensitive volume

```
EventSet (background)
RecoveredInjectionSet (foreground)
        │
        ▼  plots/legacy/main.py
        │
        │  FAR(threshold) = Nbg(≥threshold) / Tb   [events/yr]
        │
        │  For each mass combination:
        │    weight_i = p(m1,m2 | mass_combo) / p(m1,m2 | source_prior)
        │    SV(FAR) = Σ_i  weight_i × 1[score_i ≥ threshold(FAR)]  × V0
        │
        └─▶  sensitive_volume.hdf5  +  Bokeh plots
```

---

## 2. How the regression implementation works

### 2.1 Motivation

Instead of binary detection, we train a model to predict **intrinsic parameters**
(chirp mass `M_c`, mass ratio `q`) together with their **per-sample uncertainties**
using a heteroscedastic Gaussian NLL loss:

```
L = Σ_i  [  (ŷ_i - y_i)² / (2σ_i²)  +  log σ_i  ]
```

The predicted sigma (uncertainty) on the chirp mass is then used as a detection
statistic: a confident, physically-consistent chirp-mass estimate is a signature
of a real CBC signal.

### 2.2 Model output

```
out: (B, 2·n_vars)
     ├─ [:, :n_vars]  = predicted means   [M_c, q, ...]
     └─ [:, n_vars:]  = raw log-variance  → softplus → σ²
                                           → sqrt → σ
```

`n_vars` = number of target parameters (default 2: chirp_mass, mass_ratio).

The detection statistic is:

```
sigma_chirp = sqrt( softplus( out[:, n_vars + 0] ) )
```

### 2.3 Why sigma is the detection statistic

| Input type | Expected model behaviour | sigma value |
|---|---|---|
| Real BNS signal | Chirp-mass posterior concentrates around true value | **Low** |
| Noise | No consistent CBC pattern; model is uncertain | **High** |

In the sensitive volume calculation, you therefore want to **threshold on low
sigma** (tight posterior = confident detection). The `EventSet` convention is
"higher score = more likely signal", so:

> If you use plain sigma as the score, you need to pass `score < threshold`
> instead of `score >= threshold` when computing FAR/SV, OR negate sigma to
> restore the "higher = louder" convention. The current `plots/legacy/main.py`
> uses `>=` — adapt as needed.

### 2.4 Architecture choices

Two sequence models are supported. Both take `(B, n_ifos, L)` channels-first
input; the Lightning module handles the transpose internally.

#### S4D (Diagonal State Space Model)

```
S4Model  (libs/architectures/architectures/networks/s4d.py)

  encoder  Linear(d_input, d_model)
  ×n_layers:
    S4DKernel   diagonal SSM convolution via Vandermonde product
    S4D layer   FFT convolution + GLU output projection + postnorm residual
  mean pool over L
  decoder  Linear(d_model, d_output)

Input:  (B, d_input, L)   channels-first
Output: (B, d_output)
```

S4D uses a **frequency-domain convolution** (`torch.fft.rfft`). The kernel is
parameterised as a diagonal SSM:

```
A = diag(-exp(log_A_real) + i·π·[0,1,...,N/2-1])   (stable eigenvalues)
K[l] = Σ_n  C_n · exp(A_n · dt · l)                 (Vandermonde sum)
y = IFFT( FFT(u) · FFT(K) )                          (causal convolution)
```

`dt`, `A_imag`, `log_A_real` are learnable per channel. Each S4DKernel
parameter carries an `_optim` dict so the optimizer can use a distinct
learning rate and zero weight decay for SSM parameters.

#### LinOSS (Linear Oscillatory State Space)

```
LinOSSModel  (libs/architectures/architectures/networks/linoss.py)

  encoder  Linear(d_input, d_model)
  ×n_layers:
    LinOSSLayer  parallel scan over oscillatory SSM
  mean pool over L
  decoder  Linear(d_model, d_output)

Input:  (B, L, d_model)   sequence-first  (transposed in _prepare_input)
Output: (B, d_output)
```

LinOSS uses a **parallel prefix scan** (O(B·L·P) memory, O(log L) depth) over
oscillatory state-space equations. It is the ICLR 2025 "Linear Oscillatory
State-Space Models" architecture and handles very long sequences efficiently.

### 2.5 Training data pipeline (aframe-based)

`RegressionTimeDomainDataset` replaces the binary-label supervised dataset with
a parameter-label version, reusing all of aframe's injection machinery:

```
HDF5 background  ──▶  same Hdf5TimeSeriesDataset
HDF5 waveforms   ──▶  _WaveformParamLoader
                         reads cross/plus from waveforms/ group
                         reads mass_1, mass_2 from parameters/ group
                         computes chirp_mass, mass_ratio in numpy
                         yields (polarizations, params) per chunk
                       _ChunkedWaveformParamDataset
                         samples (pol[idx], param[idx]) from each chunk

on_after_batch_transfer → inject()
    ALL batch elements receive injections (waveform_prob effectively 1)
    same PSD estimation, signal inversion/reversal, projection, whitening
    returns (X_whitened, params, empty_z)  instead of (X, y_binary)
```

Validation uses the same pipeline over held-out background + validation
waveforms with their parameter labels loaded from the same HDF5 file.

### 2.6 Inference pipeline (direct PyTorch, no Triton)

Because the regression model is **windowed** (non-streaming), Triton's sequence
protocol is not needed. `regression_infer.py` runs direct PyTorch inference:

```
For each background segment × each shift combination:

  RegressionSequence.__iter__()
    _load_shifted()          apply IFO time shifts, trim to valid range
    inject(bg, t0)           add zero-lag waveforms for foreground
    yield (x_bg, x_inj)     batches of (B, n_ifos, sample_length×SR) windows

  score_sequence()
    PsdEstimator(x)          split psd_length / kernel window, compute PSD
    Whiten(x, psds)          FIR whitening filter
    model._prepare_input(x)  transpose to (B, L, d_input) for LinOSS
    model(x)                 → (B, 2·n_vars)
    sigma = sqrt(softplus(out[:, n_vars:]))[:, 0]  chirp_mass sigma
    → score timeseries at inference_sampling_rate

  _postprocess(score_ts)
    _integrate()   boxcar convolution
    _cluster()     sliding-window local max
    → EventSet (background) or RecoveredInjectionSet (foreground)

Merge all EventSets / RecoveredInjectionSets
Write background.hdf5 and foreground.hdf5
```

---

## 3. Data formats

### Background HDF5 (per-segment)

```
/H1     dataset (N,)  float32  attrs: dx=1/SR, x0=GPS_start
/L1     dataset (N,)  float32  attrs: dx=1/SR, x0=GPS_start
```

### Waveform HDF5 (WaveformPolarizationSet)

```
/waveforms/
    cross   (N, L_wf)
    plus    (N, L_wf)
/parameters/
    mass_1   (N,)
    mass_2   (N,)
    chirp_mass  (N,)   (or computed on the fly)
    ...
```

### Injection HDF5 (InterferometerResponseSet)

```
/waveforms/
    H1   (N, L_wf)
    L1   (N, L_wf)
/parameters/
    mass_1, mass_2, chirp_mass, ...
    injection_time   (N,)
    shift            (N, n_ifos)
```

### Output HDF5 (EventSet / RecoveredInjectionSet)

```
background.hdf5:
  /parameters/
    detection_statistic  (N_bg,)  sigma_chirp_mass values
    detection_time       (N_bg,)  GPS times
    shift                (N_bg, n_ifos)
  attrs: Tb (total background livetime in seconds)

foreground.hdf5:
  same as background PLUS all InterferometerResponseSet parameter fields
  (mass_1, mass_2, injection_time, ...)
```

---

## 4. Training

### Prerequisites

- **Background strain**: fetched by the standard aframe pipeline
  (`AFRAME_TRAIN_BACKGROUND_DIR` in `my-first-run/run.sh`). Reuse it.
- **Validation waveforms**: a pre-generated `val_waveforms.hdf5` with
  `right_pad=60` and `duration=64` (so the merger sits 60 s past the
  right edge of any 4 s kernel). Training waveforms are generated
  on-the-fly by `CBCGenerator` — no pre-generated waveform HDF5 needed.

### Configs

Two ready-made configs in `projects/train/configs/`:

| File | Model |
|---|---|
| `regression_s4d.yaml` | S4D (d_model=128, d_state=64, 4 layers) |
| `regression_linoss.yaml` | LinOSS (d_model=128, ssm_size=64, 4 layers) |

Both predict chirp mass only (`d_output=2`: mean + sigma), use a
4 s kernel ending 60 s before merger, BNS chirp-mass prior (1.0–1.8 M☉),
linear-warmup cosine-annealing LR schedule, and log to WandB.

Fill in the two commented paths before running:
```yaml
# background_dir: /path/to/train_background
# val_waveform_file: /path/to/val_waveforms.hdf5
```

### Run

```bash
cd projects/train

# interactive (pass paths on the command line)
uv run python -m train.regression_cli fit \
    --config configs/regression_s4d.yaml \
    --data.init_args.background_dir /path/to/train_background \
    --data.init_args.waveform_sampler.init_args.val_waveform_file /path/to/val_waveforms.hdf5

# SLURM (edit paths inside the script first)
sbatch --export=ARCH=s4d   slurm/regression_train.slurm
sbatch --export=ARCH=linoss slurm/regression_train.slurm
```

---

## 5. Inference

### Config

`regression_infer.yaml`:

```yaml
checkpoint: runs/my-run/checkpoints/last.ckpt
model_class: LitLinOSSGaussianNLL
background_dir: /data/test_background
injection_set_fname: /data/test_injections.hdf5
ifos: [H1, L1]

# Each entry is [shift_H1, shift_L1] in seconds.
# Zero-lag [0, 0] is not needed here — foreground is always zero-lag.
# More shift combinations = more background livetime.
shifts:
  - [0, 1]
  - [0, 2]
  - [0, 3]
  - [0, 5]
  - [0, 10]
  - [0, 20]
  - [0, 50]
  - [0, 100]

sample_rate: 2048
kernel_length: 4.0
fduration: 1.0
psd_length: 20.0
fftlength: 2.0
highpass: 32.0
inference_sampling_rate: 8.0
integration_window_length: 1.0
cluster_window_length: 0.5
batch_size: 128
device: cuda
outdir: results/
```

### Run

```bash
# interactive
cd projects/train
uv run regression-infer --config regression_infer.yaml

# SLURM batch
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=regression_infer
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
cd /path/to/aframe_linoss/projects/train
uv run regression-infer --config regression_infer.yaml
EOF
```

Outputs:
- `results/background.hdf5` — `EventSet` with `detection_statistic = sigma_chirp_mass`
- `results/foreground.hdf5` — `RecoveredInjectionSet` with same statistic

---

## 6. Sensitive volume calculation

The output files from inference plug directly into `plots/legacy/main.py`.

### Note on sigma as detection statistic

Standard aframe uses `score >= threshold` where higher score = more signal-like.
With `sigma_chirp_mass` as the score, **real signals have low sigma**, so you
need to **negate the statistic** before calling `main()`:

```python
import h5py, numpy as np
from ledger.events import EventSet, RecoveredInjectionSet

# Negate detection statistics so that lower sigma → higher "score"
def negate_stat(path):
    with h5py.File(path, "r+") as f:
        f["parameters/detection_statistic"][:] *= -1

negate_stat("results/background.hdf5")
negate_stat("results/foreground.hdf5")
```

Alternatively, adapt `plots/legacy/compute.py` to use `<=` for the mask.

### Python API

```python
from pathlib import Path
from plots.legacy.main import main
from priors.priors import log_normal_masses

main(
    background      = Path("results/background.hdf5"),
    foreground      = Path("results/foreground.hdf5"),
    rejected_params = Path("results/rejected_params.hdf5"),
    ifos            = ["H1", "L1"],
    mass_combos     = [(1.4, 1.4)],
    source_prior    = log_normal_masses(),
    output_dir      = Path("results/sv/"),
    max_far         = 365,          # events/yr
)
```

`rejected_params.hdf5` is the set of injections with no matched event — produced
by `RecoveredInjectionSet.recover()` but for the complement. If you don't have
it, you can derive it:

```python
from ledger.events import RecoveredInjectionSet
from ledger.injections import InterferometerResponseSet, waveform_class_factory

all_inj = waveform_class_factory(["H1","L1"], InterferometerResponseSet, "RS")\
          .read("test_injections.hdf5")
recovered = RecoveredInjectionSet.read("results/foreground.hdf5")
recovered_times = set(recovered.injection_time)
mask = np.array([t not in recovered_times for t in all_inj.injection_time])
rejected = all_inj[mask]
rejected.write("results/rejected_params.hdf5")
```

### SLURM batch (full pipeline in sequence)

```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=regression_sv
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00

BASE=/path/to/aframe_linoss
cd $BASE/projects/train

# Step 1: inference
uv run regression-infer --config regression_infer.yaml

# Step 2: negate sigma so higher = more signal-like
python - <<'PYEOF'
import h5py
for p in ["results/background.hdf5", "results/foreground.hdf5"]:
    with h5py.File(p, "r+") as f:
        f["parameters/detection_statistic"][:] *= -1
PYEOF

# Step 3: sensitive volume
cd $BASE/projects/plots
uv run python - <<'PYEOF'
from pathlib import Path
from plots.legacy.main import main
from priors.priors import log_normal_masses

main(
    background      = Path("../train/results/background.hdf5"),
    foreground      = Path("../train/results/foreground.hdf5"),
    rejected_params = Path("../train/results/rejected_params.hdf5"),
    ifos            = ["H1", "L1"],
    mass_combos     = [(1.4, 1.4)],
    source_prior    = log_normal_masses(),
    output_dir      = Path("../train/results/sv/"),
    max_far         = 365,
)
PYEOF
EOF
```

---

## 7. Config reference

### `RegressionTimeDomainDataset` init args

| Arg | Default | Description |
|---|---|---|
| `background_dir` | — | Directory of background HDF5 files |
| `waveforms_dir` | — | Directory of waveform HDF5 files |
| `ifos` | — | e.g. `[H1, L1]` |
| `sample_rate` | — | Hz |
| `kernel_length` | — | Length of whitened kernel fed to model (s) |
| `fduration` | — | Whitening filter duration (s); `fduration/2` cropped from each edge |
| `psd_length` | — | Length of PSD estimation window (s) |
| `fftlength` | `None` | FFT length for PSD (s); defaults to `kernel_length + fduration` |
| `highpass` | `None` | Hz |
| `waveform_prob` | `1.0` | Fraction of windows that receive an injection (use 1.0 for regression) |
| `target_parameters` | `(chirp_mass, mass_ratio)` | Parameters to predict |
| `batch_size` | — | Training batch size |
| `batches_per_epoch` | — | Number of gradient steps per epoch |

### `LitLinOSSGaussianNLL` init args

| Arg | Default | Description |
|---|---|---|
| `d_input` | — | Number of IFOs |
| `d_output` | — | `2 × n_vars`; first half = means, second half = pre-softplus variances |
| `d_model` | `64` | Hidden dimension |
| `ssm_size` | `64` | SSM state dimension (P in LinOSS notation) |
| `n_layers` | `4` | Number of LinOSS residual blocks |
| `discretization` | `IM` | Discretization method (`IM` = implicit midpoint) |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `weight_decay` | `0.0` | AdamW weight decay |

### `LitS4DGaussianNLL` init args

| Arg | Default | Description |
|---|---|---|
| `d_input` | — | Number of IFOs |
| `d_output` | — | `2 × n_vars` |
| `d_model` | `256` | Hidden dimension |
| `d_state` | `64` | SSM state dimension (N in S4D notation) |
| `n_layers` | `4` | Number of S4D residual blocks |
| `dropout` | `0.2` | DropoutNd rate |
| `dt_min` / `dt_max` | `0.001` / `0.1` | Range for initializing `Δt` parameters |
| `lr` | `None` | Override learning rate for SSM kernel parameters |
| `learning_rate` | `1e-3` | AdamW learning rate |
| `weight_decay` | `0.0` | AdamW weight decay |

### `regression-infer` CLI args

| Arg | Default | Description |
|---|---|---|
| `checkpoint` | — | Path to `.ckpt` file |
| `model_class` | — | `LitLinOSSGaussianNLL` or `LitS4DGaussianNLL` |
| `background_dir` | — | Directory of test background HDF5 files |
| `injection_set_fname` | — | Path to `InterferometerResponseSet` HDF5 file |
| `ifos` | — | e.g. `[H1, L1]` |
| `shifts` | — | List of `[shift_H1, shift_L1]` in seconds |
| `sample_rate` | — | Hz (must match training) |
| `kernel_length` | — | Seconds (must match training) |
| `fduration` | — | Seconds (must match training) |
| `psd_length` | — | Seconds (must match training) |
| `fftlength` | — | Seconds (must match training) |
| `inference_sampling_rate` | — | Output steps per second (e.g. 8) |
| `integration_window_length` | — | Boxcar integration window (s) |
| `cluster_window_length` | — | Peak-picking window (s) |
| `highpass` | `None` | Hz (must match training) |
| `batch_size` | `128` | Windows per forward pass |
| `device` | `cuda` | `cuda` or `cpu` |
| `outdir` | — | Output directory |
