# How aframe's Sensitive-Volume pipeline works, and what to change for the S4D regression model

This explains the **stock aframe SV pipeline** (the `aframe` repo) from the ground up, then
what has to change to run it with the **S4D chirp-mass regression** model. It deliberately
spells out jargon. For the *conceptual* "what is sensitive volume" background, see
[`sv_pipeline_explained.md`](sv_pipeline_explained.md); this doc is about the *machinery*.

---

## 0. Glossary (skim, refer back)

| term | plain meaning |
|---|---|
| **Sensitive volume (SV)** | how much of the universe (Gpc³) the search can detect BNS in, at a fixed false-alarm rate. Bigger = better. |
| **FAR** (false-alarm rate) | how often *noise alone* produces a candidate at least this loud, per year. |
| **detection statistic** | one number per candidate; higher = more signal-like. For the classifier it's the network score; for our regressor it's −σ (low predicted uncertainty = confident). |
| **time slide / shift** | sliding one detector's data vs the other by a few seconds to manufacture signal-free "background" while preserving noise statistics. |
| **background / foreground** | noise-only data / the same data with fake signals ("injections") added. |
| **whitening** | dividing the strain by the noise amplitude spectrum so all frequencies have ~equal noise; makes signals stand out. |
| **PSD** (power spectral density) | the noise power vs frequency, estimated from a stretch of data and used to whiten. |
| **kernel** | the fixed-length chunk of (whitened) strain the network actually looks at. |
| **law / luigi** | Python workflow tools. luigi defines "tasks" with dependencies (a DAG); law wraps luigi to run those tasks in containers/clusters. aframe's whole pipeline is a luigi DAG. |
| **Triton** | NVIDIA's *inference server*: you put models in a "model repository" and it serves them over the network (gRPC). aframe runs inference by streaming data to Triton. |
| **TorchScript** | a serialized, framework-independent form of a PyTorch model (`torch.jit`). |
| **ONNX / TensorRT** | ONNX = a portable model format; TensorRT = NVIDIA's compiler that turns a model into a highly-optimized GPU engine. aframe exports the net to TensorRT for speed. |
| **hermes / quiver / aeriel** | LIGO's wrapper libraries around Triton. `quiver` *builds* the model repository; `aeriel` is the streaming inference *client*. |
| **snapshotter** | a small *stateful* model on the Triton server that keeps a rolling buffer of recent strain, so you only send *new* samples each step and it reconstructs the overlapping kernels. Makes streaming inference cheap. |
| **ensemble** | several Triton models wired together so one request flows through them in order. aframe's ensemble = snapshotter → preprocessor(whiten) → network. |
| **state-space model (SSM) / S4 / S4D** | a sequence model that processes a timeseries with a learned linear recurrence (a "scan"), like a smart IIR filter. S4D is the diagonal variant. This is our regression network. |
| **EventSet / RecoveredInjectionSet** | aframe's HDF5 "ledger" data structures for triggers and for injections-matched-to-triggers. |

---

## 1. The stock aframe SV pipeline, end to end

aframe is orchestrated as a **luigi DAG** (each box is a task; arrows are dependencies):

```
 data  →  train  →  export  →  infer  →  plots (SV)
(fetch  (train    (model →   (stream    (SV vs FAR,
 strain, the       Triton    data thru   overlay
 make    network)  repo)     Triton →    GstLAL/
 inj.)                       triggers)   PyCBC)
```

### 1a. Export — turn the trained net into a served model (`projects/export`)
The trained network is saved as **TorchScript** and handed to `export()`. Using `hermes.quiver`
it builds a **Triton model repository** containing an **ensemble** of three models that run
back-to-back on the GPU at inference time:

1. **snapshotter** (`BackgroundSnapshotter`) — keeps the rolling buffer so streaming updates
   become overlapping kernels (see glossary).
2. **preprocessor** — a TorchScript module that estimates the **PSD** and **whitens** each
   kernel. (Same math as training.)
3. **the network** — compiled to **TensorRT** (`platform=qv.Platform.TENSORRT`).

Key export arguments: `kernel_length`, `sample_rate`, `inference_sampling_rate`, `fduration`,
`psd_length`, and **`num_outputs`** (= **1** for the classifier: a single "signal-ness" score).

### 1b. Infer — stream data through Triton (`projects/infer`)
- `Sequence` (`infer/data.py`) opens one background segment, applies the **time shifts**, and —
  if there are injections in this segment/shift — makes an injected copy with
  `injection_set.inject(...)`. Iterating it yields `(x, x_inj)` chunks of *new* samples (one
  stride × batch worth at a time), **not** whole kernels — the snapshotter rebuilds kernels.
- `infer()` (`infer/main.py`) sends each chunk to Triton via the **aeriel `InferenceClient`**
  as two streams: the **background** request `stack([x, x])` and the **foreground** request
  `stack([x, x_inj])` (the background copy travels with the injection so the *same* PSD is used
  to whiten both — fair comparison).
- The server streams back a **timeseries of detection statistics** (one value per inference
  step). `Sequence.__call__` stores `y[:, 0]` — i.e. it takes output column 0 as *the*
  detection statistic. ← remember this line; it changes for regression.

### 1c. Postprocess — timeseries → discrete events (`infer/postprocess.py`)
`Postprocessor` turns each statistic-timeseries into an `EventSet`:
1. drop the first `psd_length` worth of samples (used only to seed the PSD),
2. **integrate**: convolve with a boxcar (a running average) — designed to accumulate a
   matched-filter SNR transient,
3. **cluster**: walk the series keeping only **local maxima** within `cluster_window_length`,
   so one loud moment = one event.
4. each event's `detection_time = t0 + psd_length − fduration/2 − integration_window_length +
   i/inference_sampling_rate`. **Note this places the event near the *start* of its analysis
   window** — important later.

`background` triggers go to `background.hdf5`; for `foreground`, `Sequence.recover()` matches
each injection to the **closest-in-time** event → `foreground.hdf5` (a `RecoveredInjectionSet`:
the matched trigger's statistic + the injection's true parameters). Injections outside any
analyzed segment go to `rejected`.

### 1d. Plots / SV (`projects/plots/plots/legacy`)
`compute.py` + `main.py`: from the background, sort the statistics so the *k*-th loudest ↔
FAR = *k* / T_b (T_b = total background livetime). For each FAR, weight the injections recovered
above that threshold by the astrophysical population and multiply by the sampled volume → **SV
vs FAR**. `gwtc3.py` overlays the LVK pipelines (GstLAL/MBTA/PyCBC), whose numbers come from a
public injection campaign downloaded from Zenodo.

---

## 2. What changes for the S4D regression model

The stock pipeline assumes a **binary classifier**: one output, "high = signal", that fires
**at the merger**, served via **TensorRT on Triton**. Our model breaks three of those
assumptions — it's a **regressor**, its statistic is **inverted**, it can be **pre-merger**, and
**S4D is hard to export**. Here's each change.

### 2a. Output → detection statistic (regressor, not classifier)
The S4D net outputs **two** numbers per kernel: the **chirp-mass mean** and a **variance**. The
detection statistic is the model's *confidence* = **−σ**, where σ = √(softplus(variance)). A
real signal → the model is sure → small σ → large −σ.

- Stock infer takes `y[:, 0]` (the classifier score). That is now the *mean* — wrong.
- **Two clean options:** (i) wrap the network at export so its single output is −σ directly
  (keeps `num_outputs=1` and the rest of the pipeline unchanged), or (ii) set `num_outputs=2`
  and compute −σ in the client. Option (i) is cleaner.

### 2b. Polarity
Clustering keeps **maxima**. Signals are **minimum σ**, so you must cluster on **−σ** (not σ),
or clustering throws the signal away. (This was the first bug we hit.)

### 2c. Pre-merger recovery timing — `window_offset`
The classifier fires *at* the merger, so closest-in-time recovery works. A **pre-merger** model
fires *seconds before* coalescence. Worse, recall §1c: the assigned `detection_time` is the
analysis-**window start**, which is `psd_length + fduration + kernel_length` *before* the kernel.
Net effect: a confident trigger lands tens of seconds before the injection's coalescence time.
**Fix:** shift foreground detection times by `window_offset ≈ psd + fduration + kernel +
integration_window + (training pre-merger offset)` before matching — empirically ≈ 25 s for the
1 s/256 Hz models, ≈ 69 s for the 64 s-PSD model. And recover by the **most-confident event in a
± few-second window**, not the closest in time (clustering emits an event every ~0.25 s, so
closest-in-time grabs a noise sample next to the real trigger).

### 2d. Integration — keep it ON
For our statistic, integration is *helpful but for a different reason*: the signal response is a
**wide plateau** while spurious over-confident *noise* is **isolated spikes**, so the boxcar
averages the noise spikes down and preserves the signal. Keep `integration_window_length ≈ 1 s`.

### 2e. Input normalization
The regressor applies a per-sample standardization (`_prepare_input`: divide each kernel by its
own std) before the network, and trains on normalized chirp-mass targets. The exported
preprocessor (or the wrapped model) must reproduce this, or the net sees out-of-distribution
input.

### 2f. The hard one: exporting S4D to Triton/TensorRT
A CNN classifier compiles to TensorRT trivially. **S4D is a recurrent state-space model with a
sequential scan**, which ONNX/TensorRT support poorly. This is almost certainly why the
`aframe_linoss` fork **does not use Triton at all** for regression and instead ships a
standalone PyTorch driver, `projects/train/train/regression_infer.py`, that runs the model
directly. Your three realistic paths:
1. **Bypass Triton (done in the fork).** `regression_infer.py` reimplements snapshotting,
   whitening, scoring, integrate/cluster, and recovery in plain PyTorch. Simplest; already
   working; loses Triton's throughput.
2. **Host S4D on Triton with the PyTorch/TorchScript backend** (not TensorRT). Slower than
   TensorRT but reuses the stock infer/snapshotter/ensemble machinery.
3. **Export an ONNX-friendly S4D.** S4D has an equivalent **convolutional (FFT) form** for
   fixed-length inputs — unroll the recurrence into a long convolution, which *does* export.
   Most work, best inference speed.

---

## 3. Mapping: stock aframe ↔ the fork's `regression_infer.py`

| stock aframe (Triton) | `regression_infer.py` (PyTorch) | change for S4D regression |
|---|---|---|
| `export` → TensorRT ensemble | *(none — model run directly)* | S4D doesn't export; bypassed |
| `BackgroundSnapshotter` (server) | manual sliding windows | same idea, in torch |
| preprocessor (PSD + whiten) | `PsdEstimator` + `Whiten` per window | must match training whitening + `_prepare_input` |
| network, `y[:,0]` = score | `_score()` returns **−σ** = −√softplus(var) | regressor output → −σ statistic (§2a, 2b) |
| `Postprocessor.integrate/cluster` | `_integrate` / `_cluster` (copied) | keep integration ON (§2d) |
| `Sequence.recover` (closest-in-time) | `window_offset` shift + max-in-window recover | pre-merger timing fix (§2c) |
| `plots/legacy` SV vs FAR | same code, reused | unchanged (works as-is) |

**Bottom line:** the SV *math* and the *plots* are identical; everything that changes lives in
**how the model is run and how its output becomes a detection statistic** — the regressor's −σ
statistic, the polarity, the pre-merger recovery offset, keeping integration on, and (the real
engineering cost) that S4D can't ride the TensorRT/Triton path the classifier uses.
```
```
