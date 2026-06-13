# How the Sensitive-Volume pipeline works (a from-scratch explainer)

This is the conceptual companion to [`sensitive_volume.md`](sensitive_volume.md) (which is
the command cheat-sheet). Read this if the SV pipeline feels like a black box. It assumes
you know what a BNS chirp is and roughly what aframe does, but nothing about the SV machinery.

---

## 1. The one question we are answering

**"How much of the universe can this model see BNS mergers in, before it starts crying wolf too often?"**

That number is the **sensitive volume (SV)**, in Gpc³. Bigger = better detector. We always
quote it *at a fixed false-alarm rate*, because any detector can "see" everything if you let
it raise an alarm constantly. The whole pipeline exists to produce one curve: **SV vs false
alarm rate**, which we then overlay against the real LVK pipelines (GstLAL, PyCBC, MBTA) to
see if our model is competitive.

---

## 2. The trick: two populations

Everything rests on comparing two sets of data run through the model:

- **Background** = real O3a detector noise with **no real signals**. Tells us how loud the
  model rates *pure noise*.
- **Foreground** = the same noise with **fake BNS signals ("injections") added** at known
  times. Tells us how loud the model rates *real signals*.

If signals reliably score higher than noise, the model is a good detector. SV is just a
physically-calibrated way of saying *how much* higher, translated into "volume of universe."

### Where does so much background come from? — time slides
We only have ~1.6 days of clean O3a data, but we need *years* of background to measure rare
false alarms. The trick: **shift one detector's data relative to the other by a few seconds**
("time slides" / "shifts"). A real signal shows up in H1 and L1 at almost the same instant, so
shifting destroys it — but the *noise statistics* are untouched. Each shift is a fresh copy of
noise-only data. `shifts: [[0,1],[0,2],[0,3],[0,4]]` = 4 shifts × 1.6 days ≈ **6.6 days** of
background ("~1 week"). The total background livetime is called **Tb**, and it sets the
smallest false-alarm rate you can even measure: **FAR floor = 1/Tb**.

### What are injections?
A bank of simulated BNS waveforms drawn from an astrophysical population (the
`end_o3_ratesandpops_bns` prior — masses ~1–2.5 M☉, out to redshift 0.15), already projected
onto H1/L1, with their **true parameters stored**. The pipeline adds these into the real data
and checks how well the model recovers them. Most are far away and faint (median SNR ~3) — that
is realistic and expected; only the nearby loud ones should be recoverable.

---

## 3. The "detection statistic" — and why ours is unusual

A detection statistic is **one number per candidate that says "how signal-like is this?"**,
where **higher = more signal-like** by convention.

For a normal aframe *classifier*, that's just the network's output score. **But your model is
a regressor, not a classifier.** It doesn't output "signal: yes/no" — it predicts the **chirp
mass and an uncertainty σ** on that prediction.

So we turn it into a detector with one idea:

> When a real signal is in the window, the model can actually pin down the chirp mass → **small
> σ**. On pure noise it has no idea → σ ≈ the width of the prior. **Small σ = confident =
> signal-like.**

To match the "higher = better" convention we use **detection statistic = −σ**. (This sign flip
is small but it is bug #1 below — get it wrong and the whole thing reads backwards.)

---

## 4. The pipeline end to end

### Step 1 — Inference (`projects/train/train/regression_infer.py`)
Slides the model across the data and turns strain into events:

1. **Slide a window** across each data segment, stepping `inference_sampling_rate` times per
   second (8 Hz → a step every 0.125 s).
2. For each window: estimate the noise PSD from the first `psd_length` seconds, **whiten** the
   last `kernel_length` seconds, run the model → predicted chirp mass + σ → statistic **−σ**.
   This is done identically for background and for foreground (same data + injections).
   *(This whitening must match training exactly, or the model sees out-of-distribution input.)*
3. **Post-process** the resulting −σ time series into discrete **events**:
   - **integrate**: boxcar-average the time series. Keep this ON (~1 s): the signal response is
     a wide ~2 s plateau, so the boxcar crushes isolated over-confident *noise* spikes while
     leaving the signal intact (see bug #2).
   - **cluster**: keep local **maxima** so one loud moment becomes one event, not 50.
4. Outputs:
   - `background.hdf5` — every noise event, across all shifts (+ `Tb`).
   - `foreground.hdf5` — each injection **matched ("recovered")** to a nearby event, carrying
     that event's statistic *and* the injection's true parameters.
   - `rejected_params.hdf5` — injections that were generated but fell outside the analyzed
     data (needed so the SV normalization knows the true total number injected).

### Step 2 — SV + plot (`projects/plots/plots/matplotlib/main.py`)
1. **Build the FAR ↔ threshold map** from the background: sort all background statistics; the
   k-th loudest corresponds to FAR = k / Tb. (Loud threshold → rare → low FAR.)
2. For each FAR, **count the injections recovered above that threshold**, weight them by the
   astrophysical population, and multiply by the volume the injections sample → **sensitive
   volume at that FAR**.
3. **Overlay the LVK pipelines** (GstLAL/PyCBC/MBTA/cWB) computed from the public GWTC-3
   injection campaign, auto-downloaded from Zenodo and cached in `~/.aframe/cache`.

That's the `sensitive_volume.png` you get out.

---

## 5. Reading the plot

- x-axis **false alarm rate** (log), y-axis **sensitive volume [Gpc³]**. Higher curve = better.
- For BNS (1.4, 1.4) the meaningful baselines are **gstlal, mbta, pycbc_hyperbank** (~0.01 Gpc³).
  **pycbc_bbh and cwb sitting at ~0 is normal** — those configs only search BBH, not BNS.
- If *your* curve is flat at 0, that's the model/pipeline; if *every* curve including the LVK
  ones is 0, the plotting/benchmark path is broken (it isn't — we verified it works).

---

## 6. What actually went wrong here (and the fixes)

The model trained fine, but SV came out 0. It was **four pipeline issues stacked**, none of
them the model:

| # | Bug | Why it zeroed SV | Fix |
|---|-----|------------------|-----|
| 1 | **Polarity** | Clustering keeps *maxima*, but signals are *minimum* σ → clustering deleted every signal | statistic = **−σ** before clustering |
| 2 | **Recovery offset (the big one)** | The pipeline timestamps an event at the analysis-window *start*, which is `psd+fd+kernel` (**25–69 s**) *before* the kernel — so a confident trigger lands tens of seconds before the injection and recovery never found it | `window_offset ≈ psd+fd+kernel+iwl` (≈25 merger/id11, ≈69 snr5_50 with psd=64) |
| 3 | **Recovery picks wrong event** | Clustering makes an event every ~0.25 s; "closest in time" grabbed a noise sample next to the real trigger | `recovery_mode: window` → most-confident event within ~5 s |
| 4 | **Integration** | The signal is a wide ~2 s plateau; with integration OFF, isolated over-confident *noise* spikes (σ≈0.08) beat the loudest injections → SV=0 at low FAR | keep `integration_window_length=1.0` ON — crushes narrow noise spikes, preserves the broad signal |

(Plus a crash fix: empty/too-short segments no longer kill the job.)

### How we proved the model is good
`projects/train/verify_model_response.py` loads the checkpoint, drops **loud injections** into
real background, aligns the window the way training did, and runs the exact inference
preprocessing. Result: the model predicts the **correct chirp mass within a few %** with
**σ ≈ 0.01–0.03** (the prior width is 0.39), while pure background gives σ ≈ 0.4. So the model
clearly responds to signals — the sliding/clustering/recovery stages were throwing that
response away. Always run this check before blaming a model for SV = 0.

---

## 7. Running it (short version)

```bash
# 1. inference  (writes background/foreground/rejected to outdir)
cd projects/train
uv run python -m train.regression_infer --config configs/regression_infer_merger_4s_1wk_fix.yaml

# 2. SV + plot  (writes sensitive_volume.{h5,png})
cd projects/plots
uv run python -m plots.matplotlib.main \
  --background <outdir>/background.hdf5 \
  --foreground <outdir>/foreground.hdf5 \
  --rejected_params runs/regression_sv/bns_injections/rejected-parameters.hdf5 \
  --ifos "[H1, L1]" --mass_combos "[[1.4, 1.4]]" \
  --source_prior priors.priors.end_o3_ratesandpops_bns \
  --output_dir <outdir>/sv/ --max_far 1000 --sigma 0.1 --verbose true
```

Key knobs (must match training for the geometry ones): `sample_rate`, `kernel_length`,
`fduration`, `psd_length`, `highpass`; and the SV-specific ones above
(`integration_window_length`, `window_offset`, `recovery_mode`, `recovery_window`, `shifts`).

---

## 8. File map

| File | Role |
|------|------|
| `projects/train/train/regression_infer.py` | Step 1 — slide model, cluster, recover → bg/fg HDF5 |
| `projects/plots/plots/matplotlib/main.py` | Step 2 — SV vs FAR + overlay LVK pipelines |
| `projects/plots/plots/legacy/gwtc3.py` | LVK benchmark SV from the Zenodo injection campaign |
| `projects/train/verify_model_response.py` | Sanity check: does the model itself respond to signals? |
| `projects/train/configs/regression_infer_*_fix.yaml` | The corrected run configs |
| `docs/sensitive_volume.md` | The command cheat-sheet |
