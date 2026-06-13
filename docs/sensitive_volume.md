# Sensitive Volume Calculation for Regression Models

Evaluates a trained S4D/LinOSS regression model by plotting sensitive volume vs FAR,
compared against GstLAL, PyCBC, and the standard aframe matched filter pipeline.

**Detection statistic:** predicted chirp_mass sigma (negated — lower sigma = more confident detection).

---

## Prerequisites

| Item | Description |
|---|---|
| `checkpoint.ckpt` | Trained model checkpoint (`LitS4DGaussianNLL` or `LitLinOSSGaussianNLL`) |
| `background_dir/` | Directory of HDF5 background strain segments (one file per GPS segment) |
| `injection_set.hdf5` | `InterferometerResponseSet` with pre-generated waveforms and parameters |

---

## Step 0a — Generate the Injection Set

The inference pipeline needs an `InterferometerResponseSet` — GPS-timestamped BNS
waveforms projected onto H1/L1 that can be injected into real O3a background strain.
This is **not** the BNSReg training HDF5; it must be generated once.

```bash
cd projects/train

uv run python generate_bns_injection_set.py \
    --background_dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3a_H1_L1_4096Hz \
    --output_dir /path/to/bns_injections/
```

Outputs:
- `bns_injections/waveforms.hdf5` — the `InterferometerResponseSet` (use as `injection_set_fname`)
- `bns_injections/rejected-parameters.hdf5` — injections rejected by SNR cut (use as `rejected_params`)

Uses the `end_o3_ratesandpops_bns` prior (mass 1–2.5 M☉, redshift 0–0.15, spin 0–0.4),
covering the full O3a GPS range with one injection every 64 s (~1 000 total injections over 1.6 days).

---

## Step 0b (optional) — Convert a BNSReg Checkpoint

BNSReg checkpoints use a different Lightning module class and `torch.compile` key naming.
Convert them first before running inference:

```bash
cd projects/train

uv run python convert_bnsreg_checkpoint.py \
    --input  /path/to/bnsreg.ckpt \
    --output /path/to/converted_aframe.ckpt
```

The script reads `model_cfg` from the BNSReg checkpoint, constructs a matching
`LitS4DGaussianNLL`, strips the `_orig_mod.` prefix from state dict keys, and saves
a Lightning-compatible checkpoint. Use the output path as `checkpoint` in Step 1.

---

## Step 1 — Run Inference

Produces `background.hdf5`, `foreground.hdf5`, and `rejected_params.hdf5`.

```bash
cd projects/train

uv run python -m train.regression_infer \
    --checkpoint /path/to/checkpoint.ckpt \
    --model_class LitS4DGaussianNLL \
    --background_dir /path/to/background/ \
    --injection_set_fname /path/to/injection_set.hdf5 \
    --ifos "[H1, L1]" \
    --shifts "[[0,1],[0,2],[0,3],[0,5],[0,7],[0,11],[0,13]]" \
    --sample_rate 2048 \
    --kernel_length 4.0 \
    --fduration 1.0 \
    --psd_length 64.0 \
    --fftlength 2.0 \
    --inference_sampling_rate 8.0 \
    --integration_window_length 1.0 \
    --cluster_window_length 0.5 \
    --highpass 20.0 \
    --outdir /path/to/results/
```

Or use a config file:

```bash
uv run python -m train.regression_infer --config regression_infer.yaml
```

Outputs written to `outdir/`:
- `background.hdf5` — background events (noise triggers from timeslides)
- `foreground.hdf5` — recovered injections matched to detected events
- `rejected_params.hdf5` — injections outside all processed segments (for SV normalization)

**More timeslide shifts = more background livetime = better FAR estimation.** Aim for
at least 1 year of background (`Tb`). Each shift adds `segment_duration` of livetime per
background file.

---

## Step 2 — Negate the Detection Statistic

The regression model outputs chirp_mass sigma (lower = more confident). The SV pipeline
expects higher = better, so negate:

```bash
python - <<'EOF'
import h5py, pathlib

outdir = pathlib.Path("/path/to/results")
for name in ["background.hdf5", "foreground.hdf5"]:
    with h5py.File(outdir / name, "r+") as f:
        ds = f["parameters/detection_statistic"]
        ds[:] = -ds[:]
    print(f"Negated {name}")
EOF
```

---

## Step 3 — Compute Sensitive Volume and Plot

Computes SV vs FAR and overlays GstLAL, PyCBC, and aframe on the same plot.

```bash
cd projects/plots

uv run python -m plots.matplotlib.main \
    --background /path/to/results/background.hdf5 \
    --foreground /path/to/results/foreground.hdf5 \
    --rejected_params /path/to/results/rejected_params.hdf5 \
    --ifos "[H1, L1]" \
    --mass_combos "[[1.4, 1.4]]" \
    --source_prior priors.priors.nonspin_bbh \
    --output_dir /path/to/results/sv/ \
    --max_far 1000 \
    --sigma 0.1
```

Outputs written to `output_dir/`:
- `sensitive_volume.h5` — SV and error vs FAR for each mass combination
- `sensitive_volume.png` — plot with GstLAL, PyCBC, and aframe overlaid

---

## Step 4 — Interpret Results

The plot shows **sensitive volume [Gpc³]** vs **false alarm rate [yr⁻¹]**.

- Higher curve = better sensitivity at a given FAR
- Error bands = Poisson uncertainty from the finite injection set
- At FAR = 1/yr, compare your model's SV to GstLAL/PyCBC/aframe

---

## Comparing Multiple Models

Run Steps 1–2 independently for each checkpoint into separate `outdir/` folders,
then run Step 3 once per model and overlay the resulting `sensitive_volume.h5` files
using a custom plot script, or run the SV plot script separately for each and
combine the PNG outputs.

---

## Tuning Parameters

| Parameter | Effect |
|---|---|
| `shifts` | More shifts → more background livetime → lower FAR floor |
| `integration_window_length` | Longer window smooths scores but reduces time resolution |
| `cluster_window_length` | Shorter = more events per unit time (more false alarms) |
| `kernel_length` | Must match training `kernel_length` |
| `inference_sampling_rate` | Higher = finer time resolution, more compute |
| `sigma` (plot step) | Width of log-normal mass reweighting; 0.1 = narrow BNS band |
