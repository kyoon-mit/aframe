# Template-bank network checks

CPU checks of `TemplateBankNet`
(`projects/train/train/experimental/template_bank/arch.py`): the χ² test
(band sums against exact sums, reduced χ² on Gaussian and on real noise, the
χ² share of clean signals, average-spectrum against own-spectrum filters,
fitting factor on injection-file waveforms), the lag and phase-offset
features, the parabola estimate, and the score that the sensitive-volume step
wrote to its "logit" tree for the detection-MLP checkpoint.

## What it backs

- [The matched filter as a network](https://kyoon-mit.github.io/projects/bns/updates/2026-10-12-template-bank-network/),
  sections "The network" and "Detection": the χ² test, the reweighted SNR
  and the detection features.
- The experiment log entries of 2026-10-12 on the χ² test and on the code
  review (local only, not in the repo).

Per script:

- `chk1.py`: max-pool indices, band sums and reduced χ² against float64
  sums, Gaussian noise
- `chk2.py`: χ² share of whitened injection-file waveforms at seven merger
  positions, two polarization mixes
- `chk3.py`: reduced χ² and E|z|² on Gaussian noise; band variance at a fixed
  template and lag
- `chk4.py`: expected χ² from the covariance of the band sums inside the
  window
- `chk5.py`: χ² share of clean and noisy cached training windows, by SNR
- `chk6.py`: χ² share of clean signals, by merger sample
- `chk7.py`: average-spectrum filters against each window's own-spectrum
  filters
- `chk8.py`: χ² share and fitting factor for up to 40 injection-file
  waveforms
- `chk9.py`: real noise rows: plain and reweighted SNR², reduced χ², band
  power
- `check_noise.py`: E|z|² against lag on Gaussian noise; reduced χ² at the
  loudest template
- `check_signal.py`: recovered SNR², lag, template chirp mass and χ² for 24
  cached clean signals
- `check_delay.py`: lag and phase-offset features for shifted and
  phase-rotated templates; parabola vertex
- `check_logit_tree.py`: the "logit" score written by
  `dev/infer/steps/step1_local_bank.py` against the model's own detection
  output, detection-MLP checkpoint `ck_15`

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: not recorded and not needed; the scripts use absolute
paths and import `train` from the interpreter's editable install. Output goes
to stdout only.

The scripts are copied unchanged from the top level of the session scratch
directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad`.

## Data it reads (local only, not in the repo)

- bank (all scripts):
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz`
- cached training windows (`chk5.py`, `chk6.py`, `chk7.py`, `chk9.py`,
  `check_signal.py`):
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5`
- injection file (`chk2.py`, `chk8.py`):
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
- detection-MLP checkpoint (`check_logit_tree.py`):
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank_703_spin3_1024hz_detection_mlp_train_snr4_prob0.5/CLAUDE-TESTS/template_bank_703_spin3_1024hz_detection_mlp_train_snr4_prob0.5/checkpoints/ck_15.ckpt`
