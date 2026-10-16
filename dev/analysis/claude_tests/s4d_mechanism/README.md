# S4D mechanism tests

These scripts check what one S4D channel computes. A channel is a set of
damped, driven resonators. The scripts count how many resonators a channel
needs to act as the matched filter of a binary neutron star chirp, fit
resonators to chirp kernels, measure what the trained S4D channels hold, run
a pre-registered test of how the S4D trained on clean signals reads chirp
mass, train fresh read-outs on the channel outputs of the noisy networks, and
measure a GPU FFT offset in the long-convolution code.

## What it backs

The update of 16 October 2026,
[What an S4D channel computes](https://kyoon-mit.github.io/projects/bns/updates/2026-10-16-what-s4d-computes/),
its derivations, and the experiment log entry of 2026-10-16 01:20 (local
only, not in the repo).

- `theory/`: one channel as a sum of resonances (`c1_channel.py`), delays
  (`c2_delay.py`), the phase limit and the table of resonator counts
  (`c3_count.py`), one resonator alone (`c4_decay.py`, `c8_quarter.py`),
  resonator fits and counts for a match of 0.9 (`c5_fits.py`, `c6_snr.py`,
  `c9_real.py`, `c10_gram.py`), the linear part of a GELU (`c7_layer.py`).
  `audit_tables.py` checks the tables of a local write-up. Update sections 1
  and 2; derivations
  [#oct16-s4d-resonance-curve](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-resonance-curve),
  [#oct16-s4d-matched-delay](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-matched-delay),
  [#oct16-s4d-ring-delay](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-ring-delay),
  [#oct16-s4d-phase-limit](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-phase-limit),
  [#oct16-s4d-three-eighths](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-three-eighths),
  [#oct16-s4d-gelu-linear](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-gelu-linear).
- `fits/`: fits of N resonators to whitened chirp kernels (matrix pencil,
  layout along the chirp, variable projection, random restarts); `report.py`
  makes the tables and figures. Update section 2, "What the fits show";
  [#oct16-s4d-fits](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-fits).
- `kernels/`: resonances, transfer functions, group delays and template
  matches of every channel of the trained networks. Update section 3;
  [#oct16-s4d-channel-analysis](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-channel-analysis).
- `harness/`: 4,000 clean and 20,000 noisy test windows made as the training
  pipeline makes validation windows, checks against the pipeline's own code,
  and loaders for the trained networks. Used by `mechanism/`, `predict/` and
  `verify/`.
- `PREREGISTERED.md`, `mechanism/plan.json`, `mechanism/`: the pre-registered
  tests T2, T3 and T4 on clean windows, the taper-only control, and T1
  (descriptive). Update section 4.
- `predict/`: fresh read-outs trained on the channel outputs of the four
  noisy networks and of a 256-template bank; `run_readouts.sh`, `run_big.sh`
  and `run_es8.sh` run the steps in order. Update section 5. The exact-FFT
  variants in `features.py` give the scores without the offset of section 6.
- `verify/`: a second pass with its own code. `v_cufft.py`:
  [#oct16-s4d-fft-offset](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-s4d-fft-offset);
  `v_allpass.py`, `v_theory.py`: theory numbers; `v_fits.py`: fits;
  `v_kernels.py`, `v_kernels135.py`, `v_oneres.py`: trained channels;
  `v_onset.py`, `v_shuffle.py`: the window-start finding of section 4;
  `v_predict.py`, `v_scores.py`, `v_snrdraw.py`, `v_wandb.py`: section 5.
- `writeup_checks/`: numbers recomputed for the local write-up behind the
  update.
- `replot/`: `replot.py` redraws figures 1 to 3 of the update from saved
  results (default output folder: the site's
  `docs/assets/img/bns/2026-10-16`); `check_tables.py` rebuilds the fit
  tables from the raw fit files.

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`.
Working directory: `/home/kyoon/SSM-BNS/aframe/projects/train`, with each
script given by its full path (the `predict/*.sh` drivers `cd` there); the
`fits/` scripts were also run from their own folder. Scripts find their
sibling folders (`harness/`, `kernels/`) from their own location and write
their outputs next to themselves. Several need a GPU.

The scripts are copied unchanged from the session scratch directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath`.
Absolute paths inside some of them (mostly `verify/`, `writeup_checks/` and
`predict/*.sh`) still point there.

## Data it reads (local only, not in the repo)

- O3a background: `/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz`
- waveforms: `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
  and `/home/kyoon/SSM-BNS/DATA/aframe/val/end_o3_ratesandpops_bns_snr4.hdf5`
- cached training windows and their PSDs:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5`
- average O3 PSD of the template bank:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz`
- earlier tone-fit outputs: `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check/`
- local wandb files of the runs: `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/wandb/`
- in the scratch directory: checkpoint copies (`harness/ckpt/`,
  `kernels/ckpt/`, `theory/ckpt/`, `verify/ckpt/`), the saved window sets,
  features and results (`harness/data/`, `predict/data/`,
  `predict/results/`, `fits/`, `kernels/`, `mechanism/`, `theory/results/`,
  `verify/`), and the write-ups `writeup.md`, `theory/theory.md` (read by
  `theory/audit_tables.py`) and `kernels/kernels.md`

The run configs come from `dev/local/train/claude_tests/oracle/` and
`dev/local/train/claude_tests/long_conv/` in this repo.
