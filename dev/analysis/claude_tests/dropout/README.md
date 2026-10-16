# Dropout diagnosis

These scripts check why the S4D regressors trained on clean signals failed
with dropout 0.2. They score the same weights with dropout off and on, model
the maximum of a sum of channels under channel dropout, measure the
chirp-mass offset that shifted pooled values cause, count the dropout layers
in the three regressors, and plot validation within 1% by epoch with dropout
0.2 and 0.

## What it backs

The update of 16 October 2026,
[Why dropout broke our S4D chirp-mass regressor](https://kyoon-mit.github.io/projects/bns/updates/2026-10-16-why-dropout-broke-our-s4d/),
and its derivations:

- `diag_cpu.py`: the table "The same weights with dropout off and on";
  [#oct16-dropout-same-windows](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-same-windows),
  with `offset_check.py` in
  [#oct16-dropout-offset](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-offset),
  and the measured values of
  [#oct16-dropout-energy](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-energy).
- `offset_check.py`: the per-window maxima of
  [#oct16-dropout-max](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-max)
  and the offsets of #oct16-dropout-offset.
- `toy_mc.py`: [#oct16-dropout-toy](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-toy).
  `final/toy_mc.py` is the same file, rerun on 16 October.
- `final/checks_linear.py`:
  [#oct16-dropout-linear](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-linear)
  and the Student-t value of #oct16-dropout-max.
- `final/checks_arch.py`:
  [#oct16-dropout-where](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-dropout-where)
  and the group-norm check of #oct16-dropout-energy.
- `final/read_wandb_1016.py`, `final/plot_curves_1016.py`: the figure of
  validation within 1% by epoch (`dropout_curves.png`) in the section "The
  runs".
- `read_wandb.py`, `plot_dropout.py`: figures of the first draft (15
  October), not on the site.

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
on the CPU, from this folder (for example `python toy_mc.py`). Each script
writes its outputs next to itself; `final/plot_curves_1016.py` takes the
output path as its argument.

The scripts are copied unchanged from the session scratch directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/dropout`.

## Data it reads (local only, not in the repo)

- cached clean training windows:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5`
- checkpoints in
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/CLAUDE-TESTS/<run>/checkpoints/`
  of `sig_only_reg_s4d_max_64x32x4`, `sig_only_reg_s4d_max_dropout0_64x32x4`,
  `sig_only_reg_s4d_64x32x4` and `sig_only_reg_s4d_complex_energy_max_64x32x4`
- local wandb files of those runs and of `sig_only_reg_s4d_dropout0_64x32x4`:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/wandb/`
- `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/warp_table.py`, whose
  `rows()` both `read_wandb` scripts execute (the same file is in this repo
  as `dev/local/train/claude_tests/queue/warp_table.py`), and its cache copy
  `warp_table_cache_copy.json`
- in the scratch directory: the outputs that the plotting scripts read
  (`curves.json`, `final/curves_1016.json`, `diag_cpu.json`)
