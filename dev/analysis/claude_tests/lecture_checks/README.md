# Lecture number checks

These scripts recompute the numbers of the lecture-style update of 15
October 2026 and of its revision on 16 October: leading-order chirp formulas
(how long a chirp stays in step with one frequency, the loss from a steady
phase drift, leftover cycles after heterodyning, grid masses near an
estimate), the oscillator form of one S4D state, validation values read from
local wandb files, the length of the training background, and GPU time per
training batch.

## What it backs

The lecture
[Oscillators and chirps](https://kyoon-mit.github.io/projects/bns/updates/2026-10-15-oscillators-and-chirps/):

- `lecture/calc.py`: the first numbers of the lecture: in-step time
  (section 4), phase drift (section 2), oscillator form (section 3), leftover
  cycles (section 8), grid masses (section 9).
- `lecture/check_oct15.py`: independent checks of
  [#oct15-in-step-time](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-in-step-time),
  [#oct15-phase-drift](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-phase-drift),
  [#semi-coherent](https://kyoon-mit.github.io/projects/bns/derivations/#semi-coherent),
  [#oct15-het-grid](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-het-grid),
  [#oct15-grid-count](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-grid-count),
  [#oct15-oscillator](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-oscillator),
  and the data numbers of section 7.
- `lecture/wb/`: validation within 1% per epoch of the runs (sections 5 and
  6); `h5dur.py`: length of the O3a training background (section 7).
- `lecture_rev/scan.py`, `lecture_rev/verify/digits.py`,
  `lecture_rev/verify/epoch_means.py`: best and last values and means over
  epoch ranges for the revision (sections 5 and 6).
- `lecture_rev/verify/calc.py`: numbers of the revision and of its
  objections and rebuttals: entry frequency and cycles when the merger is
  3.5 s into the window (section 1), SNR per cycle (section 2), band-start
  channel match (section 5), training background and its reuse per epoch
  (sections 6 and 7), leftover after heterodyning (section 8).
- `lecture/step_time.py`, `lecture/plateau.py`: the experiment log entry of
  2026-10-15 15:40 (local only, not in the repo): GPU time per batch for data
  preparation and for the model step, and hours to 80% and 90% of the best
  validation value.

More checks of the same lecture are in `../oscillators_and_chirps_checks/`.

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`.
The scripts use absolute paths, so the working directory does not matter.
The wandb scripts take run names as arguments, except
`lecture/wb/read_wandb.py`, which takes `.wandb` file paths.
`lecture/check_oct15.py` was run with
`uv run -q --with numpy --with sympy python -I check_oct15.py`.
`lecture/step_time.py` needs a GPU and builds no data module.

The scripts are copied unchanged from the session scratch directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad`,
folders `lecture/` and `lecture_rev/`. Comments in
`lecture_rev/verify/calc.py` name other scratch scripts, which are not in the
repo.

## Data it reads (local only, not in the repo)

- local wandb files of the runs: `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/wandb/`
- O3a background files: `/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz/`
