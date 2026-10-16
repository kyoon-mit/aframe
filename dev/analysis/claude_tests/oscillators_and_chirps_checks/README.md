# Checks for the 15 October update

`check_fix.py` computes numbers for the answers added to the 15 October
update: in-step stretches in the last 4 s, the average slope of
x Φ(x) and the SNR an in-step sum keeps after it, the band start's repeat
time and decay, the mean largest of 100 standard normals, background reuse
per epoch, heterodyne leftovers in time and in frequency (with group delays),
the first minimum of a 1 s Hann window's transform, and the wider network's
learned numbers. `trend.py`: validation within 1% per 20-epoch window, mean
of the last ten epochs, best epoch, and slope over the last 60 epochs.
`arms_val_copy.py` and `multi_val.py`: validation within 1% averaged over
epoch ranges for the noisy runs. `arms_val_copy.py` is the same file as
`../lecture_checks/lecture/wb/arms_val.py`; the other checks of this update
are in `../lecture_checks/`.

## What it backs

- [Oscillators and chirps](https://kyoon-mit.github.io/projects/bns/updates/2026-10-15-oscillators-and-chirps/):
  numbers in the rebuttals and final verdicts, such as the in-step
  stretch count 1.55√N, the band start's 0.52 s repeat, the leftover
  frequencies after heterodyning, and the epoch 200 to 250 means within 1%
  (`check_fix.py`, `trend.py`, `arms_val_copy.py`, `multi_val.py`)

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: not recorded and not needed (absolute paths).
`trend.py`, `arms_val_copy.py` and `multi_val.py` take run folder names
under `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests` as arguments.

The scripts are copied unchanged from the scratch folders
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/fix`
(`check_fix.py`, `trend.py`) and `.../scratchpad/audit`
(`arms_val_copy.py`, `multi_val.py`).

## Data it reads (local only, not in the repo)

- O3a background file names and durations (`check_fix.py` part E):
  `/home/kyoon/SSM-BNS/DATA/O3a_H1_L1_2048Hz/background/background/*.hdf5`
- checkpoint of the wider network (`check_fix.py` part J):
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96/CLAUDE-TESTS/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96/checkpoints/last.ckpt`
- local wandb files:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/wandb/run-*/*.wandb`
