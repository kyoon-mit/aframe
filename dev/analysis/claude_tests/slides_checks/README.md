# Checks for the 14 October slides

Numbers on the slides of 14 October and in the derivations they link.

- `init_tones.py`: share of S4D-Lin starting tones that lie in band and last
  over 1 s, by an exact average over Δt on a 200,001-point grid
- `s4d_lin_tone_stats.py`: the same from ml4gw's `S4DKernel` initialization,
  by Monte Carlo over channels; writes `s4d_lin_tone_stats.json` next to
  itself
- `warp_tones.py`: Fourier tone counts before and after the leading-order
  time warp, for the toy chirps and four whitened training templates; writes
  `results.json` next to itself
- `unwarped_cut.py`: unwarped tone counts of the same templates cut where the
  warp cuts them
- `snrfrac.py`: SNR held by the last 3.75 s and 4 s against 30 s and against
  20 Hz, and the volume factor
- `persample.py`: per-sample size of cached training signals scaled to
  network SNR 8
- `status.py`, `status2.py`, `wbread.py`, `wbval.py`, `summ.py`: read
  validation histories from local wandb files (best, last, and means over
  epoch ranges)

## What it backs

- [Slides: chirps, tones and learned filters](https://kyoon-mit.github.io/projects/bns/updates/2026-10-14-slides-chirps-tones-filters/)
- [#s4d-tones](https://kyoon-mit.github.io/projects/bns/derivations/#s4d-tones):
  `init_tones.py`, `s4d_lin_tone_stats.py`
- [#time-warp](https://kyoon-mit.github.io/projects/bns/derivations/#time-warp):
  `warp_tones.py`, `unwarped_cut.py`
- [#proj-per-sample](https://kyoon-mit.github.io/projects/bns/derivations/#proj-per-sample):
  `persample.py` (a second check, on training windows)
- [#late-window-volume](https://kyoon-mit.github.io/projects/bns/derivations/#late-window-volume):
  `snrfrac.py` (a second check, from the noise spectrum alone)
- the slides' validation tables ("best validation so far, read from the
  training logs at 16:20 on 14 October"): `status.py`, `status2.py`,
  `wbread.py`, `wbval.py`, `summ.py`

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. `status.py <out.json>` lists the run folders of its working directory,
so it ran in `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests`;
`status2.py <out.json> <run> ...` takes run folders relative to that folder.
`wbread.py <file.wandb> [key substring ...]`,
`wbval.py <out.json> <file.wandb> ...`, `summ.py <json>=<name> ...`. The
other scripts use absolute paths only.

The scripts are copied unchanged from the scratch folders
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/qa`
(`init_tones.py`, `s4d_lin_tone_stats.py`, `status.py`, `status2.py`),
`.../scratchpad/qa/warp` (`warp_tones.py`) and `.../scratchpad/rev_calc`
(the rest). `unwarped_cut.py` reads `qa/warp/results.json` at its scratch
path.

## Data it reads (local only, not in the repo)

- cached training windows:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5`
- injection file:
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
- tone-check results:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/tone_check/results.json`
- warp results (scratch only):
  `/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/qa/warp/results.json`
- local wandb files:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/wandb/run-*/*.wandb`
