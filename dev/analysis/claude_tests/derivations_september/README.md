# Derivation checks: September and 5 October updates

`sep_stats.py` reads per-event test outputs (true value, prediction, σ, SNR)
and prints the z-score mean and spread, |z| fractions against the Gaussian
values, mean |error| over mean σ, within 1/2/5/10% by SNR band,
constant-prediction baselines, and the AUC of −σ against background.
`sep_wandb_hist.py` extracts chosen history keys from a local `.wandb` file
to JSON. `sep_count.py` prints the number of stored values (all tensor
sizes summed) and the epoch of each checkpoint.

## What it backs

- [#sep-zscore](https://kyoon-mit.github.io/projects/bns/derivations/#sep-zscore),
  [#sep-prior](https://kyoon-mit.github.io/projects/bns/derivations/#sep-prior):
  `sep_stats.py`
- [#sep-reweight](https://kyoon-mit.github.io/projects/bns/derivations/#sep-reweight):
  `sep_stats.py`, `sep_wandb_hist.py` (validation within 1% and 2%)
- [#sep-ls-gain](https://kyoon-mit.github.io/projects/bns/derivations/#sep-ls-gain),
  [#sep-curriculum](https://kyoon-mit.github.io/projects/bns/derivations/#sep-curriculum):
  `sep_wandb_hist.py` (validation ρ and gain)
- [#calibration](https://kyoon-mit.github.io/projects/bns/derivations/#calibration):
  `sep_stats.py` (mean |error| over mean σ)
- `sep_count.py`: the section it served is not recorded

These sections back the updates of
[8 September](https://kyoon-mit.github.io/projects/bns/updates/2026-09-08-denoiser-loss/),
[29 September](https://kyoon-mit.github.io/projects/bns/updates/2026-09-29-den-reg-and-project-8/)
and
[5 October](https://kyoon-mit.github.io/projects/bns/updates/2026-10-05-recipe-at-epoch-630/).

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: not recorded. Arguments:

- `sep_stats.py <test folder> ...`, each folder holding
  `param_est_results.csv` and `param_est_results_bkg.csv`
- `sep_wandb_hist.py <file.wandb> <key,key,...> <out.json>`
- `sep_count.py <checkpoint> ...`, paths relative to a folder of runs (it
  prints the first path component as the run name)

The scripts are copied unchanged from the top level of the session scratch
directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad`.

## Data it reads (local only, not in the repo)

- per-event test outputs, local `.wandb` files and checkpoints of the runs
  named in those sections, under `/home/kyoon/SSM-BNS/aframe/dev/runs/`
