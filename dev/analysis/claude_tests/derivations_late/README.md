# Derivation checks: 9 to 14 October

`checks.py`: power-law fractions by formula and by Monte Carlo, the best
scale under a squared-error loss, the maximum log-likelihood ratio ρ²/2 on a
test chirp, binomial spreads, volume factors from SNR fractions, σ
calibration, chance levels for chirp mass, and the spacing of 125 log-spaced
kernels. `last4.py`: SNR held by the last 4 s of injection-file waveforms and
the volume factor. `tests.py`: within 1/2% by SNR band on the 50k test sets
of six models. `wb.py`: per-epoch validation histories of two den-reg runs
from wandb. `dead_kernels.py`: the free-kernel ablation at epoch 141 (125
chirp-like against 131 other kernels), re-run on CPU.

## What it backs

- [#late-snr-bands](https://kyoon-mit.github.io/projects/bns/derivations/#late-snr-bands):
  `checks.py`, `tests.py`
- [#late-shrinkage](https://kyoon-mit.github.io/projects/bns/derivations/#late-shrinkage):
  `checks.py`
- [#late-binomial](https://kyoon-mit.github.io/projects/bns/derivations/#late-binomial):
  `checks.py`, `wb.py`
- [#late-beta-half](https://kyoon-mit.github.io/projects/bns/derivations/#late-beta-half):
  `checks.py`
- [#late-window-volume](https://kyoon-mit.github.io/projects/bns/derivations/#late-window-volume):
  `last4.py`, `checks.py`
- [#late-chance](https://kyoon-mit.github.io/projects/bns/derivations/#late-chance):
  `checks.py`, `dead_kernels.py`
- [#calibration](https://kyoon-mit.github.io/projects/bns/derivations/#calibration):
  `checks.py`
- the 50k test tables of
  [Without templates](https://kyoon-mit.github.io/projects/bns/updates/2026-10-13-learned-long-kernels/),
  re-computed by `tests.py`

These sections back the updates of
[9 October](https://kyoon-mit.github.io/projects/bns/updates/2026-10-09-shape-and-amplitude-terms/),
[12 October](https://kyoon-mit.github.io/projects/bns/updates/2026-10-12-template-bank-network/),
[13 October](https://kyoon-mit.github.io/projects/bns/updates/2026-10-13-learned-long-kernels/)
and
[14 October](https://kyoon-mit.github.io/projects/bns/updates/2026-10-14-den-reg-final/).

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: the scratch folder
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/late`;
`tests.py` writes `tests.pkl` and `wb.py` writes `hist_with.csv` and
`hist_without.csv` there. `dead_kernels.py` saves `dead_kernels_est.npz` to
that scratch folder by absolute path. `wb.py` needs network access and a
wandb login (project DENOISER-SCAN, default entity).

The scripts are copied unchanged from that folder.

## Data it reads (local only, not in the repo)

- test outputs `param_est_results.csv` and `param_est_results_bkg.csv` under
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/<test folder>/<population>/`
  for the free-kernel, den-reg and template-bank runs (`tests.py`)
- injection file (`last4.py`):
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
- bank files
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz`
  (`last4.py`) and `bank_mc0.5pct_spin3_q0.8.npz` (`dead_kernels.py`)
- free-kernel checkpoint
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96/test_ck_141/test.ckpt`
  and benchmark windows
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630/snr4_uniform/windows.h5`
  (`dead_kernels.py`)
- run histories in the wandb project DENOISER-SCAN, read over the network
  (`wb.py`)
