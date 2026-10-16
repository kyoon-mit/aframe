# Matched-filter benchmark

A matched filter run on the same test windows the den-reg network sees at
epoch 630. The code is in this repo, in `dev/benchmarks/` (commit
`ac0cc0d`). `run_dump.sh` runs that checkpoint in test mode on three
populations (SNR power law of index −3 from 4 and from 8, uniform 4 to 50;
400 batches of 32) with the callback `dumpcb.DumpTest`, which saves each
batch's whitened windows, clean signals, PSDs, parameters and the network's
mean and σ to `windows.h5`. `mf.py` builds a bank from the injection file
(chirp-mass grid 0.5% apart, mass ratios 0.5, 0.7 and 0.9, lowest spin),
whitens every template with each window's own spectrum, and writes the best
template's chirp mass and the network SNR per window to `mf.h5`. `compare.py`
prints within 2% by SNR band and the AUC of signal against background, for
the filter and the network. `plots.py` draws the accuracy, ROC and SNR
figures.

## What it backs

- [A matched filter on the same windows the network sees](https://kyoon-mit.github.io/projects/bns/updates/2026-10-07-matched-filter-benchmark/)
- The reference table at the top of the experiment log (local only, not in
  the repo)
- Its outputs are read by `../derivations_oct7/`
  ([#oct7-noise-floor](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-noise-floor),
  [#oct7-bank-spin](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-bank-spin))

## How it was run

`bash run_dump.sh`: it changes to `/home/kyoon/SSM-BNS/aframe/projects/train`
and runs `.venv/bin/python -m train test` on GPU 0, with `PYTHONPATH` set to
the folder that holds `dumpcb.py`. Then, with
`/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`:
`mf.py <windows.h5> <mf.h5>` (CUDA), `compare.py <population folder>`,
`plots.py <benchmark folder>`.

The run used the scratch copy
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/mfbench`.
Its files equal those in `dev/benchmarks/`, except that its `run_dump.sh`
set `PYTHONPATH` to that scratch folder. The same files also sit in the
benchmark folder's `scripts/`.

## Data it reads (local only, not in the repo)

- run folder
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/`:
  `mf_bench_ep630/run_config_snapshot.yaml`, the checkpoint
  `test_powerlaw_ep630/last_ep630.ckpt`, and the outputs
  `mf_bench_ep630/<population>/windows.h5` and `mf.h5`
- injection file
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
- the background strain named in the run config

`run_dump.sh` also reads the test config
`dev/local/train/denoiser/claude_tests/test/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_snr_d128_powerlaw.yaml`
(in this repo).
