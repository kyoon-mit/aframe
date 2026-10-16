# Run management

Launchers, a watcher and table readers for the runs whose configs are in
`../oracle/` and `../long_conv/`, test runners for the den-reg runs, and a
waveform loader test.

- `launch_lowmem2.sh`: the launcher from 2026-10-16 11:30. Steps through a
  fixed job list, starting a job or swapping a running one right after its
  end-of-epoch checkpoint. A launch waits until at most 11 training jobs run,
  MemAvailable stays at or above 70 GB for 60 s (85, then 100 GB on retries),
  /dev/shm has 30 GB free, 20 min have passed since the last launch (10 min
  for a swap), and the previous job is past an epoch (or 45 min). Every job
  loads waveforms in the main process (`waveform_workers 0`,
  `pin_waveforms false`) and gets `oom_score_adj` 1000 and a 30 min guard
  that kills only it if MemAvailable falls below 20 GB or /dev/shm free below
  10 GB. Up to 3 tries; a clean early stop counts as done.
- `launch_fast.sh`: the launcher from 2026-10-16 11:25: a launch every 3 min
  while MemAvailable stays at or above 60 GB and /dev/shm free at or above
  30 GB for 30 s; for 3 min after each launch, a monitor kills only the
  newest job if MemAvailable falls below 15 GB.
- `watch_lm2.sh`: exits on a new launch or guard line in
  `launch_lowmem2.out`, an OOM kill in the kernel log, a shared-memory error
  or traceback in a training log, or after 2 h.
- `warp_table.py`: the 4 x 3 time-warp table (four endings of the S4D
  64x32x4 on merger, warped merger and warped premerger windows), validation
  within 1/2/5/10%, from local wandb files; modes `last`, `best`, `matched`.
  Caches its scans in `queue/warp_table_cache.json` under the runs folder.
- `cellstate.py`, `cellstate2.py`: for each table run, the last epoch, the
  best within 1% and its epoch, the epoch at which early stopping fires
  (patience 80, or 160 for the noisy free-kernel run, and not before the end
  of the learning-rate cycle that holds the best epoch), and the count of
  early-stop messages in its log. Both load `warp_table.py` from the runs
  folder.
- `run_tests_ep336.sh`, `run_tests_ep630.sh`, `run_tests_ft.sh`: test runs
  (`python -m train test` with `PlotParamEstCallback`): epoch 336 of the
  SNR-8 den-reg run and the last checkpoint of the fine-tune on SNR 8 and
  above (`ft_m3_snr8_rop_from_d128_n6_ep634`) on three populations (SNR
  power law of index −3 from 4 and from 8, uniform 4 to 50); epoch 630 of
  the d128 run on two (power law from 8, uniform 4 to 50).
- `wfw_test.py`: the waveform chunk loader with 0 workers: 200 batches, time
  and peak RSS.

## What it backs

- [Status: dropout was the bug](https://kyoon-mit.github.io/projects/bns/updates/2026-10-16-status-dropout-and-signal-only/),
  the time-warp table and its stopping rule: `cellstate.py`,
  `warp_table.py`
- [Why denoising does not detect](https://kyoon-mit.github.io/projects/bns/updates/2026-10-07-why-denoising-does-not-detect/),
  the table of epoch 630 against epoch 336: `run_tests_ep630.sh`,
  `run_tests_ep336.sh`
- [5 October, epoch 630](https://kyoon-mit.github.io/projects/bns/updates/2026-10-05-recipe-at-epoch-630/):
  `run_tests_ep630.sh`
- [5 October, window bound and prior conditioning](https://kyoon-mit.github.io/projects/bns/updates/2026-10-05-window-bound-and-prior-conditioning/),
  the fine-tune on SNR 8 and above: `run_tests_ft.sh`
- The experiment log entries of 2026-10-15 14:42 (warp table), 2026-10-16
  05:51 (low-memory loading test), 2026-10-16 11:30 (`launch_lowmem2.sh`)
  and 2026-10-16 12:05 (`launch_fast.sh`) (local only, not in the repo)

## How it was run

Shell scripts with `bash`; they change to
`/home/kyoon/SSM-BNS/aframe/projects/train` and run `.venv/bin/python -m
train ...`. Python scripts with
`/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`. Working
directory: any (absolute paths). The launchers write job state (`.sid`
files, `.out` logs, training logs) under
`/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests`.

Copied unchanged from `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests`
(`warp_table.py`), its `queue/` (`launch_fast.sh`, `launch_lowmem2.sh`,
`watch_lm2.sh`), and the top level of the session scratch directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad`
(the rest). Paths inside point to the runs folder, not to this folder.

## Data it reads (local only, not in the repo)

- run folders, training logs, checkpoints and wandb files under
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/`
- for the test runners: each run's `config.yaml` or
  `run_config_snapshot.yaml` and checkpoint in its run folder; the test
  config
  `dev/local/train/denoiser/claude_tests/test/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_snr_d128_powerlaw.yaml`
  is in this repo
- injection file (`wfw_test.py`):
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
- machine state: `/proc/meminfo`, `/dev/shm`, the process list, and the
  kernel log (`watch_lm2.sh`)
