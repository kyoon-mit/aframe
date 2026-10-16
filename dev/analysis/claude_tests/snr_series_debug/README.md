# SNR-series denoiser debug callbacks

`snrdbg.py` holds three Lightning callbacks for the denoiser trained on the
matched-filter SNR series (`train.experimental.snr_series`). `SNRDebug`
prints, at train steps 0 and 9 and on the first validation batch, the output
size, the peak SNR series of the output on signal and on noise-only rows in
train and eval mode, and the loss. `SNRRows` prints each validation row's
loss against its true peak SNR. `RowShare` prints the share of the loss and
of the output gradient carried by the largest row, in two windows of steps.

## What it backs

- The experiment log entry of 2026-10-07 for
  `den_snr_series_eps0.3_train_snr8_prob0.5` (local only, not in the repo)
- The loss is described in
  [Why denoising does not detect](https://kyoon-mit.github.io/projects/bns/updates/2026-10-07-why-denoising-does-not-detect/),
  section "Next: a loss built on the matched filter"

## How it was run

Imported as the module `snrdbg` and passed to the trainer as callbacks
(`snrdbg.SNRDebug`, `snrdbg.SNRRows`, `snrdbg.RowShare`) in
`/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python -m train fit`.
The exact command is not recorded.

The file is copied unchanged from
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/snrdbg`.

## Data it reads (local only, not in the repo)

- the training and validation batches of the run, through its config:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_snr_series_eps0.3_train_snr8_prob0.5/config.yaml`
