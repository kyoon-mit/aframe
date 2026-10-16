# Early-stopping patience replay

`replay_patience.py` replays plain early stopping on every local training
history in `dev/runs/claude_tests` with at least 60 validation epochs of
`val/within_1pct_chirp_mass`, and prints the largest shortfall against the full
history for each patience, separately for signal-only and noisy runs. It reads
the runs' local wandb files through `dev/runs/claude_tests/warp_table.py` (a copy
is in `dev/local/train/claude_tests/queue/`); the run folders are local only.

Backs [#oct16-stopping](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-stopping).
Run with `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python replay_patience.py`.
On 16 October at 18:04 (36 histories, 18 signal-only and 18 noisy) it gave a
largest signal-only shortfall of 5.57 points up to patience 40, 5.18 at 50 and
0.05 at 60 and 80, and at patience 160 one noisy run above 0.5 points.
