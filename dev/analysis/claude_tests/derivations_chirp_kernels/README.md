# Derivation checks: chirp-kernel proposal

`memory.py` reads trained S4D checkpoints and prints, per layer, the median
memory 1/(Δt α) in samples and milliseconds, the share longer than 8,192
samples, and the share of tones whose phase step per sample passes π.
`tuning.py` measures a fixed chirp kernel's tuning curve (half-maximum
points, response at −3%) and where a 1%-off kernel peaks on a clean signal.
`mvm.py` gives the match against the kernel's chirp-mass offset.

## What it backs

- [#ck-s4d-step](https://kyoon-mit.github.io/projects/bns/derivations/#ck-s4d-step):
  `memory.py`, on the epoch-630 den-reg checkpoint of the matched-filter
  benchmark
- [#ck-tuning](https://kyoon-mit.github.io/projects/bns/derivations/#ck-tuning):
  `tuning.py`, `mvm.py`
- [From damped tones to chirps](https://kyoon-mit.github.io/projects/bns/updates/2026-10-07-chirp-kernels/)

`checks.py` gives the checks for #ck-chirp, #ck-quadratures, #ck-gelu-mean,
#ck-fourier-tones (BNS chirps only: its Project 8 part read local Project 8
records and is left out) and #ck-noise.

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. `memory.py <checkpoint> ...`. `mvm.py` executes `tuning.py` from the
working directory, so the working directory is the folder that holds both:
originally the scratch folder
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/ckcheck`,
from which the scripts are copied unchanged.

## Data it reads (local only, not in the repo)

- `memory.py`: den-reg checkpoints given as arguments, such as
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/test_powerlaw_ep630/last_ep630.ckpt`
- `tuning.py`, `mvm.py`: none
