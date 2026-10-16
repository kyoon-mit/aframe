# Derivation checks: free-kernel page

`early_numbers.py` prints SNR power-law fractions and medians (density index
3 and 4, from SNR 4 and 8), the H1 to L1 light travel time from ml4gw's
detector vertices, the ±10 ms window in samples at 2048 and 1024 Hz, S4D and
free-kernel parameter counts by formula, the step-time ratio, the mixing
operation count, the band-start spacing, a relative half-width quoted from
DINGO-BNS, the sensitive-volume ratios of 12 October, and the size of the
chirp-mass grid.

## What it backs

- [#early-light-travel](https://kyoon-mit.github.io/projects/bns/derivations/#early-light-travel)
- [#early-s4d-count](https://kyoon-mit.github.io/projects/bns/derivations/#early-s4d-count)
  (the counts by formula)
- [The free-kernel network](https://kyoon-mit.github.io/projects/bns/free-kernel/)

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: not recorded and not needed (absolute path only).

The script is copied unchanged from
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/early_checks`.

## Data it reads (local only, not in the repo)

- ml4gw's source file, read for the L1 vertex:
  `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/lib/python3.11/site-packages/ml4gw/utils/interferometer.py`
