# Derivation checks: overview, background and story pages

`calc.py`: leading-order time from 20 Hz (also 10 and 30 Hz) to merger and
cycle counts, frequency and cycles in the last 4 s, signal per sample for
8,192 samples, SNR population fractions with the 29 September test set's
counts per SNR band, the H1 to L1 distance from LALSuite, leftover cycles
after heterodyning with a chirp-mass error, and the step of a 100-point log
grid. `persample.py`: per-sample size of a whitened 1.4 + 1.4 M☉ IMRPhenomD
signal (aLIGO design spectrum) in its last 4 s. `persample_o3b.py`: the same
with a Welch spectrum of O3b H1 strain, plus a check that whitened noise has
unit variance.

## What it backs

- [#proj-time-in-band](https://kyoon-mit.github.io/projects/bns/derivations/#proj-time-in-band):
  `calc.py`
- [#proj-per-sample](https://kyoon-mit.github.io/projects/bns/derivations/#proj-per-sample):
  `persample_o3b.py`, `persample.py`, `calc.py`
- [#proj-ring](https://kyoon-mit.github.io/projects/bns/derivations/#proj-ring):
  `calc.py`
- [#proj-heterodyne](https://kyoon-mit.github.io/projects/bns/derivations/#proj-heterodyne):
  `calc.py`
- [#snr-population](https://kyoon-mit.github.io/projects/bns/derivations/#snr-population):
  `calc.py` (Monte Carlo median, 29 September test counts)
- pages: [overview](https://kyoon-mit.github.io/projects/bns/),
  [background](https://kyoon-mit.github.io/projects/bns/background/),
  [story](https://kyoon-mit.github.io/projects/bns/story/)

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: not recorded and not needed. `persample.py` and
`persample_o3b.py` import `lal` and `lalsimulation`.
`persample_o3b.py <O3b background HDF5 file>` reads the file's `H1` dataset.

The scripts are copied unchanged from
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/proj_calc`.

## Data it reads (local only, not in the repo)

- the O3b background HDF5 file passed to `persample_o3b.py` (its first
  1,024 s of H1 strain)
