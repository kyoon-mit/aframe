# Derivation checks: 7 and 8 October

Checks of the derivations behind the 7 October updates and the 8 October
chirp-kernel update: the matched-filter noise floor on empty data, validation
events per epoch, a tone's frequency precision, the Newtonian chirp scalings,
the time average of an oscillating output, the bank's spin limit, and the
benchmark's tables.

## What it backs

Derivations page:

- [#oct7-noise-floor](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-noise-floor):
  `noisefloor_mc.py` (the benchmark filter on Gaussian white noise and on
  real background), `numbers.py` (median largest value over many trials)
- [#oct7-val-count](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-val-count):
  `numbers.py`
- [#oct7-tone-precision](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-tone-precision):
  `tone_crb.py`
- [#oct7-chirp-kernels](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-chirp-kernels):
  `numbers.py`
- [#oct7-time-average](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-time-average):
  `numbers.py`
- [#oct7-bank-spin](https://kyoon-mit.github.io/projects/bns/derivations/#oct7-bank-spin):
  `mfstats.py`
- [#snr-population](https://kyoon-mit.github.io/projects/bns/derivations/#snr-population):
  `numbers.py` (power-law and uniform-in-volume fractions)

Updates:

- [A matched filter on the same windows the network sees](https://kyoon-mit.github.io/projects/bns/updates/2026-10-07-matched-filter-benchmark/):
  `mfstats.py` (recovered against injected SNR, background scores, weak
  signals), `snrloss.py` (SNR held by the window and found by the filter,
  against injected), `verify_bench.py` (within 1/2/5/10% tables, AUC)
- [Chirp kernels work](https://kyoon-mit.github.io/projects/bns/updates/2026-10-08-chirp-kernels-work/):
  `numbers.py`

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
CPU. Working directory: the scratch folder
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/oct7work`,
where the saved outputs sit. `noisefloor_mc.py` takes the number of
background windows as its argument (300 in the derivation) and saves
`noisefloor_gauss.npy` to that scratch folder by absolute path.

The scripts are copied unchanged from that folder.

## Data it reads (local only, not in the repo)

- benchmark windows and filter outputs
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630/<population>/windows.h5`
  and `mf.h5` (made by the benchmark code in `dev/benchmarks/`; see
  `../matched_filter_benchmark/`)
- injection file (`noisefloor_mc.py`):
  `/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5`
