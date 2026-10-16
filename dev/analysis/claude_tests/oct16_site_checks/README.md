# Checks behind the 16 October site sources

Scripts that drafted and then independently checked the numbers added to the
BNS pages on 16 October 2026 (site commit fef42d6). Each pair of folders is a
draft and its adversarial check.

| folder | checks | site section |
|---|---|---|
| `nonlinearity/` | share of the SNR a sum in step keeps after a GELU, ReLU or threshold: closed forms, quadrature, Monte Carlo | [#oct16-nonlinearity-linear-part](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-nonlinearity-linear-part) |
| `heterodyne_dropout_timing/`, `heterodyne_dropout_timing_2/` | worst case of the ten-mass heterodyne grid; dropout in the 23 September configs; the step timing table | [#oct15-het-grid-worst](https://kyoon-mit.github.io/projects/bns/derivations/#oct15-het-grid-worst), [#oct16-step-time](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-step-time), [23 September update](https://kyoon-mit.github.io/projects/bns/updates/2026-09-23-denoiser-scans/) |
| `derivations_checker/` | independent rerun of the three folders above, and the KaTeX and kramdown render of the new text | same |
| `merger_position/` | where the merger falls in the 4 s windows, from the data code; the cycle and resonator counts for that range | [#oct16-merger-position](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-merger-position), [#cycles](https://kyoon-mit.github.io/projects/bns/derivations/#cycles) |
| `glossary/` | glossary entries: format, YAML, the Hann window's main lobe | [definitions](https://kyoon-mit.github.io/projects/bns/definitions/) |
| `merger_glossary_checker/`, `merger_glossary_checker_2/` | independent check of the two folders above, including a Monte Carlo of merger placement with sky positions | as above |
| `figure_captions/` | figure captions of the S4D update against the saved fits | [What S4D computes](https://kyoon-mit.github.io/projects/bns/updates/2026-10-16-what-s4d-computes/) |
| `chirplet/`, `chirplet_checker/` | how long a chirp stays in step with a linear chirp, how many chirplets cover a window, chirplet fits to whitened templates | [#oct16-chirplet-count](https://kyoon-mit.github.io/projects/bns/derivations/#oct16-chirplet-count), [brainstorm](https://kyoon-mit.github.io/projects/bns/brainstorm/) |
| `chirplet_first_attempt_fits/`, `chirplet_first_attempt_literature/` | an earlier, unfinished chirplet attempt; its fits were reused by `chirplet/` | none |

All ran on CPU, single-threaded, with `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`.
They read inputs from the session scratch directory
(`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/`):
saved fits, whitened kernels and earlier outputs. Those inputs are local only, not in the repo.
