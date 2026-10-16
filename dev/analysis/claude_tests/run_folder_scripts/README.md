# Scripts kept in the run folders

Copies of the scripts that live next to the test runs in the git-ignored
`dev/runs/claude_tests/<folder>/`, with the same folder names. They run from
there: inputs and outputs are in those run folders (local only, not in the
repo), and the trained checkpoints are on Hugging Face (`kyoon-mit/SSM-BNS-WIP`,
`claude_tests/<date>/<run>/`).

| folder | what it does | site |
|---|---|---|
| `tone_check/` | tone count one S4D channel needs to reproduce a whitened BNS matched-filter kernel | [#early-tone-fit](https://kyoon-mit.github.io/projects/bns/derivations/#early-tone-fit) |
| `bank_check/` | peak network SNR² of every template of a real-waveform bank in every benchmark window, and the chirp-mass accuracy of bank variants | [template-bank network, 12 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-12-template-bank-network/) |
| `template_bank/` | figures, test comparisons, ROC, sensitive volume and model cost (parameters, FLOPs, throughput) of the template-bank network and the compared models | [12 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-12-template-bank-network/), [13 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-13-learned-long-kernels/), [14 Oct den-reg](https://kyoon-mit.github.io/projects/bns/updates/2026-10-14-den-reg-final/) |
| `oracle_plots/` | validation curves of the signal-only test, the free-kernel diagrams, and the max-pool S4D error with dropout off and on | [14 Oct signal-only test](https://kyoon-mit.github.io/projects/bns/updates/2026-10-14-signal-only-test/), [free kernels](https://kyoon-mit.github.io/projects/bns/free-kernel/), [16 Oct dropout](https://kyoon-mit.github.io/projects/bns/updates/2026-10-16-why-dropout-broke-our-s4d/) |
| `s4d_energy_diag/` | CPU test: whether an energy step plus a maximum over time lets one S4D layer learn chirp mass | [8 Oct, section 1](https://kyoon-mit.github.io/projects/bns/updates/2026-10-08-chirp-kernels-work/) |
| `chirp_kernel_eval/` | scores a chirp-kernel checkpoint on the matched-filter benchmark windows; architecture and population figures | [8 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-08-chirp-kernels-work/), and the den-reg population figures of [29 Sep](https://kyoon-mit.github.io/projects/bns/updates/2026-09-29-den-reg-and-project-8/), [5 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-05-recipe-at-epoch-630/) and [14 Oct](https://kyoon-mit.github.io/projects/bns/updates/2026-10-14-den-reg-final/) |

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`.
