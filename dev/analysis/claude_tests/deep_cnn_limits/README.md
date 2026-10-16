# Deep CNN limits

These scripts compare a deep CNN with 3-sample kernels (the aframe ResNet)
with one layer of 4 s kernels (free kernels) for chirp-mass estimation. They
compute the ResNet's receptive field, how much of a weak signal survives one
nonlinearity before an in-step sum, what a network that treats every time
alike can see of a weak signal, how much the chirp-mass Cramer-Rao bound
grows when phase is kept only inside 0.44 s segments, and a simulation of
coherent, segmented and time-averaged matched filters. `adv_verify/` re-checks
each part with its own code.

## What it backs

- The lecture of 15 October 2026,
  [Oscillators and chirps](https://kyoon-mit.github.io/projects/bns/updates/2026-10-15-oscillators-and-chirps/),
  section 5: a GELU before an in-step sum keeps 84 to 91% of the SNR, a ReLU
  86%, a threshold at zero 80%. Scripts: `a_nonlinearity.py`,
  `a_gelu_closed_form.py`, `nonlin_check.py`, `side_checks.py`,
  `adv_verify/v_thmA.py`. Also the correction of 16 October in
  [#semi-coherent](https://kyoon-mit.github.io/projects/bns/derivations/#semi-coherent).
- The experiment log entry of 2026-10-15 19:35 (local only, not in the
  repo), the deep-CNN analysis:
  - receptive field: `erf_rf/rf_resnet.py`, `erf_rf/rf_longconv.py`,
    `thm_rf_resnet.py`, `rf_check.py`, `adv_verify/v_rf.py`;
    `side_checks.py` also gives the stride-32 average of a tone
  - networks that treat every time alike: `thm_signals.py` (test signals),
    `b_bounds.py`, `b_cnn_mc.py`, `b_checks_extra.py`, `b_reverse.py`,
    `b_steep_mean.py`, `b_cnn_table.py`, `adv_verify/v_thm3.py`,
    `adv_verify/v_edge.py`, `adv_verify/v_edge2.py`, `adv_verify/v_prec.py`,
    `adv_verify/v_freekernel.py`, `adv_verify/v_freekernel_even.py`
  - phase kept inside segments: `common.py`, `fisher_seg.py`,
    `detection.py`, `mc_ml.py`, `mc_pop.py`, `pop_summary.py`, `popavg.py`,
    `analyze_mc.py`, `discrete_check.py`, `small_R.py`, `extra_checks.py`,
    `tables.py`, `adv_verify/v_fisher.py`, `adv_verify/v_mlmc.py`,
    `adv_verify/v_snrwindow.py`
  - simulation: `sim.py`, `sim_runs.sh`, `adv_verify/v_simwidth.py`,
    `adv_verify/v_simrepro.py`

The top-level `rf_common.py`, `rf_longconv.py` and `side_checks.py` are the
same files as those in `erf_rf/`.

## How it was run

Interpreter: `/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python`,
some scripts with `-I`. Working directory: this folder for the top-level
scripts and `adv_verify/` for its own. Outputs go to `out/`, `mc/` and
`sim_out/` under the working directory, or to the absolute paths in the
scripts. `sim_runs.sh` runs `sim.py` over the SNR bands. `thm_signals.py`
needs LALSimulation.

The scripts are copied unchanged from the session scratch directory
`/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof`.
Absolute paths inside several of them still point there.

## Data it reads (local only, not in the repo)

- average O3 PSD:
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz`
- checkpoints in
  `/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/<run>/CLAUDE-TESTS/<run>/checkpoints/`
  of `aframe_resnet1d_3463_softmax_train_snr8_prob0.96` (`ck_244`, `last`),
  `free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96` (`ck_248`),
  `free_kernel_taps_energy_max_256ch_softmax_train_snr8_prob0.96` (`ck_133`),
  `s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96` (`ck_266`),
  `s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96` (`ck_288`),
  `s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96`
  (`ck_122`)
- in the scratch directory: the test signals `out/sig_newtonian.npy` and
  `out/sig_whitened_bns.npy` (written by `thm_signals.py`), and the saved
  outputs in `out/`, `mc/`, `sim_out/` and `mc_test.json`
