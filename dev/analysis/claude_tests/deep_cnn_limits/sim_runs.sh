#!/bin/bash
# Runs for sim.md: four SNR bands, fixed SNR 25, noise-free, and the template-spacing check.
cd /tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof
PY=/home/kyoon/SSM-BNS/aframe/projects/train/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
run() { /usr/bin/time -v $PY -I sim.py "$@" 2>&1 | grep -vE "^\s+(Average|Exit|Socket|Signals|Page|File|Swaps|Volun|Invol|Minor|Major|Percent|System|User|Command)" ; }
run run 8 10 2000 801 sim_out/band_8_10.npz      > sim_out/band_8_10.log
run run 10 12 2000 802 sim_out/band_10_12.npz    > sim_out/band_10_12.log
run run 12 16 2000 803 sim_out/band_12_16.npz    > sim_out/band_12_16.log
run run 16 25 2000 804 sim_out/band_16_25.npz    > sim_out/band_16_25.log
run run 25 25 2000 805 sim_out/snr25.npz         > sim_out/snr25.log
run noisefree 2000 806 sim_out/noisefree.npz     > sim_out/noisefree.log
# template spacing check: same injections as the 8-10 and 10-12 bands, ratio 1.001, coherent and 0.44 s segments
run run 8 10 2000 801 sim_out/fine_8_10.npz 1.001 0.44   > sim_out/fine_8_10.log
run run 10 12 2000 802 sim_out/fine_10_12.npz 1.001 0.44 > sim_out/fine_10_12.log
echo ALLDONE > sim_out/ALLDONE
