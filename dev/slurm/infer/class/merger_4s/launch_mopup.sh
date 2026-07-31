#!/bin/bash
# Mop-up pass: fill whatever branches are still missing from results_class,
# on a full-node gpu_requeue server (200G) that survives the 36341s branches.
# Safe to run after (or alongside) the main run -- it uses serve slot 5, so its
# ports/IP don't touch the main run's slots 0-3, and skip-logic means it only
# does the gaps.
#
#   bash launch_mopup.sh
set -euo pipefail
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/class/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

rm -f "$TS/triton_ip_class_5.txt"
# clear the reused server log so a monitor can't match a PREVIOUS run's
# "failed to load" line and cancel this healthy server (that bit us once)
rm -f /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/server_class_5.log

# --exclude known-bad nodes: 0705 deadlocked at sequence-end; 0709 and
# 8a25404 fail to load models with CUDA error 999 ("unable to get number of
# CUDA devices") -- broken GPU/driver state on those nodes.
SERVE_ID=$(sbatch --parsable \
    --partition=gpu_requeue --mem=200G --constraint="cc7.0|cc8.0" \
    --exclude=holygpu7c0705,holygpu7c0709,holygpu8a25404 \
    --time=08:00:00 --requeue \
    --export=ALL,SHARD=5 \
    "$JOBDIR/step0_serve.slurm")
CLIENT_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" \
    "$JOBDIR/step1_mopup.slurm")
echo "mop-up: serve=$SERVE_ID (gpu_requeue, slot 5) client=$CLIENT_ID"
