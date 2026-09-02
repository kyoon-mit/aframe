#!/bin/bash
# One gpu server + a strided %1 array of per-branch CPU tasks (the regression
# infer design). Run once per shard to spread across more gpu servers:
#   bash launch_array.sh <shard> <nshard> [partition]
# Shard k owns branch indices k, k+nshard, ... via --array=k-249:nshard%1.
set -euo pipefail
SHARD=${1:?shard}
NSHARD=${2:?nshard}
PARTITION=${3:-gpu_test}
EXCLUDE=${4:-}   # optional node(s) to keep this server off (comma-separated)
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/class/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

rm -f "$TS/triton_ip_class_${SHARD}.txt"

# nodes with broken GPU state / deadlocks observed this run
BAD_NODES=holygpu7c0705,holygpu7c0709,holygpu7c0715,holygpu8a25404
if [ "$PARTITION" = "gpu_requeue" ]; then
    EXCL="$BAD_NODES"; [ -n "$EXCLUDE" ] && EXCL="$EXCL,$EXCLUDE"
    OVERRIDE=(--partition=gpu_requeue --mem=200G --constraint="cc7.0|cc8.0"
              --exclude="$EXCL" --time=08:00:00 --requeue)
else
    OVERRIDE=(--partition="$PARTITION")
    [ -n "$EXCLUDE" ] && OVERRIDE+=(--exclude="$EXCLUDE")
fi

SERVE_ID=$(sbatch --parsable "${OVERRIDE[@]}" \
    --export=ALL,SHARD=$SHARD \
    "$JOBDIR/step0_serve.slurm")
ARRAY_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" \
    --array=${SHARD}-249:${NSHARD}%1 \
    --export=ALL,SHARD=$SHARD \
    "$JOBDIR/step1_array.slurm")
echo "shard $SHARD/$NSHARD on $PARTITION: serve=$SERVE_ID array=$ARRAY_ID"
