#!/bin/bash
# Launch one serve+client pair for the classification SV run:
#   bash launch.sh <index> <total> [partition]
# <total> is the number of servers; each client runs the branch indices with
# index %% total, serially, against its own server. [partition] defaults to
# gpu_test; pass gpu_requeue to put this server on a full node instead.
#
# Mixed example (2 gpu_test + 2 gpu_requeue = 4 servers):
#   bash launch.sh 0 4 gpu_test
#   bash launch.sh 1 4 gpu_test
#   bash launch.sh 2 4 gpu_requeue
#   bash launch.sh 3 4 gpu_requeue
set -euo pipefail
SHARD=${1:?index}
NSHARD=${2:?total}
PARTITION=${3:-gpu_test}
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/class/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

rm -f "$TS/triton_ip_class_${SHARD}.txt"   # never read a dead server's IP

# The serve file defaults to gpu_test (64G cap, 12h). For gpu_requeue, override
# to a full node (200G host RAM, compatible-arch constraint, preemptible).
if [ "$PARTITION" = "gpu_requeue" ]; then
    OVERRIDE=(--partition=gpu_requeue --mem=200G
              --constraint="cc7.0|cc8.0" --time=08:00:00 --requeue)
else
    OVERRIDE=(--partition="$PARTITION")
fi

SERVE_ID=$(sbatch --parsable "${OVERRIDE[@]}" \
    --export=ALL,SHARD=$SHARD \
    "$JOBDIR/step0_serve.slurm")
CLIENT_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" \
    --export=ALL,SHARD=$SHARD,NSHARD=$NSHARD \
    "$JOBDIR/step1_client.slurm")
echo "class $SHARD/$NSHARD on $PARTITION: serve=$SERVE_ID client=$CLIENT_ID"
