#!/bin/bash
# Launch one serve+client shard pair for the 2-channel run:
#   bash launch_shard_2ch.sh <shard_index> <nshard>
# The client waits (via the IP file) for its own server, then runs its stride
# of branches serially. Run this once per shard to fan out across gpu_requeue.
set -euo pipefail
SHARD=${1:?shard index}
NSHARD=${2:?total shards}
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

rm -f "$TS/triton_ip_2ch_shard${SHARD}.txt"   # never read a dead server's IP

SERVE_ID=$(sbatch --parsable \
    --export=ALL,SHARD=$SHARD \
    "$JOBDIR/step0_serve_2ch_shard.slurm")
CLIENT_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" \
    --export=ALL,SHARD=$SHARD,NSHARD=$NSHARD \
    "$JOBDIR/step1_client_2ch_shard.slurm")
echo "shard $SHARD/$NSHARD: serve=$SERVE_ID client=$CLIENT_ID"
