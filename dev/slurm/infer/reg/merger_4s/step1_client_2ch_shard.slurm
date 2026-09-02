#!/bin/bash
#SBATCH --job-name=client2ch
#SBATCH --partition=shared             # client only; GPU work is on the server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/client2ch_%j.log
#
# Serial client for shard $SHARD of $NSHARD: runs every branch index with
# index %% NSHARD == SHARD, one at a time, against that shard's server. Stride
# sharding spreads the long background files across shards for balanced load.
# Skips branches already on disk, so a preempted/resumed run only fills gaps.

set -euo pipefail
: "${SHARD:?set SHARD}"
: "${NSHARD:?set NSHARD}"
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer/reg/merger_4s_test7.yaml
IP_FILE=$TS/triton_ip_2ch_shard${SHARD}.txt

echo "shard $SHARD/$NSHARD waiting for $IP_FILE ..."
for i in $(seq 1 720); do [ -s "$IP_FILE" ] && break; sleep 10; done
[ -s "$IP_FILE" ] || { echo "ERROR: server never became ready"; exit 1; }
# ip file holds: "<ip> <grpc_port> <http_port>"
read -r TRITON_IP GRPC_PORT HTTP_PORT < "$IP_FILE"
GRPC_ADDR="${TRITON_IP}:${GRPC_PORT}"
curl -sf "http://${TRITON_IP}:${HTTP_PORT}/v2/health/ready" >/dev/null \
    || { echo "ERROR: $GRPC_ADDR not healthy"; exit 1; }
echo "shard $SHARD server ready at $GRPC_ADDR"

# The snapshotter is stateful with a 30s idle timeout. Two branches closer
# together than that share the server before the first sequence's slot evicts,
# and the second branch's START request collides ("must specify the START
# flag") -- a timing race that killed some shards. Force a >30s gap between
# branches so the previous sequence always evicts first. Deterministic, and it
# needs no server restart (the running gpu_requeue servers are left alone).
IDLE_GAP=35
NB=$(wc -l < "$JOBDIR/branches.txt")
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
first=1
for ((idx = SHARD; idx < NB; idx += NSHARD)); do
    OUTDIR=$TS/results_test7_2ch/branch_${idx}
    if [ -s "$OUTDIR/timeseries.hdf5" ]; then
        echo "branch $idx already done -- skip"
        continue
    fi
    if [ "$first" -eq 0 ]; then
        sleep "$IDLE_GAP"   # let the previous sequence's slot evict
    fi
    first=0
    LINE=$(sed -n "$((idx + 1))p" "$JOBDIR/branches.txt")
    BG=$(echo "$LINE" | cut -d' ' -f1)
    SH=$(echo "$LINE" | cut -d' ' -f2)
    echo "shard $SHARD -> branch $idx ($(basename "$BG") $SH)"
    uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step1_raw_scores.py \
        --config "$CONFIG" \
        --client.address "$GRPC_ADDR" \
        --data.triton_address "$GRPC_ADDR" \
        --data.background_fname "$BG" \
        --data.shifts "$SH" \
        --outdir "$OUTDIR"
done
echo "shard $SHARD complete"
