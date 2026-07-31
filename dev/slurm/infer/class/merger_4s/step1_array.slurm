#!/bin/bash
#SBATCH --job-name=merger_4s_class_client
#SBATCH --partition=shared             # client only; GPU work is on the server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/merger_4s_class_client_%A_%a.log
#
# Array client, exactly like the regression step1_raw_scores.slurm: one branch
# per array task (branch index = SLURM_ARRAY_TASK_ID), %1 serial. Each task is a
# fresh process against this shard's server. Launched with a strided --array
# (e.g. 0-249:4) so shard $SHARD owns indices SHARD, SHARD+NSHARD, ...
# Skips branches already on disk.

set -euo pipefail
: "${SHARD:?set SHARD}"
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
CONFIG=${CONFIG:-/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer/class/merger_4s.yaml}
IP_FILE=$TS/triton_ip_${TAG:-class}_${SHARD}.txt

idx=$SLURM_ARRAY_TASK_ID
OUTDIR=$TS/${RESULTS:-results_class}/branch_${idx}
if [ -s "$OUTDIR/timeseries.hdf5" ]; then
    echo "branch $idx already done -- skip"
    exit 0
fi

echo "shard $SHARD waiting for $IP_FILE ..."
for i in $(seq 1 720); do [ -s "$IP_FILE" ] && break; sleep 10; done
[ -s "$IP_FILE" ] || { echo "ERROR: server never became ready"; exit 1; }
read -r TRITON_IP GRPC_PORT HTTP_PORT < "$IP_FILE"
GRPC_ADDR="${TRITON_IP}:${GRPC_PORT}"
curl -sf "http://${TRITON_IP}:${HTTP_PORT}/v2/health/ready" >/dev/null \
    || { echo "ERROR: $GRPC_ADDR not healthy"; exit 1; }

LINE=$(sed -n "$((idx + 1))p" "$JOBDIR/branches.txt")
BG=$(echo "$LINE" | cut -d' ' -f1)
SH=$(echo "$LINE" | cut -d' ' -f2)
echo "shard $SHARD -> branch $idx ($(basename "$BG") $SH) on $GRPC_ADDR"

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step1_raw_scores.py \
    --config "$CONFIG" \
    --client.address "$GRPC_ADDR" \
    --data.triton_address "$GRPC_ADDR" \
    --data.background_fname "$BG" \
    --data.shifts "$SH" \
    --outdir "$OUTDIR"
