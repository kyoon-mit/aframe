#!/bin/bash
#SBATCH --job-name=merger_4s_class_mopup
#SBATCH --partition=shared             # client only; GPU work is on the server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/merger_4s_class_mopup_%j.log
#
# Mop-up client: walks ALL 250 branches and runs only the ones still missing
# (skip-logic), against a full-node gpu_requeue server (200G host RAM) that can
# handle the 36341s branches the 64G gpu_test servers OOM on. Uses serve slot 5
# so its ports/IP never collide with the main run's slots 0-3.

set -euo pipefail
# branches.txt (the 250 background_file + shifts pairs) lives in the reg dir
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer/class/merger_4s.yaml
IP_FILE=$TS/triton_ip_class_5.txt

echo "mop-up waiting for $IP_FILE ..."
for i in $(seq 1 720); do [ -s "$IP_FILE" ] && break; sleep 10; done
[ -s "$IP_FILE" ] || { echo "ERROR: server never became ready"; exit 1; }
read -r TRITON_IP GRPC_PORT HTTP_PORT < "$IP_FILE"
GRPC_ADDR="${TRITON_IP}:${GRPC_PORT}"
curl -sf "http://${TRITON_IP}:${HTTP_PORT}/v2/health/ready" >/dev/null \
    || { echo "ERROR: $GRPC_ADDR not healthy"; exit 1; }
echo "mop-up server ready at $GRPC_ADDR"

IDLE_GAP=35
NB=$(wc -l < "$JOBDIR/branches.txt")
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
first=1
filled=0
for ((idx = 0; idx < NB; idx++)); do
    OUTDIR=$TS/results_class/branch_${idx}
    if [ -s "$OUTDIR/timeseries.hdf5" ]; then
        continue
    fi
    if [ "$first" -eq 0 ]; then
        sleep "$IDLE_GAP"
    fi
    first=0
    LINE=$(sed -n "$((idx + 1))p" "$JOBDIR/branches.txt")
    BG=$(echo "$LINE" | cut -d' ' -f1)
    SH=$(echo "$LINE" | cut -d' ' -f2)
    echo "mop-up -> branch $idx ($(basename "$BG") $SH)"
    uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step1_raw_scores.py \
        --config "$CONFIG" \
        --client.address "$GRPC_ADDR" \
        --data.triton_address "$GRPC_ADDR" \
        --data.background_fname "$BG" \
        --data.shifts "$SH" \
        --outdir "$OUTDIR"
    filled=$((filled + 1))
done
echo "mop-up complete: filled $filled branches"
