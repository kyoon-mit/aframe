#!/bin/bash
#SBATCH --job-name=step1_raw_scores
#SBATCH --partition=shared             # client only; the GPU work happens on the server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-249%6                # 250 branches, 6 at a time (the 80GB GPU can take it)
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step1_%a.log
#
# Step 1: one branch per task. Saves the RAW (un-integrated) scores to
# timeseries.hdf5, plus the usual background/foreground event files.
# Needs step 0 running. Launch both with: bash launcher.sh

set -euo pipefail
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
IP_FILE=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/triton_ip.txt

echo "waiting for Triton to publish $IP_FILE ..."
for i in $(seq 1 720); do [ -s "$IP_FILE" ] && break; sleep 10; done
[ -s "$IP_FILE" ] || { echo "ERROR: Triton never became ready"; exit 1; }
TRITON_IP=$(cat "$IP_FILE")
curl -sf "http://${TRITON_IP}:8000/v2/health/ready" >/dev/null \
    || { echo "ERROR: $TRITON_IP not healthy"; exit 1; }
echo "Triton ready at $TRITON_IP"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBDIR/branches.txt")
BACKGROUND_FILE=$(echo "$LINE" | cut -d' ' -f1)
SHIFTS=$(echo "$LINE" | cut -d' ' -f2)

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step1_raw_scores.py \
      --config /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer/reg/merger_4s_id2.yaml \
      --client.address "${TRITON_IP}:8001" \
      --data.triton_address "${TRITON_IP}:8001" \
      --data.background_fname "$BACKGROUND_FILE" \
      --data.shifts "$SHIFTS" \
      --outdir "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/results/branch_${SLURM_ARRAY_TASK_ID}"
