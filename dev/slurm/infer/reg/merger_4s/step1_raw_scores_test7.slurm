#!/bin/bash
#SBATCH --job-name=step1_test7
#SBATCH --partition=shared             # client only; the GPU work is on the server
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-249%1                # 250 branches; serial (%1) -- S4D batch-128 fills 80GB, no concurrency
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step1_test7_%a.log
#
# Step 1 (test7): one branch per task. Saves the RAW (un-integrated) scores to
# timeseries.hdf5, plus the usual background/foreground event files.
# Needs step 0 running. Launch both with: bash step1_launcher_test7.sh

set -euo pipefail
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
IP_FILE=$TS/triton_ip_test7.txt

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

# Resume support: the 12h gpu_test wall can cut the array off partway (the
# background files vary 40x in length, so ~250 branches need ~18h on a MIG
# slice). Branches are independent, so a relaunch only needs the gaps.
OUTDIR=$TS/results_test7/branch_${SLURM_ARRAY_TASK_ID}
if [ -s "$OUTDIR/timeseries.hdf5" ]; then
    echo "branch ${SLURM_ARRAY_TASK_ID} already done -- skipping"
    exit 0
fi

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step1_raw_scores.py \
      --config /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer/reg/merger_4s_test7.yaml \
      --client.address "${TRITON_IP}:8001" \
      --data.triton_address "${TRITON_IP}:8001" \
      --data.background_fname "$BACKGROUND_FILE" \
      --data.shifts "$SHIFTS" \
      --outdir "$OUTDIR"
