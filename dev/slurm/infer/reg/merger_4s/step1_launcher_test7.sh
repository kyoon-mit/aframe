#!/bin/bash
# Launches STEP 0 (Triton server) and STEP 1 (branch array) for the test7 model.
#
# Step 1 needs a live server, so submitting them by hand in the wrong order --
# or against a dead server's stale IP -- is the classic way this breaks. This
# clears the old IP, submits both, and makes step 1 wait for step 0.
#
#   bash step1_launcher_test7.sh
set -euo pipefail

JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
LOGS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
IP_FILE=$TS/triton_ip_test7.txt

cd "$JOBDIR"
[ -s branches.txt ] || {
    echo "branches.txt missing -- run: bash make_branches.sh" >&2
    exit 1
}
NUM_BRANCHES=$(wc -l < branches.txt)
mkdir -p "$LOGS"

# never let step 1 connect to a dead server's address
rm -f "$IP_FILE"

echo "launching STEP 0 (Triton server) and STEP 1 (${NUM_BRANCHES} branches)"
echo

SERVE_ID=$(sbatch --parsable step0_serve_test7.slurm)
ARRAY_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" \
    step1_raw_scores_test7.slurm)

echo "  STEP 0  serve       = $SERVE_ID   (gpu, full A100 80GB, 200G host)"
echo "  STEP 1  raw scores  = $ARRAY_ID   (${NUM_BRANCHES} branches, serial)"
echo
echo "  results -> $TS/results_test7/branch_*"
echo "  logs    -> $LOGS"
echo "  after STEP 1 finishes, stop the server:  scancel $SERVE_ID"
