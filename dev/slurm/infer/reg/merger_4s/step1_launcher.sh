#!/bin/bash
# Launches STEP 0 (Triton server) and STEP 1 (the branch array), wired together.
#
# Step 1 needs a live server, so submitting them by hand in the wrong order --
# or against a dead server's stale IP -- is the classic way this breaks. This
# clears the old IP, submits both, and makes step 1 wait for step 0.
#
#   bash launcher.sh
set -euo pipefail

JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
LOGS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer
IP_FILE=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/triton_ip.txt

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

SERVE_ID=$(sbatch --parsable step0_serve.slurm)
ARRAY_ID=$(sbatch --parsable --dependency=after:"$SERVE_ID" step1_raw_scores.slurm)

echo "  STEP 0  serve        = $SERVE_ID   (iaifi_gpu, full A100 80GB)"
echo "  STEP 1  raw scores   = $ARRAY_ID   (${NUM_BRANCHES} branches, 6 at a time)"
echo
echo "  logs  -> $LOGS"
echo "  after STEP 1 finishes, stop the server:  scancel $SERVE_ID"
echo "  then: sbatch step2_postprocess.slurm [gaussian 4]"
