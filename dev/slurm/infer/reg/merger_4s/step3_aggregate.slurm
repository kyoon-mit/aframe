#!/bin/bash
#SBATCH --job-name=step3_aggregate
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step3_%j.log
#
# Step 3: combine all branches into one background + foreground, summing Tb.

set -euo pipefail
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step3_aggregate.py --results /n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/results
