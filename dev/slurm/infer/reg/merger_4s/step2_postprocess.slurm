#!/bin/bash
#SBATCH --job-name=step2_postprocess
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step2_%j.log
#
# Step 2: re-integrate + cluster the cached raw scores. CPU only, so you can
# rerun this with any filter without touching Triton.
#   sbatch step2_postprocess.slurm                       (boxcar, default)
#   sbatch step2_postprocess.slurm gaussian 4            (gaussian, std=4)

set -euo pipefail
INTEGRATION=${1:-boxcar}
GAUSSIAN_STD=${2:-4}

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step2_postprocess.py \
      --results /n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/results \
      --integration "$INTEGRATION" \
      --gaussian-std "$GAUSSIAN_STD"
