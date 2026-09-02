#!/bin/bash
#SBATCH --job-name=step0_serve
#SBATCH --partition=gpu               # full A100 80GB, dedicated node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step0_serve_%j.log
#
# Step 0: serve the exported model with Triton. Publishes its IP only once the
# server is READY -- that file is the go-signal step 1 waits on.

set -euo pipefail
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/serve.py
