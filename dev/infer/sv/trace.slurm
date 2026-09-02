#!/bin/bash
# Trace a checkpoint into a Triton repo's model.pt ON gpu_test. Arch, dims and
# served outputs are read straight from the run's config.yaml (--config), so
# nothing has to be passed by hand for any model type. Required env: CKPT CONFIG OUT.
#SBATCH --job-name=sv_trace
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/sv_trace_%j.log
set -euo pipefail
: "${CKPT:?}"; : "${CONFIG:?}"; : "${OUT:?}"
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/train
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/trace_model_gpu.py \
  --weights "$CKPT" --config "$CONFIG" --out "$OUT"
