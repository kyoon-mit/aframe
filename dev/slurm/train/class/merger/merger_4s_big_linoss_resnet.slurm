#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=big_linoss_resnet_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/train/class/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/train/class/slurm_logs/%x_%j.err

VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=linoss_resnet_regdev
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/train/class/${BASE_NAME}_${VERSION}
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/class/merger/merger_4s_big_linoss_resnet.yaml

mkdir -p "${RUN_DIR}/logs" "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/train/class/slurm_logs"
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${BASE_NAME}_${VERSION}" \
    --trainer.logger.id "${BASE_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
