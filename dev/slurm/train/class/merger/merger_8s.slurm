#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=merger8s_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_8s/train/class/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_8s/train/class/slurm_logs/%x_%j.err

# Bump VERSION below before each new run.
VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_22-30s_d64_s64_l4_regdev
WANDB_NAME=chirp_mass_snr_4_50_22-30s_d64_s64_l4_regdev
RUN_DIR=${REPO}/dev/runs/merger_8s/train/class/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/class/merger/merger_8s.yaml

mkdir -p "${RUN_DIR}/logs"
source ~/.bash_profile

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
