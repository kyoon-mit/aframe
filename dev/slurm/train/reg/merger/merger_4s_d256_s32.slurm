#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=merger4s_d256s32_reg
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=2-12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256_s32/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256_s32/train/reg/slurm_logs/%x_%j.err

# gpu_test verification run. Requires the repo checked out on reg-dev-latest.
# Bump VERSION below before each new run.
VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_60-64s_d256_s32_l4_regdev
WANDB_NAME=chirp_mass_snr_4_50_60-64s_d256_s32_l4_regdev
RUN_DIR=${REPO}/dev/runs/merger_4s_d256_s32/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_d256_s32.yaml

mkdir -p "${RUN_DIR}/logs"
source ~/.bash_profile

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
