#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=linoss_resnet_groupnorm_reg
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=2-12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32_resnet_groupnorm/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32_resnet_groupnorm/train/reg/slurm_logs/%x_%j.err

VERSION=test2

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_26-30s_d256_s32_l4_linoss_resnet_groupnorm_regdev
WANDB_NAME=chirp_mass_snr_4_50_26-30s_d256_s32_l4_linoss_resnet_groupnorm_regdev
RUN_DIR=${REPO}/dev/runs/linoss_merger_4s_d256s32_resnet_groupnorm/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/merger/linoss_merger_4s_d256s32_resnet_groupnorm.yaml

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/linoss_merger_4s_d256s32_resnet_groupnorm/train/reg/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
