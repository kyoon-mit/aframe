#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=linoss_merger_4s_d256s32_reg
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32/train/reg/slurm_logs/%x_%j.err

VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_26-30s_d256_s32_l4_linoss_regdev
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32/train/reg/${BASE_NAME}_${VERSION}
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/reg/merger/linoss_merger_4s_d256s32.yaml

mkdir -p "${RUN_DIR}/logs" "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/linoss_merger_4s_d256s32/train/reg/slurm_logs"
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "chirp_mass_snr_4_50_26-30s_d256_s32_l4_linoss_regdev_${VERSION}" \
    --trainer.logger.id "chirp_mass_snr_4_50_26-30s_d256_s32_l4_linoss_regdev_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
