#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=s4d_resnet_groupnorm_reg_test2
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg/slurm_logs/%x_%j.err

VERSION=test2
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_regdev
WANDB_NAME=chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_regdev
RUN_DIR=${REPO}/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_d256s32_ssmlr_resnet_groupnorm.yaml
# resume from latest ckpt of test1
CKPT=${REPO}/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg/chirp_mass_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_regdev_test1/BNS-PUBLICATION/chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_regdev_test1/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
