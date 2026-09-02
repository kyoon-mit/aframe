#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=merger_4s_big_ssmlr_resnet_prenorm_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_prenorm/train/class/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_prenorm/train/class/slurm_logs/%x_%j.err

VERSION=test2

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=chirp_mass_26-30s_d256_s32_l4_ssmlr_resnet_prenorm_regdev
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_prenorm/train/class/${BASE_NAME}_${VERSION}
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_prenorm/train/class/chirp_mass_26-30s_d256_s32_l4_ssmlr_resnet_prenorm_regdev_test1/BNS-PUBLICATION-CLASS/chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_prenorm_regdev_test1/checkpoints/last.ckpt
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/class/merger/merger_4s_d256s32_ssmlr_resnet_prenorm.yaml

mkdir -p "${RUN_DIR}/logs" "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_prenorm/train/class/slurm_logs"
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_prenorm_regdev_${VERSION}" \
    --trainer.logger.id "chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_prenorm_regdev_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
