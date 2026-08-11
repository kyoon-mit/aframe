#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=merger4s_d1024s64_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema/train/class/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema/train/class/slurm_logs/%x_%j.err

VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema_gn
WANDB_NAME=merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema_gn
RUN_DIR=${REPO}/dev/runs/merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema/train/class/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/class/merger/merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema_gn.yaml
# CKPT_PATH=${REPO}/dev/runs/merger_4s_d1024s64n4_ssmlr_resnet_prenorm_ema/train/class/${BASE_NAME}_test1/BNS-PUBLICATION-CLASS/${WANDB_NAME}_test1/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs"
source ~/.bash_profile

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
    # --ckpt_path "${CKPT_PATH}"
