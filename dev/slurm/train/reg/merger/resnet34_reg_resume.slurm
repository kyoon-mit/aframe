#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=resnet34_reg_test4
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_reg/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_reg/train/reg/slurm_logs/%x_%j.err

VERSION=test4
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=resnet34_reg_regdev
WANDB_NAME=resnet34_reg_regdev
RUN_DIR=${REPO}/dev/runs/resnet34_reg/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/merger/resnet34_reg.yaml
# resume from latest ckpt of test2
CKPT=${REPO}/dev/runs/resnet34_reg/train/reg/resnet34_reg_regdev_test2/BNS-PUBLICATION/resnet34_reg_regdev_test2/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/resnet34_reg/train/reg/slurm_logs"
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
