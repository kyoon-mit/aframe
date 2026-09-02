#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=s4d_resnet_reg_test5
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/slurm_logs/%x_%j.err

VERSION=test5
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=s4d_resnet_regdev
RUN_DIR=${REPO}/dev/runs/merger_4s_big_s4d_resnet/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_d256s32_ssmlr_resnet.yaml
# resume from latest ckpt of test4
CKPT=${REPO}/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test4/BNS-PUBLICATION/s4d_resnet_regdev_test4/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/merger_4s_big_s4d_resnet/train/reg/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${BASE_NAME}_${VERSION}" \
    --trainer.logger.id "${BASE_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
