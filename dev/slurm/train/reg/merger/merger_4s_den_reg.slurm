#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=merger_4s_den_reg
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.err

VERSION=test6
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=merger_4s_den_reg_${VERSION}
RUN_DIR=${REPO}/dev/runs/denoise_regress_4s/train/reg/${BASE_NAME}
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_den_reg.yaml

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/denoise_regress_4s/train/reg/slurm_logs"
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${BASE_NAME}" \
    --trainer.logger.id "${BASE_NAME}" \
    --trainer.logger.save_dir "${RUN_DIR}"
