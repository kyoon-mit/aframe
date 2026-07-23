#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=s4d_pre4s_dur4s_resnet
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_4s/duration_4s/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_4s/duration_4s/train/reg/slurm_logs/%x_%j.err

# gpu_test verification run. Requires the repo checked out on reg-dev-latest.
# Bump VERSION below before each new run.
VERSION=test1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=s4d_pre_4s_dur_4s_resnet
WANDB_NAME=s4d_pre_4s_dur_4s_resnet
RUN_DIR=${REPO}/dev/runs/premerger/pre_4s/duration_4s/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/premerger/pre_4s/duration_4s_resnet.yaml
# CKPT_PATH=${REPO}/dev/runs/premerger/pre_4s/duration_4s/train/reg/${BASE_NAME}_test2/BNS-PUBLICATION/${WANDB_NAME}_test2/checkpoints/last.ckpt

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
