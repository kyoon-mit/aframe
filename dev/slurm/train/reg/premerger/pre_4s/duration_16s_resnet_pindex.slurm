#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=s4d_pre4s_dur16s_resnet_pindex
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_4s/duration_16s/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_4s/duration_16s/train/reg/slurm_logs/%x_%j.err

# gpu_test verification run. Requires the repo checked out on reg-dev-latest.
# Bump VERSION below before each new run.
VERSION=test4

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=s4d_pre_4s_dur_16s_resnet_pindex
WANDB_NAME=s4d_pre_4s_dur_16s_resnet_pindex
RUN_DIR=${REPO}/dev/runs/premerger/pre_4s/duration_16s/train/reg/${BASE_NAME}_${VERSION}
CONFIG=${REPO}/dev/configs/train/reg/premerger/pre_4s/duration_16s_resnet_pindex.yaml
CKPT=${REPO}/dev/runs/premerger/pre_4s/duration_16s/train/reg/s4d_pre_4s_dur_16s_resnet_pindex_test3/BNS-PUBLICATION/s4d_pre_4s_dur_16s_resnet_pindex_test3/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs"
source ~/.bash_profile

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.id "${WANDB_NAME}_${VERSION}" \
    --trainer.logger.save_dir "${RUN_DIR}"
