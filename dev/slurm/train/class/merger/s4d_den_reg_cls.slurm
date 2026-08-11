#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=s4d_den_reg_cls_best1029
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_best1029/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_best1029/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/configs/train/class/merger/s4d_den_reg_cls.yaml
NAME=s4d_den_reg_cls_best1029
RUN_DIR=${REPO}/dev/runs/s4d_den_reg_cls/train/class/${NAME}
# resume (full state) from the BEST auroc ckpt (epoch 1029, AUROC 0.526)
CKPT=${REPO}/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_regdev/BNS-PUBLICATION-CLASS/s4d_den_reg_cls_regdev/checkpoints/s4d_cls_auroc_1029.ckpt

mkdir -p "${RUN_DIR}/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${NAME}" \
    --trainer.logger.id "${NAME}" \
    --trainer.logger.save_dir "${RUN_DIR}"
