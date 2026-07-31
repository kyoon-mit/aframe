#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=s4d_den_reg_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_test2/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_test2/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/class/merger/s4d_den_reg_cls.yaml
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_test2

mkdir -p "${RUN_DIR}/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "s4d_den_reg_cls_test2" \
    --trainer.logger.id "s4d_den_reg_cls_test2" \
    --trainer.logger.save_dir "${RUN_DIR}"
