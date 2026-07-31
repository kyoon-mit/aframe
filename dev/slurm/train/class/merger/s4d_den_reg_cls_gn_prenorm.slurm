#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=s4d_den_reg_cls_gn_prenorm
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=2-12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls_gn_prenorm/train/class/s4d_den_reg_cls_gn_prenorm_regdev/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls_gn_prenorm/train/class/s4d_den_reg_cls_gn_prenorm_regdev/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/configs/train/class/merger/s4d_den_reg_cls_gn_prenorm.yaml
RUN_DIR=${REPO}/dev/runs/s4d_den_reg_cls_gn_prenorm/train/class/s4d_den_reg_cls_gn_prenorm_regdev

mkdir -p "${RUN_DIR}/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "s4d_den_reg_cls_gn_prenorm_regdev" \
    --trainer.logger.id "s4d_den_reg_cls_gn_prenorm_regdev" \
    --trainer.logger.save_dir "${RUN_DIR}"
