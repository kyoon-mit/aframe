#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=s4d_resnet_reg_test4
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test4/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test4/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/reg/merger/merger_4s_d256s32_ssmlr_resnet.yaml
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test3/BNS-PUBLICATION/s4d_resnet_regdev_test3/checkpoints/last.ckpt
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test4

mkdir -p "${RUN_DIR}/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "s4d_resnet_regdev_test4" \
    --trainer.logger.id "s4d_resnet_regdev_test4" \
    --trainer.logger.save_dir "${RUN_DIR}"
