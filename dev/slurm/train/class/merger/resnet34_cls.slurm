#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=resnet34_cls
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_cls/train/class/resnet34_cls_test3/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_cls/train/class/resnet34_cls_test3/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_cls/train/class/resnet34_cls_test2/BNS-PUBLICATION-CLASS/resnet34_cls_test2/checkpoints/last.ckpt
CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/train/class/merger/resnet34_cls.yaml
RUN_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/resnet34_cls/train/class/resnet34_cls_test3

mkdir -p "${RUN_DIR}/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE

cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG} \
    --ckpt_path ${CKPT} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "resnet34_cls_test3" \
    --trainer.logger.id "resnet34_cls_test3" \
    --trainer.logger.save_dir "${RUN_DIR}"
