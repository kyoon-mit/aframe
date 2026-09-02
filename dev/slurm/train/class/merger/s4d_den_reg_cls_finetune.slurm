#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=s4d_den_reg_cls_ft
#SBATCH --gres=gpu:1
#SBATCH --mem=127G
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/s4d_den_reg_cls/train/class/slurm_logs/%x_%j.err

VERSION=regdev_ft
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
NAME=s4d_den_reg_cls_${VERSION}
RUN_DIR=${REPO}/dev/runs/s4d_den_reg_cls/train/class/${NAME}
CONFIG=${REPO}/dev/configs/train/class/merger/s4d_den_reg_cls.yaml
# fine-tune from the BEST auroc ckpt (epoch 1029, AUROC 0.526), weights only
BEST=${REPO}/dev/runs/s4d_den_reg_cls/train/class/s4d_den_reg_cls_regdev/BNS-PUBLICATION-CLASS/s4d_den_reg_cls_regdev/checkpoints/s4d_cls_auroc_1029.ckpt

mkdir -p "${RUN_DIR}" "${REPO}/dev/runs/s4d_den_reg_cls/train/class/slurm_logs"
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

# weight-only warm start (no --ckpt_path) + low, non-restarting lr so the
# 0.526 peak is fine-tuned instead of over-trained away by warm restarts
uv run --no-sync python -m train fit --config ${CONFIG} \
    --model.init_args.warm_start_ckpt "${BEST}" \
    --model.init_args.learning_rate 1e-4 \
    --model.init_args.ssm_lr 1e-3 \
    --model.init_args.lr_scheduler.init_args.warmup_epochs 5 \
    --model.init_args.lr_scheduler.init_args.T_0 5000 \
    --model.init_args.lr_scheduler.init_args.eta_min 1.0e-6 \
    --model.init_args.lr_scheduler.init_args.peak_decay 1.0 \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${NAME}" \
    --trainer.logger.id "${NAME}" \
    --trainer.logger.save_dir "${RUN_DIR}"
