#!/bin/bash
#SBATCH --partition=gpu_h200
#SBATCH --job-name=regression_s4d
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_s4d/logs/slurm_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/runs/regression_s4d/logs/slurm_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss
BACKGROUND_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3b_H1_L1_4096Hz
RUN_DIR=${REPO}/runs/regression_s4d
CONFIG_PATH=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss/projects/train/configs/ai4gw/chirp_mass_1s_d64_s64_l4.yaml

mkdir -p "${RUN_DIR}/logs"

source ~/.bash_profile

export WANDB_DIR="${RUN_DIR}"

cd ${REPO}/projects/train
uv run --no-sync python -m train.regression_cli fit \
    --config $CONFIG_PATH \
    --data.init_args.background_dir "${BACKGROUND_DIR}" \
    --trainer.default_root_dir "${RUN_DIR}"
