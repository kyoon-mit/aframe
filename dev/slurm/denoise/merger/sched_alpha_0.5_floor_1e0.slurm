#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=den_sched_alpha_0.5_floor_1e0
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise/merger/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise/merger/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/configs/denoise/merger/sched_alpha_0.5_floor_1e0.yaml

mkdir -p ${REPO}/dev/runs/denoise/merger/slurm_logs
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG}
