#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=s4d_den_reg_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/configs/claude_tests/merger_4s_den_reg_gputest.yaml

mkdir -p ${REPO}/dev/runs/denoise_regress_4s/train/reg/slurm_logs
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG}
