#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=linoss_reg_gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/claude_tests/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/claude_tests/slurm_logs/%x_%j.err

# usage: sbatch --job-name=<name> run_linoss_reg_gpu.slurm <self_contained_config.yaml>
CONFIG=$1
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
mkdir -p ${REPO}/dev/runs/claude_tests/slurm_logs
source ~/.bash_profile
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG}
