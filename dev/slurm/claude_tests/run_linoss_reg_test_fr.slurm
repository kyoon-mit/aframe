#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=linoss_reg_test
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/claude_tests/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/claude_tests/slurm_logs/%x_%j.err

# usage: sbatch run_linoss_reg_test.slurm <model_config.yaml>
MODEL_CONFIG=$1

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
DATA_CONFIG=${REPO}/dev/configs/claude_tests/_data_linoss_4s_fullres.yaml

mkdir -p ${REPO}/dev/runs/claude_tests/slurm_logs
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export HDF5_USE_FILE_LOCKING=FALSE

uv run --no-sync python -m train fit \
    --config ${DATA_CONFIG} \
    --config ${MODEL_CONFIG}
