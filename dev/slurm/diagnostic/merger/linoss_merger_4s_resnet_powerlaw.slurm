#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=test_linoss_merger_4s_resnet_powerlaw
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/test_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUN_CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/train/reg/linoss_resnet_regdev_test1/config.yaml
TEST_CONFIG=${REPO}/dev/configs/train/test/merger/linoss_merger_4s_resnet_powerlaw.yaml

mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_linoss_resnet/test_logs
source ~/.bash_profile
set -eo pipefail
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

uv run --no-sync python -m train test \
    --config ${RUN_CONFIG} \
    --config ${TEST_CONFIG}
