#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=test_merger_4s_ssm_lr_powerlaw
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_ssm_lr/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_ssm_lr/test_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUN_CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_ssm_lr/train/reg/chirp_mass_60-64s_d64_s64_l4_ssm_lr_regdev_test4/config.yaml
TEST_CONFIG=${REPO}/dev/configs/train/test/merger/merger_4s_ssm_lr_powerlaw.yaml
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_ssm_lr/train/reg/chirp_mass_60-64s_d64_s64_l4_ssm_lr_regdev_test4/BNS-PUBLICATION/chirp_mass_snr_4_50_60-64s_d64_s64_l4_ssm_lr_regdev_test4/checkpoints/last.ckpt

mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_ssm_lr/test_logs
source ~/.bash_profile
set -eo pipefail
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train test \
    --config ${RUN_CONFIG} \
    --config ${TEST_CONFIG} \
    --ckpt_path ${CKPT}
