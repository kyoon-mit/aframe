#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=test_pre_8s_uniform
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/test_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUN_CONFIG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_8s/duration_4s/train/reg/chirp_mass_18-22s_d64_s64_l4_regdev_test2/config.yaml
TEST_CONFIG=${REPO}/dev/configs/train/test/premerger/pre_8s_uniform.yaml
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/pre_8s/duration_4s/train/reg/chirp_mass_18-22s_d64_s64_l4_regdev_test2/BNS-PUBLICATION/chirp_mass_snr_4_50_18-22s_d64_s64_l4_regdev_test2/checkpoints/s4d_chirp_mass_mse_01.ckpt

mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/premerger/test_logs
source ~/.bash_profile
set -eo pipefail
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train test \
    --config ${RUN_CONFIG} \
    --config ${TEST_CONFIG} \
    --ckpt_path ${CKPT}
