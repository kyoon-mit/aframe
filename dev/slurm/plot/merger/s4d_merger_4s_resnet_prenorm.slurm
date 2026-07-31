#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=s4d_merger_4s_resnet_prenorm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/test_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUN_CONFIG=${REPO}/dev/runs/merger_4s_big_s4d_resnet/train/reg/s4d_resnet_regdev_test4/config.yaml
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/tmp/s4d_resnet_test4_last.ckpt
OUT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/test_csv/merger

mkdir -p /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_big_s4d_resnet/test_logs
source ~/.bash_profile
set -eo pipefail
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

for DIST in powerlaw uniform; do
    uv run --no-sync python -m train test \
        --config ${RUN_CONFIG} \
        --config ${REPO}/dev/configs/train/test/merger/s4d_resnet_test4_${DIST}.yaml \
        --ckpt_path ${CKPT}
done
