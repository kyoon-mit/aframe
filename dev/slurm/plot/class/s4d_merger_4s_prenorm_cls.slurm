#!/bin/bash
#SBATCH --partition=gpu_requeue
#SBATCH --job-name=s4d_merger_4s_prenorm_cls
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M
#SBATCH --time=00:30:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_prenorm/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_prenorm/test_logs/%x_%j.err

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUN_CONFIG=${REPO}/dev/runs/merger_4s_d256s32_ssmlr_prenorm/train/class/chirp_mass_26-30s_d256_s32_l4_ssmlr_prenorm_regdev_test2/config.yaml
CKPT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/tmp/s4d_prenorm_cls_test2_last.ckpt

mkdir -p ${REPO}/dev/runs/merger_4s_d256s32_ssmlr_prenorm/test_logs
source ~/.bash_profile
set -eo pipefail
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

for DIST in powerlaw uniform; do
    uv run --no-sync python -m train test \
        --config ${RUN_CONFIG} \
        --config ${REPO}/dev/configs/train/test/class/prenorm_cls_${DIST}.yaml \
        --ckpt_path ${CKPT}
done
