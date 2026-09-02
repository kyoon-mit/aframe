#!/bin/bash
#SBATCH --partition=iaifi_gpu
#SBATCH --job-name=merger_4s_den_reg
#SBATCH --gres=gpu:1
#SBATCH --mem=127998M
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/slurm_logs/%x_%j.err

VERSION=test9
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
BASE_NAME=merger_4s_den_reg_${VERSION}
RUN_DIR=${REPO}/dev/runs/denoise_regress_4s/train/reg/${BASE_NAME}
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_den_reg.yaml

# Resume from the last checkpoint of the previous version.
RESUME_CKPT=${REPO}/dev/runs/denoise_regress_4s/train/reg/merger_4s_den_reg_test8/BNS-PUBLICATION/merger_4s_den_reg_test8/checkpoints/last.ckpt

mkdir -p "${RUN_DIR}/logs" "${REPO}/dev/runs/denoise_regress_4s/train/reg/slurm_logs"
source ~/.bash_profile
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

# batch_size 32 -> 64, max_num_workers 8 -> 16. batches_per_epoch is halved so an
# epoch still covers 8192 samples, matching the 8 x 1024 waveform chunk budget:
# batches_per_chunk = batches_per_epoch // chunks_per_epoch + 1 = 17, and
# 17 * 64 = 1088 samples drawn per 1024-waveform chunk (was 33 * 32 = 1056).
uv run --no-sync python -m train fit --config ${CONFIG} \
    --trainer.default_root_dir "${RUN_DIR}" \
    --trainer.logger.name "${BASE_NAME}" \
    --trainer.logger.id "${BASE_NAME}" \
    --trainer.logger.save_dir "${RUN_DIR}" \
    --data.init_args.batch_size 64 \
    --data.init_args.batches_per_epoch 128 \
    --data.init_args.max_num_workers 16 \
    --ckpt_path "${RESUME_CKPT}"
