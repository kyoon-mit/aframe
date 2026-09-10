#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=den_no_curr_1e-3_prob0.5
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoiser_4s/train/merger/fix_loss_grad/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoiser_4s/train/merger/fix_loss_grad/slurm_logs/%x_%j.err

# The no_curriculum 1e-3 variant with waveform_prob dropped to 0.5, so half
# the rows are background only and the denoiser is actually asked to output
# silence on noise. Nothing else differs from that config.

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/local/train/denoiser/merger/fix_loss_grad/s4d_den_no_curriculum_1e-3_prob0.5_floor_alpha_track.yaml

mkdir -p ${REPO}/dev/runs/denoiser_4s/train/merger/fix_loss_grad/slurm_logs
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

# 8 cpus on this allocation, and the train and validation workers are both
# alive at the handover, so keep the count well under the core count
uv run --no-sync python -m train fit \
  --config ${CONFIG} \
  --data.init_args.max_num_workers=4
