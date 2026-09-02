#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --job-name=s4d_den_reg_dynmix_smooth
#SBATCH --gres=gpu:1
#SBATCH --mem=63999M
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/merger_4s_den_reg_dynmix_smooth_test1/slurm_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/denoise_regress_4s/train/reg/merger_4s_den_reg_dynmix_smooth_test1/slurm_logs/%x_%j.err

# Fresh (from-scratch) denoise+regress run with the log-|FFT| spectral
# smoothness term (lambda_smooth): matches pred bin-to-bin raggedness to the
# smooth target, on top of the DynamicMixtureLoss time/spectral mix.
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
CONFIG=${REPO}/dev/configs/train/reg/merger/merger_4s_den_reg_dynmix_smooth.yaml

mkdir -p ${REPO}/dev/runs/denoise_regress_4s/train/reg/merger_4s_den_reg_dynmix_smooth_test1/slurm_logs
source ~/.bash_profile
export HDF5_USE_FILE_LOCKING=FALSE
cd ${REPO}/projects/train
source ${REPO}/projects/train/.venv/bin/activate

uv run --no-sync python -m train fit --config ${CONFIG}
