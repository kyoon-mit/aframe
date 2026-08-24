#!/bin/bash
#SBATCH --job-name=rawcsv_groupnorm
#SBATCH --partition=gpu_test          # A100 MIG 20GB is plenty for this model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/test_logs/%x_%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/test_logs/%x_%j.err
#
# Paired sig/bkg raw dump (same events, injected vs background from the
# pre-made waveforms_sig_bkg_50k.hdf5 -- counts always match) for the three
# groupnorm reg checkpoints, script generalized to build model.arch from
# each run's own config.yaml (ResNet+GroupNorm decoder, not bare S4Model).

set -euo pipefail
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/train

REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe
RUNBASE=${REPO}/dev/runs/merger_4s_d256s32_ssmlr_resnet_groupnorm/train/reg

declare -A CKPT_SUBDIR=(
  [test2]=chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_test2
  [test3]=chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_test3
  [regdev_test1]=chirp_mass_snr_4_50_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_regdev_test1
)

for VER in test2 test3 regdev_test1; do
  RUNDIR=${RUNBASE}/chirp_mass_26-30s_d256_s32_l4_ssmlr_resnet_groupnorm_${VER}
  CFG=${RUNDIR}/config.yaml
  CKPT=${RUNDIR}/BNS-PUBLICATION/${CKPT_SUBDIR[$VER]}/checkpoints/last.ckpt
  OUTDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/raw_csv/s4d_resnet_groupnorm_${VER}

  uv run python ${REPO}/dev/infer/diagnostic/save_outputs.py \
    --outdir "$OUTDIR" --config "$CFG" --ckpt "$CKPT"
  uv run python ${REPO}/dev/infer/diagnostic/plot_outputs.py --outdir "$OUTDIR"
done
