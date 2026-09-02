#!/bin/bash
#SBATCH --job-name=rawcsv_merger4s_id2
#SBATCH --partition=gpu_test          # A100 MIG 20GB is plenty for this model
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/raw_csv/dump_%j.log
#
# Raw per-window model-output dump (pure torch, no Triton): slides the kernel
# across each injection's neighbourhood and writes every kernel position's raw
# (pred_chirp_mass, pred_sigma) to raw_sig.csv / raw_bkg.csv -- the front-end
# for the SV pipeline (cluster/integrate as post-processing on the CSV).

set -euo pipefail

OUTDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/raw_csv/merger_4s_id2

# cd into the train uv project for its env; the script itself lives in dev/.
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/train
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/diagnostic/save_outputs.py --outdir "$OUTDIR"
# Options: --max-strains N  --min-snr X  --edge-min/-max/-step  (default grid -> [-8,-4]..[-3,1])
