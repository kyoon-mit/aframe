#!/bin/bash
#SBATCH --job-name=step4_sv_plot
#SBATCH --partition=shared
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step4_%j.log
#
# Step 4: sensitive volume vs FAR (aframe's SensitiveVolumePlot).

set -euo pipefail
TAG=${1:-boxcar}
cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/plots
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/step4_sv_plot.py --dir /n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s/aggregated/"$TAG"
