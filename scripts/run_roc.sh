#!/bin/bash
#SBATCH --job-name=roc_metrics
#SBATCH --partition=gpu_test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-roc-%j.out
#SBATCH --error=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/slurm_logs/slurm-roc-%j.err

# ROC curves (-sigma vs |mu|/sigma) for the three fine-tuned checkpoints.
# Submit with:  sbatch scripts/run_roc.sh

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe_linoss
PY=projects/train/.venv/bin/python
ROC=scripts/roc_metrics.py
OUT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/aframe_data/roc

for w in 59-60s 60-64s 63-64s; do
    echo "==================== ROC ${w} ===================="
    $PY $ROC --config $OUT/cfg_${w}.yaml --output $OUT/roc_${w}.png --device cuda \
        && echo "OK ${w}" || echo "FAILED ${w}"
done
echo "ALL ROC DONE"
