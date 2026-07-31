#!/bin/bash
# CPU sweep: verify the branch list is complete, then fold the cached raw scores
# into SV curves for one or more metrics (step2->3->4->5). No GPU, no server.
# Runs as a DEPENDENT job after the co-shards, so the whole SV finishes inside
# SLURM (survives session idle). Required env: RESULTS BRLIST METRICS AGG.
#SBATCH --job-name=sv_sweep
#SBATCH --partition=shared
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/sv_sweep_%j.log
set -uo pipefail
: "${RESULTS:?}"; : "${BRLIST:?}"; : "${METRICS:?}"; : "${AGG:?}"
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
RES=$TS/$RESULTS
n=$(wc -w < "$BRLIST"); c=0
for i in $(cat "$BRLIST"); do [ -s "$RES/branch_$i/timeseries.hdf5" ] && c=$((c+1)); done
echo "raw complete: $c/$n"
[ $c -lt $n ] && { echo "INCOMPLETE ($c/$n) -- rerun co-shards to fill gaps, they skip done branches"; exit 1; }
for m in $METRICS; do
  echo "=== SWEEP METRIC $m ==="
  bash /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps/run_metric_pipeline.sh \
    "$RES" "$TS/$AGG" "$m"
done
echo "SWEEP DONE $TS/$AGG"
