#!/bin/bash
# Run the full integration sweep for every metric on the 2-channel cache,
# organized as <aggregated>/<metric>/<integration>/. One metric after another
# so the CPU load and logs stay manageable; each metric ends with its own
# ranked summary.txt.
#
#   bash run_all_metrics.sh <results_dir> <aggregated_root>
set -euo pipefail
RESULTS=${1:?results dir}
AGG_ROOT=${2:?aggregated root}
STEPS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps

for metric in sigma mass_over_sigma mass_minus_ksigma mass2_over_sigma; do
    echo "########## METRIC $metric ##########"
    bash "$STEPS/run_metric_pipeline.sh" "$RESULTS" "$AGG_ROOT" "$metric"
done
echo "ALL_METRICS_DONE $AGG_ROOT"
