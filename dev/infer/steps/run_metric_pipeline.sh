#!/bin/bash
# Drive step2->3->4->5 for one metric across a set of integration variants,
# writing into a metric/integration directory tree:
#
#   <aggregated>/<metric>/<integration>/{background,foreground,sensitive_volume}.*
#
# METRIC folds the cached raw model output into a detection statistic
# (sigma | mass_over_sigma | mass_minus_ksigma | mass2_over_sigma). With the
# 2-channel [mass, sigma] cache, every metric is computed offline here -- no
# re-serve. A 1-channel legacy cache supports only METRIC=sigma.
#
#   bash run_metric_pipeline.sh <results_dir> <aggregated_root> [metric]
set -euo pipefail

RESULTS=${1:?usage: run_metric_pipeline.sh <results_dir> <aggregated_root> [metric]}
AGG_ROOT=${2:?need aggregated root}
METRIC=${3:-sigma}
STEPS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps

# step2/3/5 need the infer env (infer.postprocess, ledger); step4's SV plot
# needs the plots env (plots.legacy). Select each per-step via uv --project so
# no cd juggling and no cross-env import errors.
INFER_ENV=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
PLOTS_ENV=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/plots

# integration variants: "tag|step2 args". A single-sample boxcar = no
# integration. Gaussian std is set to width_samples/4 (std = 4*width_s at
# sr=16), so each gaussian spans a proper bell over its physical window.
VARIANTS=(
  "no_integration|--integration boxcar --integration-width 0.0625"
  "boxcar_0.25s|--integration boxcar --integration-width 0.25"
  "boxcar_0.5s|--integration boxcar --integration-width 0.5"
  "boxcar_1s|--integration boxcar --integration-width 1.0"
  "gaussian_0.0625s|--integration gaussian --integration-width 0.0625 --gaussian-std 0.25"
  "gaussian_0.125s|--integration gaussian --integration-width 0.125 --gaussian-std 0.5"
  "gaussian_0.25s|--integration gaussian --integration-width 0.25 --gaussian-std 1"
  "gaussian_0.5s|--integration gaussian --integration-width 0.5 --gaussian-std 2"
  "gaussian_1s|--integration gaussian --integration-width 1.0 --gaussian-std 4"
  "gaussian_2s|--integration gaussian --integration-width 2.0 --gaussian-std 8"
)

for entry in "${VARIANTS[@]}"; do
  itag="${entry%%|*}"
  args="${entry#*|}"
  outdir="$AGG_ROOT/$METRIC/$itag"
  # per-branch files are tagged metric+integration so different metrics never
  # clobber each other's intermediate background_<tag>.hdf5 in a branch dir
  tag="${METRIC}_${itag}"
  echo "=== [$METRIC/$itag] step2 ==="
  uv run --project "$INFER_ENV" python "$STEPS/step2_postprocess.py" \
      --results "$RESULTS" --metric "$METRIC" $args --tag "$tag"
  echo "=== [$METRIC/$itag] step3 aggregate -> $outdir ==="
  uv run --project "$INFER_ENV" python "$STEPS/step3_aggregate.py" \
      --results "$RESULTS" --tag "$tag" --outdir "$outdir"
  echo "=== [$METRIC/$itag] step4 plot ==="
  uv run --project "$PLOTS_ENV" python "$STEPS/step4_sv_plot.py" --dir "$outdir"
done

echo "=== [$METRIC] step5 summary ==="
uv run --project "$INFER_ENV" python "$STEPS/step5_summary.py" \
    --aggregated "$AGG_ROOT/$METRIC"
echo "DONE: $AGG_ROOT/$METRIC"
