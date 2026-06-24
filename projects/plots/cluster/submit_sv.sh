#!/bin/bash

# submit_sv.sh
# Submit one CPU-only condor job per run directory to compute the
# alternative-scoring sensitive volume (projects/plots/scoring_sv.py).
#
# For each directory <name> in RUNS_DIR it runs scoring_sv.py against
# <name>/results_aframe.  Directories without a results_aframe folder are
# skipped.
#
# Usage:
#   ./submit_sv.sh [--bid <n>] [extra scoring_sv.py args...]
#
# Examples:
#   ./submit_sv.sh                       # all runs, default bid
#   ./submit_sv.sh --bid 50 --force      # re-run everything, bid 50

set -euo pipefail

RUNS_DIR="${RUNS_DIR:-/home/barmstrong/aframe_official/runs/aframe_results/runs}"
PLOTS_DIR="/home/barmstrong/aframe_official/projects/plots"
OUT_DIR="/fast/barmstrong/outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_FILE="${SCRIPT_DIR}/sv.sub"

BID=25
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bid)
            BID="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

mkdir -p "${OUT_DIR}"

for run in "${RUNS_DIR}"/*/; do
    name="$(basename "${run}")"
    run_dir="${run}results_aframe"
    if [[ ! -f "${run_dir}/timeseries.hdf5" ]]; then
        echo "skipping ${name}: no ${run_dir}/timeseries.hdf5"
        continue
    fi
    echo "submitting ${name} -> ${run_dir}"
    /usr/local/bin/condor_submit_bid \
        ${BID} \
        -append "arguments = run --no-sync --directory ${PLOTS_DIR} python scoring_sv.py --run-dir ${run_dir} ${EXTRA_ARGS[*]}" \
        "${SUB_FILE}"
done
