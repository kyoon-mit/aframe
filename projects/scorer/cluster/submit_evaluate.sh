#!/bin/bash

# submit_evaluate.sh -- one CPU-only condor job per model that evaluates the
# integration methods (scorer evaluate --skip-learned) on the held-out segments.
#
# One job per run directory in RUNS_DIR that has a results_aframe/timeseries.hdf5.
# Everything after `--` is passed through to `scorer evaluate`.
#
# Usage:
#   ./submit_evaluate.sh [--bid <n>] [model ...] [-- extra scorer evaluate args]
#
# Examples:
#   ./submit_evaluate.sh                       # all runs, default methods
#   ./submit_evaluate.sh --bid 50 -- --cluster-windows 1 2 4 8 16
#
# With no model names it submits every run under RUNS_DIR with a timeseries.hdf5.

set -euo pipefail

RUNS_DIR="${RUNS_DIR:-/home/barmstrong/aframe_official/runs/aframe_results/runs}"
SCORER_DIR="/home/barmstrong/aframe_official/projects/scorer"
OUT_DIR="/fast/barmstrong/outputs"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_FILE="${SCRIPT_DIR}/evaluate.sub"

BID=25
MODELS=()
EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bid) BID="$2"; shift 2 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) MODELS+=("$1"); shift ;;
    esac
done

# default: every run with a populated timeseries
if [[ ${#MODELS[@]} -eq 0 ]]; then
    for run in "${RUNS_DIR}"/*/; do
        name="$(basename "${run}")"
        [[ -f "${run}results_aframe/timeseries.hdf5" ]] && MODELS+=("${name}")
    done
fi

mkdir -p "${OUT_DIR}"

for name in "${MODELS[@]}"; do
    run_dir="${RUNS_DIR}/${name}/results_aframe"
    if [[ ! -f "${run_dir}/timeseries.hdf5" ]]; then
        echo "skipping ${name}: no ${run_dir}/timeseries.hdf5"
        continue
    fi
    echo "submitting ${name}"
    /usr/local/bin/condor_submit_bid \
        ${BID} \
        -append "arguments = run --no-sync --directory ${SCORER_DIR} scorer evaluate --runs ${name} --skip-learned --device cpu ${EXTRA[*]}" \
        "${SUB_FILE}"
done
