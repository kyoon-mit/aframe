#!/bin/bash

# submit.sh -- one GPU condor job per model that trains + evaluates the
# learned scorers (scorer run).
#
# Usage:
#   ./submit.sh [--bid <n>] [model ...] [-- extra scorer args]
#
# With no model names it submits the prototype pair (s4d_merger_1s,
# linoss_4s_merger).  Everything after `--` is passed through to `scorer run`.

set -euo pipefail

SCORER_DIR="/home/barmstrong/aframe_official/projects/scorer"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUB_FILE="${SCRIPT_DIR}/scorer.sub"
UV=/home/barmstrong/.local/bin/uv

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

if [[ ${#MODELS[@]} -eq 0 ]]; then
    MODELS=(s4d_merger_1s linoss_4s_merger)
fi

for name in "${MODELS[@]}"; do
    echo "submitting scorer run for ${name}"
    /usr/local/bin/condor_submit_bid \
        ${BID} \
        -append "arguments = ${UV} run --no-sync --directory ${SCORER_DIR} scorer run --runs ${name} --device cuda ${EXTRA[*]}" \
        "${SUB_FILE}"
done
