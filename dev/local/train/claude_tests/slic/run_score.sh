#!/bin/bash
# Usage: run_score.sh <gpu index> [other arguments]
# Trains the SLIC score network (score.yaml) on one GPU, resuming from the
# run's last checkpoint when there is one. Output goes to
# dev/runs/claude_tests/slic/score/train.log.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
aframe=$(cd "$here/../../../../.." && pwd)
gpu=$1
shift
folder=$aframe/dev/runs/claude_tests/slic/score
ckpt=$folder/CLAUDE-TESTS/slic_score/checkpoints/last.ckpt
mkdir -p "$folder"
args=()
if [[ -f $ckpt ]]; then
  args+=(--ckpt_path "$ckpt")
  echo "=== $(date) resume from $ckpt" >> "$folder/train.log"
fi
cd "$aframe/projects/train"
CUDA_VISIBLE_DEVICES=$gpu exec .venv/bin/python -m train.experimental.slic fit \
  --config "$here/score.yaml" "${args[@]}" "$@" >> "$folder/train.log" 2>&1
