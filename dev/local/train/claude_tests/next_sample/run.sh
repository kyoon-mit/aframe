#!/bin/bash
# Usage: run.sh <gpu index> [config]
# Trains the next-sample noise model, detached from the shell; resumes from
# last.ckpt if present. Log: <default_root_dir>/train.log.
# Before launching: check power.limit (<= 250 W) and free RAM.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
aframe=$(cd "$here/../../../../.." && pwd)
config=$(realpath "${2:-$here/k3_shared_mean_asinh.yaml}")
root=$(grep default_root_dir "$config" | awk '{print $2}')
mkdir -p "$root"
ckpt=$(find "$root" -name last.ckpt 2>/dev/null | head -1)
resume=()
[[ -n $ckpt ]] && resume=(--ckpt_path "$ckpt")
cd "$aframe/projects/train"
CUDA_VISIBLE_DEVICES=$1 setsid nohup .venv/bin/python -m train.experimental.next_sample \
  fit --config "$config" "${resume[@]}" >> "$root/train.log" 2>&1 < /dev/null &
echo "pid $! log $root/train.log"
