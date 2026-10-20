#!/bin/bash
# Usage: run_pe.sh <gpu index> <windows.h5> <score checkpoint> <out.h5> [other arguments]
# SLIC posterior samples (pe.py, the paper's MALA settings) for windows of a
# dump from dump_test.sh. Other arguments go to pe.py, e.g.
# --start 0 --count 16. An empty gpu index ("") runs on CPU.
# Log: <out.h5>.log. Coverage of finished runs:
#   python -m train.experimental.slic.coverage --pe <out.h5> ... --labels <name> --out <folder>
# One job at a time. Before launching, check host RAM (free -g; one CPU job
# with 32 walkers peaked at 24.5 GB), that the GPU is free (nvidia-smi) and
# its power cap (nvidia-smi --query-gpu=power.limit --format=csv).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
aframe=$(cd "$here/../../../../.." && pwd)
gpu=$1
windows=$(realpath "$2")
ckpt=$(realpath "$3")
out=$(realpath -m "$4")
shift 4
device=cuda
[[ -z $gpu ]] && device=cpu
mkdir -p "$(dirname "$out")"
cd "$aframe/projects/train"
CUDA_VISIBLE_DEVICES=$gpu exec .venv/bin/python -m train.experimental.slic.pe \
  --windows "$windows" --checkpoint "$ckpt" --out "$out" --device "$device" \
  "$@" >> "$out.log" 2>&1
