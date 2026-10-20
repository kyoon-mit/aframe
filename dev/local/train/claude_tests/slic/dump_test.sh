#!/bin/bash
# Usage: dump_test.sh <out.h5> [count] [other arguments, e.g. --split val --seed 7]
# Writes count (default 16) injected windows of the oracle data config, on CPU.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
aframe=$(cd "$here/../../../../.." && pwd)
out=$(realpath -m "$1")
count=${2:-16}
shift $(( $# < 2 ? $# : 2 ))
export CUDA_VISIBLE_DEVICES=
cd "$aframe/projects/train"
exec .venv/bin/python -m train.experimental.slic.dump_windows \
  --config "$aframe/dev/local/train/claude_tests/oracle/sig_only_reg_s4d_64x32x4.yaml" \
  --out "$out" --count "$count" "$@"
