#!/bin/bash
# Usage: run_pe_all.sh <gpu index> <windows.h5> <score checkpoint> <out dir> [chunk]
# run_pe.sh over every window of the dump, ``chunk`` windows per process
# (default 4), one process at a time, writing <out dir>/pe_<start>.h5.
# Finished parts are skipped. Stops before a part if the GPU power cap is
# above 250 W.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
gpu=$1
windows=$(realpath "$2")
ckpt=$(realpath "$3")
out=$(realpath -m "$4")
chunk=${5:-4}
mkdir -p "$out"
total=$("$here/../../../../../projects/train/.venv/bin/python" -c \
  "import h5py, sys; print(len(h5py.File(sys.argv[1], 'r')['X']))" "$windows")
for ((start = 0; start < total; start += chunk)); do
  part=$out/pe_$(printf %04d "$start").h5
  [[ -f $part ]] && continue
  cap=$(nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader,nounits)
  if awk -v c="$cap" 'BEGIN {exit !(c > 250)}'; then
    echo "$(date +%T) GPU $gpu power cap $cap W > 250 W, stopping before $start"
    exit 1
  fi
  echo "$(date +%T) windows $start..$((start + chunk - 1))"
  "$here/run_pe.sh" "$gpu" "$windows" "$ckpt" "$part.tmp" \
    --start "$start" --count "$chunk"
  mv "$part.tmp" "$part"
  mv "$part.tmp.log" "$part.log"
done
echo "$(date +%T) all $total windows done"
