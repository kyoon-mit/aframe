#!/bin/bash
# After the train2 features exist: read-out selection, learning curves, unstandardized features, summary.
# Three processes at a time on GPU 1 (each capped at 1.5 GB of allocator memory).
P=/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/predict
cd /home/kyoon/SSM-BNS/aframe/projects/train || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
until [ -f $P/data/train2/info.json ]; do sleep 5; done
for step in select curve raw; do
  i=0
  for group in "d64 d64_exact band" "d512 d512_exact" "free bank"; do
    i=$((i + 1))
    /usr/bin/time -v .venv/bin/python $P/readout.py $step $group > $P/logs/readout_${step}_$i.log 2>&1 &
  done
  wait
  grep -h -E "Traceback|Error" $P/logs/readout_${step}_*.log && echo "FAILED at $step" && exit 1
  echo "done $step"
done
.venv/bin/python $P/summary.py > $P/logs/summary.log 2>&1 && echo "done summary" || echo "FAILED summary"
