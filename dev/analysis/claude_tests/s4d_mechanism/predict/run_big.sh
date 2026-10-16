#!/bin/bash
# After run_readouts.sh: six more training chunks (train3..train8, 64,000 windows each), then the chosen
# read-outs retrained on 256,000 and 512,000 windows (three processes at a time on GPU 1).
P=/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/predict
cd /home/kyoon/SSM-BNS/aframe/projects/train || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
until grep -q -E "done raw|FAILED" $P/logs/run_readouts.log 2>/dev/null; do sleep 10; done
for s in train3 train4 train5 train6 train7 train8; do
  [ -f $P/data/$s/info.json ] && continue
  /usr/bin/time -v .venv/bin/python $P/features.py $s > $P/logs/features_$s.log 2>&1 || { echo "FAILED features $s"; exit 1; }
  echo "done features $s"
done

for n in 256000 512000; do
  i=0
  if [ $n -le 256000 ]; then groups=("d64 d64_exact band" "d512 d512_exact" "free bank"); else groups=("free d512 d64 bank" "band d512_exact d64_exact"); fi
  for group in "${groups[@]}"; do
    i=$((i + 1))
    /usr/bin/time -v .venv/bin/python $P/readout_big.py $n $group > $P/logs/readout_big_${n}_$i.log 2>&1 &
  done
  wait
  grep -h -E "Traceback|Error" $P/logs/readout_big_${n}_*.log && echo "FAILED big $n" && exit 1
  echo "done big $n"
done
