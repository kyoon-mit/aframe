#!/bin/bash
# After run_big.sh: the SNR >= 8 early-stopping set, read-outs stopped on it (512,000 training windows),
# per-epoch traces for two networks, then summary and figure.
P=/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/predict
cd /home/kyoon/SSM-BNS/aframe/projects/train || exit 1
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
until grep -q -E "done big 512000|FAILED" $P/logs/run_big.log 2>/dev/null; do sleep 10; done
grep -q FAILED $P/logs/run_big.log && { echo "run_big failed"; exit 1; }
[ -f $P/data/es8/info.json ] || /usr/bin/time -v .venv/bin/python $P/features.py es8 > $P/logs/features_es8.log 2>&1 || { echo "FAILED features es8"; exit 1; }
echo "done features es8"
i=0
for group in "free d512 d64 bank" "band d512_exact d64_exact"; do
  i=$((i + 1))
  /usr/bin/time -v .venv/bin/python $P/readout_es8.py 512000 $group > $P/logs/readout_es8_$i.log 2>&1 &
done
wait
grep -h -E "Traceback|Error" $P/logs/readout_es8_*.log && { echo "FAILED es8 read-outs"; exit 1; }
echo "done es8 read-outs"
/usr/bin/time -v .venv/bin/python $P/diag_epochs.py free mlp1x1024_d0.1 512000 30 > $P/logs/diag_free.log 2>&1 &
/usr/bin/time -v .venv/bin/python $P/diag_epochs.py d512 mlp3x1024_d0.2 512000 30 > $P/logs/diag_d512.log 2>&1 &
wait
echo "done diag"
.venv/bin/python $P/summary.py > $P/logs/summary.log 2>&1 && echo "done summary" || echo "FAILED summary"
.venv/bin/python $P/figure.py > $P/logs/figure.log 2>&1 && echo "done figure" || echo "FAILED figure"
