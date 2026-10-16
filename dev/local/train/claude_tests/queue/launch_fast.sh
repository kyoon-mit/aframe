#!/bin/bash
# Launches the queued test runs one every 3 min, with low-memory loading for every job
# (waveform_workers 0, pin_waveforms false). A launch every 3 min while MemAvailable >= 60 GB and
# /dev/shm free >= 30 GB hold for 30 s. A monitor kills only the newest job if MemAvailable < 15 GB.
A=/home/kyoon/SSM-BNS/aframe; R=$A/dev/runs/claude_tests; O=$A/dev/local/train/claude_tests/oracle; LC=$A/dev/local/train/claude_tests/long_conv
JOBS=(
  "sig_only_reg_s4d_dropout0_64x32x4|$O" "sig_only_reg_s4d_max_dropout0_64x32x4|$O"
  "sig_only_reg_s4d_dropout0_64x32x4_warp_merger|$O" "sig_only_reg_s4d_max_dropout0_64x32x4_warp_merger|$O"
  "sig_only_reg_s4d_complex_mean_dropout0_64x32x4|$O" "sig_only_reg_s4d_complex_mean_dropout0_64x32x4_warp_merger|$O"
  "sig_only_reg_s4d_complex_mean_dropout0_64x32x4_warp_premerger|$O" "sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4|$O"
  "sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4_warp_merger|$O" "sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4_warp_premerger|$O"
  "sig_only_reg_resnet_energy_max|$O" "free_kernel_time_energy_max_256ch_softmax_train_snr4_2048hz_prob0.96|$LC"
)
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
shmfree() { df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9'; }
ready() { local ok=0; while [ $ok -lt 15 ]; do if [ "$(avail)" -ge 60 ] && [ "$(shmfree)" -ge 30 ]; then ok=$((ok + 1)); else ok=0; fi; sleep 2; done; }
newest=""
monitor() { while true; do if [ "$(avail)" -lt 15 ] && [ -n "$newest" ]; then s=$(cat $R/queue/$newest.sid 2>/dev/null); [ -n "$s" ] && kill -TERM -"$s" && echo "$(date '+%m-%d %H:%M:%S') MONITOR killed newest $newest at $(avail) GB"; sleep 60; fi; sleep 2; done; }
gpu=1
for j in "${JOBS[@]}"; do
  IFS='|' read -r name cdir <<< "$j"
  ready
  log=$R/$name/train.log; ck=$R/$name/CLAUDE-TESTS/$name/checkpoints/last.ckpt; sidf=$R/queue/$name.sid
  mkdir -p $R/$name; rm -f $sidf; cd $A/projects/train
  OV="--data.init_args.waveform_workers 0 --data.init_args.pin_waveforms false"
  if [ -f "$ck" ]; then
    echo "=== $(date) resume from last.ckpt (launch_fast)" >> $log
    CUDA_VISIBLE_DEVICES=$gpu setsid bash -c "echo 1000 > /proc/self/oom_score_adj; echo \$\$ > $sidf; exec .venv/bin/python -m train fit --config $cdir/$name.yaml --ckpt_path $ck $OV" >> $log 2>&1 < /dev/null &
    mode=resume
  else
    [ -f $log ] && mv $log $log.old_$(date +%H%M)
    CUDA_VISIBLE_DEVICES=$gpu setsid bash -c "echo 1000 > /proc/self/oom_score_adj; echo \$\$ > $sidf; exec .venv/bin/python -m train fit --config $cdir/$name.yaml $OV" > $log 2>&1 < /dev/null &
    mode=fresh
  fi
  newest=$name
  echo "$(date '+%m-%d %H:%M') launched $name on GPU $gpu ($mode), MemAvailable $(avail) GB, shm free $(shmfree) GB"
  [ "$gpu" = 1 ] && gpu=0 || gpu=1
  ( monitor ) & MON=$!; sleep 180; kill $MON 2>/dev/null
done
echo "$(date '+%m-%d %H:%M') all launched"
( newest=""; while true; do sleep 600; done ) &
