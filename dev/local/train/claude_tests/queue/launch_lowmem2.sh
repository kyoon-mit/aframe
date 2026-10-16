#!/bin/bash
# The only launcher (from 2026-10-16 11:30; successor of launch_lowmem.sh, adding cycle-aware early stopping,
# a force kill for hung jobs and no retries after a clean early stop). Earlier note: (from 2026-10-16 06:00; successor of launch_es2.sh after the 05:51 OOM: pinned waveform
# caches grow to ~24 GB per job). Every launch now loads waveform chunks in the main process
# (waveform_workers 0, pin_waveforms false: one chunk held, no /dev/shm, no pinned cache; same chunks).
# Swaps (running job restarted at its epoch end) free memory, so they need only 10 min spacing. Every run gets early stopping on val within 1%
# (configs: train.callbacks.ResumableEarlyStopping, patience 80 signal-only, 160 noisy).
# A launch needs: <= 11 training jobs running (12 at most), MemAvailable >= 70 GB for 60 s,
# /dev/shm free >= 30 GB, 20 min since the previous launch, and the previous launch past an
# epoch (or 45 min). "swap" restarts a running job right after its end-of-epoch checkpoint so
# it picks up the new config; nothing is lost. New jobs: oom_score_adj 1000 and a guard that
# kills only the newcomer if MemAvailable < 20 GB or /dev/shm free < 10 GB in its first 30 min.
A=/home/kyoon/SSM-BNS/aframe
R=$A/dev/runs/claude_tests
O=$A/dev/local/train/claude_tests/oracle
LC=$A/dev/local/train/claude_tests/long_conv
STEPS=(
  "swap|sig_only_reg_s4d_max_dropout0_64x32x4_warp_premerger|$O"
  "start|sig_only_reg_s4d_dropout0_64x32x4|$O"
  "start|sig_only_reg_s4d_max_dropout0_64x32x4|$O"
  "start|sig_only_reg_s4d_dropout0_64x32x4_warp_merger|$O"
  "start|sig_only_reg_s4d_max_dropout0_64x32x4_warp_merger|$O"
  "start|sig_only_reg_s4d_complex_mean_dropout0_64x32x4|$O"
  "start|sig_only_reg_s4d_complex_mean_dropout0_64x32x4_warp_merger|$O"
  "start|sig_only_reg_s4d_complex_mean_dropout0_64x32x4_warp_premerger|$O"
  "start|sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4|$O"
  "start|sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4_warp_merger|$O"
  "start|sig_only_reg_s4d_complex_energy_max_dropout0_64x32x4_warp_premerger|$O"
  "start|sig_only_reg_resnet_energy_max|$O"
  "start|free_kernel_time_energy_max_256ch_softmax_train_snr4_2048hz_prob0.96|$LC"
)
avail() { awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo; }
shmfree() { df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9'; }
njobs() { ps -eo sid=,args= | awk '/[p]ython -m train fit/ {print $1}' | sort -u | wc -l; }
sustained() { local ok=0; while [ $ok -lt 30 ]; do if [ "$(avail)" -ge "$1" ] && [ "$(shmfree)" -ge 30 ]; then ok=$((ok + 1)); else ok=0; fi; sleep 2; done; }
last_epoch() { tail -c 20000 "$1" 2>/dev/null | tr '\r' '\n' | grep -ao "^Epoch [0-9]*" | tail -1 | awk '{print $2}'; }
main_pid() { ps -eo pid=,args= | awk -v c="/$1.yaml" '$2 ~ /python$/ && index($0, "-m train fit --config") && index($0, c) {print $1}' | while read p; do
  pp=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '); [ -n "$pp" ] || continue
  ps -o args= -p $pp 2>/dev/null | grep -q "train fit" || echo $p; done | head -1; }
guard() {
  local t0=$(date +%s) sid
  while [ $(( $(date +%s) - t0 )) -lt 1800 ]; do
    sid=$(cat "$1" 2>/dev/null)
    if [ -n "$sid" ] && { [ "$(avail)" -lt 20 ] || [ "$(shmfree)" -lt 10 ]; }; then
      kill -TERM -"$sid"; echo "$(date '+%m-%d %H:%M:%S') GUARD killed $2 (session $sid), MemAvailable $(avail) GB, shm free $(shmfree) GB"; return 1
    fi
    if [ -n "$sid" ] && ! kill -0 "$sid" 2>/dev/null; then
      local lg=$R/$2/train.log
      if tail -c 20000 "$lg" | grep -aq "did not improve in the last" && ! tail -c 8000 "$lg" | grep -aq "Traceback"; then
        echo "$(date '+%m-%d %H:%M:%S') $2 finished: early stop (plateau)"; return 0; fi
      echo "$(date '+%m-%d %H:%M:%S') $2 exited during its first 30 min"; return 1; fi
    sleep 2
  done
  return 0
}
start_job() {
  local name=$1 cdir=$2 log=$R/$1/train.log ck=$R/$1/CLAUDE-TESTS/$1/checkpoints/last.ckpt sidf=$R/queue/$1.sid
  mkdir -p $R/$name; rm -f $sidf; cd $A/projects/train
  if [ -f "$ck" ]; then
    local start=$(last_epoch $log); start=${start:-0}
    echo "=== $(date) resume from last.ckpt (launch_es, early stopping)" >> $log
    CUDA_VISIBLE_DEVICES=$gpu setsid bash -c "echo 1000 > /proc/self/oom_score_adj; echo \$\$ > $sidf; exec .venv/bin/python -m train fit --config $cdir/$name.yaml --ckpt_path $ck --data.init_args.waveform_workers 0 --data.init_args.pin_waveforms false" >> $log 2>&1 < /dev/null &
    need=$((start + 2)); mode=resume
  else
    [ -f $log ] && mv $log $log.old_$(date +%H%M)
    CUDA_VISIBLE_DEVICES=$gpu setsid bash -c "echo 1000 > /proc/self/oom_score_adj; echo \$\$ > $sidf; exec .venv/bin/python -m train fit --config $cdir/$name.yaml --data.init_args.waveform_workers 0 --data.init_args.pin_waveforms false" > $log 2>&1 < /dev/null &
    need=1; mode=fresh
  fi
  echo "$(date '+%m-%d %H:%M') launched $name on GPU $gpu ($mode), MemAvailable $(avail) GB, shm free $(shmfree) GB, jobs $(njobs)"
}
gpu=0; prev=""; need=0; tlast=0
for s in "${STEPS[@]}"; do
  IFS='|' read -r kind name cdir <<< "$s"
  if [ -n "$prev" ]; then
    t0=$(date +%s)
    while true; do e=$(last_epoch $prev); e=${e:--1}; [ "$e" -ge "$need" ] && break; [ $(( $(date +%s) - t0 )) -ge 2700 ] && break; sleep 30; done
  fi
  gap=1200; [ "$kind" = swap ] && gap=600
  while [ $(( $(date +%s) - tlast )) -lt $gap ]; do sleep 30; done
  if [ "$kind" = swap ]; then
    pid=$(main_pid $name)
    if [ -n "$pid" ]; then
      ck=$R/$name/CLAUDE-TESTS/$name/checkpoints/last.ckpt; m0=$(stat -c %Y $ck)
      log=$R/$name/train.log
      while [ "$(stat -c %Y $ck)" = "$m0" ]; do
        sleep 10
        [ $(( $(date +%s) - $(stat -c %Y $log) )) -gt 1200 ] && { echo "$(date '+%m-%d %H:%M') $name log silent 20 min; treating it as dead"; break; }
      done; sleep 30
      sid=$(ps -o sid= -p $pid | tr -d ' '); kill -TERM -"$sid"
      for i in $(seq 30); do kill -0 $pid 2>/dev/null || break; sleep 2; done
      kill -0 $pid 2>/dev/null && { kill -KILL -"$sid"; echo "$(date '+%m-%d %H:%M') $name ignored TERM; killed"; sleep 5; }; sleep 10
      echo "$(date '+%m-%d %H:%M') stopped $name after its epoch checkpoint (session $sid)"
    else
      echo "$(date '+%m-%d %H:%M') $name not running; starting it instead"
    fi
  fi
  until [ "$(njobs)" -le 11 ]; do sleep 60; done
  ok=0
  for try in 1 2 3; do
    sustained $(( 70 + 15 * (try - 1) ))
    start_job $name $cdir; tlast=$(date +%s)
    if guard $R/queue/$name.sid $name; then ok=1; break; fi
    sleep 120
  done
  gpu=$((1 - gpu))
  [ $ok = 1 ] && prev=$R/$name/train.log || { echo "$(date '+%m-%d %H:%M') GAVE UP on $name"; prev=""; }
done
echo "$(date '+%m-%d %H:%M') all steps done"
