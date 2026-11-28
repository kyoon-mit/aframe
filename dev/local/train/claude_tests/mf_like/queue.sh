#!/bin/bash
# Usage: queue.sh <file listing configs, one per line>
# Runs the configs in order through run.sh, keeping up to 7 runs on GPU 1
# and 2 on GPU 0. A run starts only when at least 40 GB of host RAM is
# free and the GPU's power limit is at most 250 W; launches are 2 min
# apart. Finished runs are skipped. Log: dev/runs/claude_tests/mf_like/scan/queue.log
# Configs with a live run are skipped; only one queue runs at a time; the
# queue stops if a run it started is dead 2 min later.
# Start detached: setsid nohup ./queue.sh scan_order.txt > /dev/null 2>&1 &
set -u
here=$(cd "$(dirname "$0")" && pwd)
list=$(realpath "$1")
log=/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/mf_like/scan/queue.log
mkdir -p "$(dirname "$log")"
declare -A slots=([1]=7 [0]=2)
declare -A gpu_of=()

say() { echo "$(date '+%F %T') $*" >> "$log"; }

reap() {
  for pid in "${!gpu_of[@]}"; do
    kill -0 "$pid" 2>/dev/null || unset "gpu_of[$pid]"
  done
}

free_gpu() {
  for g in 1 0; do
    n=0
    for pid in "${!gpu_of[@]}"; do [[ ${gpu_of[$pid]} == "$g" ]] && n=$((n + 1)); done
    if (( n < slots[$g] )); then echo "$g"; return; fi
  done
}

exec 9> "$(dirname "$log")/queue.lock"
if ! flock -n 9; then
  echo "another queue is running" >&2
  exit 1
fi
say "queue start, pid $$"
while read -r -u 3 config; do
  [[ -z $config ]] && continue
  root=$(grep default_root_dir "$config" | awk '{print $2}')
  if grep -q "max_epochs=.*reached" "$root/train.log" 2>/dev/null; then
    continue
  fi
  if pgrep -f -- "--config $config" > /dev/null; then
    say "already running, skipped: $(basename "$config" .yaml)"
    continue
  fi
  while true; do
    reap
    gpu=$(free_gpu)
    ram=$(free -g | awk 'NR==2{print $7}')
    if [[ -n $gpu ]]; then
      watts=$(nvidia-smi -i "$gpu" --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1)
      (( ram >= 40 && watts <= 250 )) && break
    fi
    sleep 60
  done
  pid=$("$here/run.sh" "$gpu" "$config" | awk '{print $2}')
  gpu_of[$pid]=$gpu
  say "gpu $gpu pid $pid ram ${ram}G $(basename "$config" .yaml)"
  sleep 120
  if ! kill -0 "$pid" 2>/dev/null; then
    say "pid $pid died within 2 min, queue stopped; see $(grep default_root_dir "$config" | awk '{print $2}')/train.log"
    exit 1
  fi
done 3< "$list"
while reap; (( ${#gpu_of[@]} > 0 )); do sleep 60; done
say "queue done"
