#!/bin/bash
# Exit (to wake Claude) on: a new launch, a new GUARD/GAVE UP line, an OOM kill, a real Traceback in a log
# written after start (not our own SIGTERM stops), or after 2 h.
R=/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests
Q=$R/queue/launch_lowmem2.out
n0=$(grep -c launched $Q); g0=$(grep -cE "GUARD|GAVE UP|exited during" $Q); t0=$(date +%s)
since=$(date '+%Y-%m-%d %H:%M:%S'); M=$(mktemp); touch $M
while true; do
  sleep 60
  n=$(grep -c launched $Q); g=$(grep -cE "GUARD|GAVE UP|exited during" $Q)
  [ "$n" -gt "$n0" ] && { echo "NEW LAUNCH"; grep launched $Q | tail -1; break; }
  [ "$g" -gt "$g0" ] && { echo "GUARD EVENT"; grep -E "GUARD|GAVE UP|exited during" $Q | tail -1; break; }
  journalctl -k --since "$since" 2>/dev/null | grep -q "oom-kill" && { echo "OOM KILL"; journalctl -k --since "$since" | grep "Killed process" | tail -5; break; }
  for f in $R/sig_only_reg_s4d*/train.log $R/free_kernel_*/train.log $R/s4d_*/train.log $R/aframe_resnet1d*/train.log; do
    [ "$f" -nt "$M" ] || continue
    tail -c 4000 $f | grep -aq "unable to allocate shared memory" && { echo "SHM ERROR in $f"; exit 0; }
    if tail -c 4000 $f | grep -aq "Traceback" && ! tail -c 4000 $f | grep -aq "killed by signal: Terminated"; then echo "TRACEBACK in $f"; tail -c 1500 $f; exit 0; fi
  done
  [ $(( $(date +%s) - t0 )) -ge 7200 ] && { echo "2H TICK"; break; }
done
tail -3 $Q; awk '/MemAvailable/ {print "MemAvailable", int($2/1048576), "GB"}' /proc/meminfo
