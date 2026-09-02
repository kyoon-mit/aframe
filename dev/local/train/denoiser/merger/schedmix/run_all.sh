#!/bin/bash
# Launch all four denoiser variants concurrently, two per GPU.
#
#   ./run_all.sh
#
# Each writes its own log next to this script. All four survive logout.
# Stop everything with: pkill -f "python -m train"

set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
LOGS=${HERE}/logs
mkdir -p "${LOGS}"

# variant:gpu -- the two normalized runs on GPU 0, the two un-normalized on 1
JOBS=(
    "s4d_den_d64s64n4:0"
    "s4d_den_d64s64n4_low_log_floor:0"
    "s4d_den_d64s64n4_no_norm:1"
    "s4d_den_d64s64n4_low_log_floor_no_norm:1"
)

# Seconds between launches. Dataloader workers are forked after the parent
# initialises CUDA, and a CUDA context does not survive fork, so two jobs
# initialising on the same device at the same instant can race and leave one
# with "CUDA error: initialization error" in its workers. Staggering the
# starts avoids that.
STAGGER=${STAGGER:-45}

for i in "${!JOBS[@]}"; do
    job=${JOBS[$i]}
    name=${job%%:*}
    gpu=${job##*:}
    log=${LOGS}/${name}.log

    [ "$i" -gt 0 ] && sleep "${STAGGER}"

    GPUS=${gpu} setsid nohup "${HERE}/run_denoiser.sh" "${name}" \
        > "${log}" 2>&1 < /dev/null &

    echo "launched ${name} on gpu ${gpu}  (pid $!)  -> ${log}"
done

echo
echo "watch:  tail -f ${LOGS}/*.log"
echo "status: nvidia-smi; pgrep -af 'python -m train'"
