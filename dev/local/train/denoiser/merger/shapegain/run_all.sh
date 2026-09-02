#!/bin/bash
# Launch the shape/gain denoiser scan. All jobs on GPU 0 by default.
#
#   ./run_all.sh
#   GPU=1 ./run_all.sh          # put them all on GPU 1 instead
#
# Each writes its own log next to this script and survives logout.
# Stop everything with: pkill -f "python -m train"

set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
LOGS=${HERE}/logs
mkdir -p "${LOGS}"
GPU=${GPU:-0}

JOBS=(
    shapegain_g03         # fixed weights, moderate gain pressure
    shapegain_g10         # strong amplitude pressure
    shapegain_shapeonly   # gain nearly off: ceiling on achievable rho
    shapegain_learned     # learned uncertainty weighting of the four terms
    shapegain_spline      # spline output basis, edge impulse inexpressible
)

# Dataloader workers are forked after the parent initialises CUDA, and a CUDA
# context does not survive fork, so simultaneous starts on one device can race
# and leave a job with "CUDA error: initialization error". Stagger them.
STAGGER=${STAGGER:-45}

for i in "${!JOBS[@]}"; do
    name=${JOBS[$i]}
    log=${LOGS}/${name}.log

    [ "$i" -gt 0 ] && sleep "${STAGGER}"

    GPUS=${GPU} setsid nohup "${HERE}/run_denoiser.sh" "${name}" \
        > "${log}" 2>&1 < /dev/null &

    echo "launched ${name} on gpu ${GPU}  (pid $!)  -> ${log}"
done

echo
echo "watch:  tail -f ${LOGS}/*.log"
echo "status: /home/kyoon/SSM-BNS/bin/gpustat.sh"
