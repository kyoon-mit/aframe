#!/bin/bash
# Run detached so it survives logout:
#   nohup ./merger_4s_den_cls_64s64n4_resnet14_dynmix.sh > train.log 2>&1 &
#
# Version auto-increments from existing run dirs. Override with VERSION=v12.

set -euo pipefail

REPO=/home/kyoon/SSM-BNS/aframe
CONFIG=$(dirname "$(readlink -f "$0")")/merger_4s_den_cls_64s64n4_resnet14_local.yaml
RUNS=${REPO}/dev/runs/denoise_cls_4s/train/class
BASE=s4d_den_cls_64s64n4_resnet14

# Highest existing v<N> for this BASE, plus one.
LAST=$(ls -d "${RUNS}/${BASE}_v"* 2>/dev/null | sed 's/.*_v//' | grep -E '^[0-9]+$' | sort -n | tail -1)
VERSION=${VERSION:-v$(( ${LAST:-0} + 1 ))}
RUN=${BASE}_${VERSION}
RUN_DIR=${RUNS}/${RUN}

echo "run: ${RUN}"
echo "dir: ${RUN_DIR}"

export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPUS:-0}

cd "${REPO}/projects/train"
uv run python -m train fit --config "${CONFIG}" \
    --trainer.default_root_dir="${RUN_DIR}" \
    --trainer.logger.init_args.name="${RUN}" \
    --trainer.logger.init_args.id="${RUN}" \
    --trainer.logger.init_args.save_dir="${RUN_DIR}"
