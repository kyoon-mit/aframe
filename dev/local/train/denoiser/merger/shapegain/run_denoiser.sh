#!/bin/bash
# Launch one denoiser variant, detached so it survives logout.
#
#   ./run_denoiser.sh s4d_den_d64s64n4
#   GPUS=1 ./run_denoiser.sh s4d_den_d64s64n4_no_norm
#   RESUME=no ./run_denoiser.sh s4d_den_d64s64n4      # ignore last.ckpt
#
# All trainer config (run name, root dir, wandb) lives in the variant yaml,
# not here. This script only picks the config, the GPU, and the checkpoint.

set -euo pipefail

HERE=$(dirname "$(readlink -f "$0")")
REPO=/home/kyoon/SSM-BNS/aframe

NAME=${1:?usage: $0 <variant-name>}
CONFIG=${HERE}/${NAME}.yaml
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }

# Resume from the run's own last.ckpt if present, unless RESUME=no.
# RESUME=<path> resumes from a specific checkpoint.
CKPT_ARGS=()
case "${RESUME:-auto}" in
    no|none|false) ;;
    auto)
        ROOT=$(grep -m1 'default_root_dir:' "${CONFIG}" | awk '{print $2}')
        LAST=$(find "${ROOT}" -name last.ckpt 2>/dev/null | head -1)
        [ -n "${LAST:-}" ] && CKPT_ARGS=(--ckpt_path "${LAST}")
        ;;
    *) CKPT_ARGS=(--ckpt_path "${RESUME}") ;;
esac

export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPUS:-0}

echo "run:    ${NAME}"
echo "gpu:    ${CUDA_VISIBLE_DEVICES}"
echo "resume: ${CKPT_ARGS[1]:-none (fresh)}"

cd "${REPO}/projects/train"
exec uv run python -m train fit \
    --config "${HERE}/base.yaml" \
    --config "${CONFIG}" \
    "${CKPT_ARGS[@]}"
