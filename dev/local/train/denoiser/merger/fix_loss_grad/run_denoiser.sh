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
REPO=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe

NAME=${1:?usage: $0 <variant-name>}
shift
# anything after the variant name is passed straight to the CLI, so a run
# can be relabelled or retargeted without editing its config
EXTRA_ARGS=("$@")
CONFIG=${HERE}/${NAME}.yaml
[ -f "${CONFIG}" ] || { echo "no such config: ${CONFIG}" >&2; exit 1; }

# Resume from the run's own last.ckpt if present, unless RESUME=no.
# RESUME=<path> resumes from a specific checkpoint.
CKPT_ARGS=()
case "${RESUME:-auto}" in
    no|none|false) ;;
    auto)
        ROOT=$(grep -m1 'default_root_dir:' "${CONFIG}" | awk '{print $2}')
        # On a fresh run ROOT does not exist yet, so find exits 1 and
        # pipefail would fail the whole assignment, which set -e turns
        # into a silent exit before anything is printed. Look only if the
        # directory is there, and let a failed search be a normal miss.
        LAST=""
        if [ -d "${ROOT}" ]; then
            LAST=$(find "${ROOT}" -name last.ckpt 2>/dev/null | head -1 || true)
        fi
        if [ -n "${LAST}" ]; then
            CKPT_ARGS=(--ckpt_path "${LAST}")
        fi
        ;;
    *) CKPT_ARGS=(--ckpt_path "${RESUME}") ;;
esac

export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPUS:-0}

echo "run:    ${NAME}"
echo "gpu:    ${CUDA_VISIBLE_DEVICES}"
echo "resume: ${CKPT_ARGS[1]:-none (fresh)}"

# A shared base.yaml is optional: a self-contained variant carries
# everything itself and there is nothing to layer under it.
BASE_ARGS=()
if [ -f "${HERE}/base.yaml" ]; then
    BASE_ARGS=(--config "${HERE}/base.yaml")
fi

cd "${REPO}/projects/train"
exec uv run python -m train fit \
    "${BASE_ARGS[@]}" \
    --config "${CONFIG}" \
    "${CKPT_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
