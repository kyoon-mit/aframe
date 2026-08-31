#!/bin/bash
# Run the test step on background only, by setting waveform_prob to 0.
#
# With no injections the clean target is identically zero, so whatever the
# denoiser emits here is produced from noise alone.
#
# Test background comes from the held-out validation files.
#
#   ./test_background_only.sh <checkpoint>

set -euo pipefail

REPO=/home/kyoon/SSM-BNS/aframe
CONFIG=$(dirname "$(readlink -f "$0")")/merger_4s_den_cls_64s64n4_resnet14_local.yaml
TEST_BACKGROUND=/home/kyoon/SSM-BNS/DATA/O3a_val_only

CKPT=${1:?usage: $0 <checkpoint>}
OUT=${2:-$(dirname "${CKPT}")/background_only_test}

export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=${GPUS:-0}

mkdir -p "${OUT}"

cd "${REPO}/projects/train"
uv run python -m train test --config "${CONFIG}" \
    --ckpt_path "${CKPT}" \
    --data.init_args.waveform_prob=0.0 \
    --data.init_args.swap_prob=0.0 \
    --data.init_args.mute_prob=0.0 \
    --data.init_args.test_background_dir="${TEST_BACKGROUND}" \
    --data.init_args.test_batches="${TEST_BATCHES:-50}" \
    --trainer.devices=1 \
    --trainer.logger=false \
    --trainer.callbacks="[{class_path: train.callbacks.DenoiserEvolutionCallback, init_args: {n_examples: 4, sample_rate: 2048, out_dir: ${OUT}}}]"

echo
echo "plot: ${OUT}/test.png"
