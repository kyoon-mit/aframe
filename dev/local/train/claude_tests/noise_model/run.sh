#!/bin/bash
# Usage: run.sh fit|test <run name> <config.yaml> [<config.yaml> ...] [other arguments]
# Config names are taken from this folder, after base.yaml. Each run gets its
# own folder, dev/runs/claude_tests/noise_model/<run name>, and wandb name and
# id. test uses the run's last checkpoint unless --ckpt_path is given.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
aframe=$(cd "$here/../../../../.." && pwd)
stage=$1
name=$2
shift 2
folder=$aframe/dev/runs/claude_tests/noise_model/$name
mkdir -p "$folder"
args=()
ckpt_given=false
for arg in "$@"; do
  case $arg in
    *.yaml) args+=(--config "$here/$arg") ;;
    --ckpt_path*) ckpt_given=true; args+=("$arg") ;;
    *) args+=("$arg") ;;
  esac
done
if [[ $stage == test && $ckpt_given == false ]]; then
  args+=(--ckpt_path "$folder/CLAUDE-TESTS/$name/checkpoints/last.ckpt")
fi
cd "$aframe/projects/train"
exec .venv/bin/python -m train.experimental.noise_model "$stage" --config "$here/base.yaml" "${args[@]}" \
  --trainer.logger.init_args.name "$name" --trainer.logger.init_args.id "$name" \
  --trainer.logger.init_args.save_dir "$folder" --trainer.default_root_dir "$folder"
