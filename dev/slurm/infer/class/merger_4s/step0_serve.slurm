#!/bin/bash
#SBATCH --job-name=merger_4s_class_serve
#SBATCH --partition=gpu_test           # A100 MIG: homogeneous, compatible arch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=63999M                   # gpu_test host-RAM cap (~64GB/slice)
#SBATCH --time=12:00:00                # gpu_test max
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/merger_4s_class_serve_%j.log
#
# One Triton server for the classification repo (merger_4s_cls), serving shard
# $SHARD on its own port triple so co-located servers never collide. Each
# shard's client runs SERIALLY against its own server. Needs $SHARD in env.

set -euo pipefail
: "${SHARD:?set SHARD}"
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

HTTP=$((8000 + SHARD * 10))
GRPC=$((8001 + SHARD * 10))
METRICS=$((8002 + SHARD * 10))

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/serve.py \
    --repo "$TS/${REPO:-merger_4s_cls}" \
    --ip-file "$TS/triton_ip_${TAG:-class}_${SHARD}.txt" \
    --http-port "$HTTP" --grpc-port "$GRPC" --metrics-port "$METRICS" \
    --log-file "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/server_class_${SHARD}.log"
