#!/bin/bash
#SBATCH --job-name=serve2ch
#SBATCH --partition=gpu_requeue        # preemptible; plenty of GPUs
#SBATCH --gres=gpu:1
#SBATCH --constraint="cc7.0|cc8.0"     # V100/A100 only: the Triton 23.01 CUDA
                                       # can't JIT on newer archs -- Blackwell
                                       # (sm_120) gives "nvrtc: invalid -arch"
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G                     # host RAM is the real limit (~73G/serve)
#SBATCH --time=08:00:00
#SBATCH --requeue                      # resume if preempted
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/serve2ch_shard%a_%j.log
#
# One Triton server for the 2-channel [mass, sigma] repo, serving shard $SHARD.
# Each shard's client runs SERIALLY against its own server, so no server ever
# juggles concurrent sequences -- concurrency across the run comes from many of
# these pairs at once, not from one server. Needs $SHARD in the environment.

set -euo pipefail
: "${SHARD:?set SHARD}"
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

# unique port triple per shard so co-located servers on one gpu_requeue node
# never collide on 8000/8001/8002
HTTP=$((8000 + SHARD * 10))
GRPC=$((8001 + SHARD * 10))
METRICS=$((8002 + SHARD * 10))

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/serve.py \
    --repo "$TS/chirp_mass_snr_4_50_60-64s_d64_s64_l4_regdev_test7_2ch" \
    --ip-file "$TS/triton_ip_2ch_shard${SHARD}.txt" \
    --http-port "$HTTP" --grpc-port "$GRPC" --metrics-port "$METRICS" \
    --log-file "/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/server2ch_shard${SHARD}.log"
