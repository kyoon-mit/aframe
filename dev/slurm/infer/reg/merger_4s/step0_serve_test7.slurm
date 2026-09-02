#!/bin/bash
#SBATCH --job-name=step0_serve_test7
#SBATCH --partition=gpu               # full A100 80GB, dedicated node
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#
# NOT gpu_test: the 200G is for HOST RAM, not the GPU. Tried gpu_test
# 2026-07-17 (job 32037699) and it was oom_killed at MaxRSS 60.5GB against that
# partition's 64GB/slice cap, after 100/250 branches -- it dies on the 36341s
# background block (branch 100), which is 4x longer than anything before it.
#
# Triton logs "CNMEM_STATUS_OUT_OF_MEMORY, falling back to pinned system
# memory" at the first inference request -- that happens on the full A100 too
# (the CUDA pool is a fixed 64MB) and is NOT the failure. It is just what pushes
# buffers into host RAM, which is what then blows a small --mem cap.
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/step0_serve_test7_%j.log
#
# Step 0 (test7): serve the exported test7 model with Triton. Publishes its IP
# only once the server is READY -- that file is the go-signal step 1 waits on.

set -euo pipefail
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/serve.py \
    --repo "$TS/chirp_mass_snr_4_50_60-64s_d64_s64_l4_regdev_test7" \
    --ip-file "$TS/triton_ip_test7.txt" \
    --log-file /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/server_test7.log
