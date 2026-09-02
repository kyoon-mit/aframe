#!/bin/bash
# One self-contained SV shard: start OWN triton, stream this shard's branch
# subset serially (%1), then tear the server down. Co-locating server+client in
# one job removes every failure mode we hit: no shared IP files, no cross-job
# server death, no login-node loop (survives session idle), no %2/%3 eviction.
#
# Serve on gpu_test (A100 MIG 20GB, one slice per job) -- gpu_requeue packs many
# servers onto one 9.5GB MIG and OOMs. See project_sv_pipeline_gotchas.
#
# Submit as an array 0..NSHARD-1; shard k streams BRLIST[k::NSHARD].
# Required env: NSHARD REPO RESULTS CONFIG BRLIST  (REPO/RESULTS are TS-relative,
# BRLIST is an ABSOLUTE path on SHARED fs -- never node-local /tmp).
# Optional env: CACHE_REL -- if set and a per-branch injections.hdf5 exists
# there (see step0_cache.slurm), step1 loads it via injection_set_fname
# override instead of the full injection set. Avoids the ~10+min fancy-index
# HDF5 read per branch eating scarce gpu_test time; see
# project_sv_pipeline_gotchas / the co_shard-cpu-split note.
#SBATCH --job-name=sv_coshard
#SBATCH --partition=gpu_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/sv_coshard_%A_%a.log
set -uo pipefail
: "${NSHARD:?}"; : "${REPO:?}"; : "${RESULTS:?}"; : "${CONFIG:?}"; : "${BRLIST:?}"
SHARD=$SLURM_ARRAY_TASK_ID
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
INFER=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
STEPS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps
HTTP=$((8000+SHARD*10)); GRPC=$((8001+SHARD*10)); METRICS=$((8002+SHARD*10))
IPF=$TS/sv_ip_${SLURM_ARRAY_JOB_ID}_${SHARD}.txt; rm -f "$IPF"
SLOG=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/sv_server_${SLURM_ARRAY_JOB_ID}_${SHARD}.log

cd $INFER
uv run python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/serve.py \
  --repo "$TS/$REPO" --ip-file "$IPF" \
  --http-port $HTTP --grpc-port $GRPC --metrics-port $METRICS --log-file "$SLOG" &
SPID=$!
for i in $(seq 1 120); do [ -s "$IPF" ] && break; sleep 5; done
[ -s "$IPF" ] || { echo "server never ready"; kill $SPID 2>/dev/null; exit 1; }
read -r IP GP HP < "$IPF"; ADDR="$IP:$GP"
for i in $(seq 1 60); do curl -sf "http://$IP:$HP/v2/health/ready" >/dev/null && break; sleep 5; done

mapfile -t BR < <(tr ' ' '\n' < "$BRLIST" | grep -v '^$')
for ((k=SHARD; k<${#BR[@]}; k+=NSHARD)); do
  idx=${BR[$k]}
  [ -s "$TS/$RESULTS/branch_$idx/timeseries.hdf5" ] && { echo "branch $idx done, skip"; continue; }
  line=$(sed -n "$((idx+1))p" "$JOBDIR/branches.txt")
  bg=$(echo "$line"|cut -d' ' -f1); sh=$(echo "$line"|cut -d' ' -f2)
  echo "shard $SHARD -> branch $idx"
  INJECT_OVERRIDE=()
  if [ -n "${CACHE_REL:-}" ] && [ -s "$TS/$CACHE_REL/branch_$idx/injections.hdf5" ]; then
    TRUE_INJECT=$(grep -A2 'injection_set_fname:' "$CONFIG" | head -1 | sed 's/.*injection_set_fname:\s*//')
    INJECT_OVERRIDE=(--data.injection_set_fname "$TS/$CACHE_REL/branch_$idx/injections.hdf5" \
                      --true_injection_set_fname "$TRUE_INJECT")
  fi
  uv run python "$STEPS/step1_raw_scores.py" \
    --config "$CONFIG" --client.address "$ADDR" --data.triton_address "$ADDR" \
    --data.background_fname "$bg" --data.shifts "$sh" "${INJECT_OVERRIDE[@]}" \
    --outdir "$TS/$RESULTS/branch_$idx" || echo "branch $idx FAILED"
done
kill $SPID 2>/dev/null
echo "shard $SHARD done"
