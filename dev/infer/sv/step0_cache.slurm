#!/bin/bash
# CPU-only prep: pre-extract each branch's injection subset from the full
# injection-set file, so co_shard (gpu_test, QOS-capped) never pays the
# ~10+min fancy-indexed HDF5 read per branch -- only the fast ~10-15s Triton
# stream. Runs as an array on `shared`, many more cores than gpu_test allows,
# no 2-job QOS cap. Idempotent: skips branches already cached.
# Required env: INJECT_FNAME CACHE_REL BRLIST (BRLIST absolute, shared FS).
#SBATCH --job-name=sv_cache_inj
#SBATCH --partition=shared
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-71%12
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/infer/sv_cache_inj_%A_%a.log
set -uo pipefail
: "${INJECT_FNAME:?}"; : "${CACHE_REL:?}"; : "${BRLIST:?}"
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
INFER=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/infer
STEPS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/steps

mapfile -t BR < <(tr ' ' '\n' < "$BRLIST" | grep -v '^$')
idx=${BR[$SLURM_ARRAY_TASK_ID]}
[ -z "${idx:-}" ] && { echo "no branch for array index $SLURM_ARRAY_TASK_ID, skip"; exit 0; }

line=$(sed -n "$((idx+1))p" "$JOBDIR/branches.txt")
bg=$(echo "$line"|cut -d' ' -f1); sh=$(echo "$line"|cut -d' ' -f2)
sh=${sh//[\[\]]/}; sh=${sh//,/ }

cd $INFER
uv run python "$STEPS/step0_cache_injections.py" \
  --injection_set_fname "$INJECT_FNAME" \
  --background_fname "$bg" \
  --ifos H1 L1 \
  --shifts $sh \
  --outdir "$TS/$CACHE_REL/branch_$idx"
