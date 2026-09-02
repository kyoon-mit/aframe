#!/bin/bash
# ONE command to run an end-to-end sensitive-volume measurement for ANY model.
# Everything runs in SLURM (survives session idle); serving is on gpu_test
# (no gpu_requeue OOM); clients are %1 co-located per shard (no eviction). See
# project_sv_pipeline_gotchas for why this is the only design that works.
#
# Usage:
#   run_sv.sh --ckpt <last.ckpt> --config <run config.yaml> \
#             --group <reg|class> --name <run_name> [--livetime 1wk|1mo] \
#             [--metrics "m1 m2 ..."] [--repo-skeleton <dir>] [--inject <yaml>]
#
# It: (1) traces the ckpt on gpu_test (arch/dims/outputs auto-read from config),
#     (2) runs NSHARD=2 co-located gpu_test shards over the branch subset,
#     (3) runs a dependent CPU sweep -> SV curves per metric.
# Outputs land under: triton_ts/merger_4s/<group>/<name>/
#   model_repo/  step1_raw_output_<lt>/  step2_aggregated_<lt>/<metric>/<integ>/
set -euo pipefail
SVDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/sv
TS=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/triton_ts/merger_4s
JOBDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s
LIVETIME=1wk METRICS="" SKEL="" INJECT=""

while [ $# -gt 0 ]; do case "$1" in
  --ckpt) CKPT=$2; shift 2;; --config) CONFIG=$2; shift 2;;
  --group) GROUP=$2; shift 2;; --name) NAME=$2; shift 2;;
  --livetime) LIVETIME=$2; shift 2;; --metrics) METRICS=$2; shift 2;;
  --repo-skeleton) SKEL=$2; shift 2;; --inject) INJECT=$2; shift 2;;
  *) echo "unknown arg $1"; exit 1;; esac; done
: "${CKPT:?--ckpt}"; : "${CONFIG:?--config}"; : "${GROUP:?--group reg|class}"; : "${NAME:?--name}"

RUN=$TS/$GROUP/$NAME
REPO_REL=$GROUP/$NAME/model_repo
RESULTS_REL=$GROUP/$NAME/step1_raw_output_$LIVETIME
AGG_REL=$GROUP/$NAME/step2_aggregated_$LIVETIME
BRLIST=$RUN/branches_$LIVETIME.txt
mkdir -p "$RUN"

# --- served output channels from config d_output: 1 -> score/sigma (1ch), 2 -> mass_sigma (2ch)
DOUT=$(grep -A6 'arch:' "$CONFIG" | grep -m1 'd_output:' | grep -oE '[0-9]+')
CFGDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/configs/infer
if [ "$DOUT" = "2" ]; then CH=2; [ -z "$SKEL" ] && SKEL=$TS/reg/chirp_mass_snr_4_50_60-64s_d64_s64_l4_regdev_test7_2ch/model_repo
  [ -z "$METRICS" ] && METRICS="sigma mass_over_sigma inv_sigma neg_log_sigma mass2_over_sigma"
  [ -z "$INJECT" ] && INJECT=$CFGDIR/reg/merger_4s_id2.yaml
else CH=1; [ -z "$SKEL" ] && SKEL=$TS/class/merger_4s_cls/model_repo
  [ -z "$METRICS" ] && METRICS="score"
  [ -z "$INJECT" ] && INJECT=$CFGDIR/class/merger_4s_big_s4d_prenorm.yaml; fi
# trace reads the TRAIN config (arch/dims); step1 reads the INFER config
# (postprocessor + injection set). They are DIFFERENT files -- do not conflate.
echo "d_output=$DOUT ($CH-channel); trace-config=$CONFIG; infer-config=$INJECT; metrics: $METRICS"

# --- repo skeleton (clone once), keep its model.pt to be overwritten by the trace
if [ ! -d "$REPO_REL" ] && [ ! -d "$TS/$REPO_REL" ]; then
  [ -d "$SKEL" ] || { echo "no repo skeleton at $SKEL"; exit 1; }
  cp -r "$SKEL" "$TS/$REPO_REL"; echo "cloned repo skeleton from $SKEL"
fi
OUT=$TS/$REPO_REL/aframe/1/model.pt

# --- branch list: 1wk = shifts 0-7 of the non-big files; 1mo = all 250
python - "$JOBDIR/branches.txt" "$LIVETIME" > "$BRLIST" <<'PY'
import sys, os, re
lines=open(sys.argv[1]).read().splitlines(); lt=sys.argv[2]
dur=lambda p: int(re.search(r'-(\d+)\.hdf5', p).group(1))
durs=[dur(l.split()[0]) for l in lines]
if lt=="1mo":
    print(" ".join(str(i) for i in range(len(lines)))); sys.exit()
big=durs.index(max(durs))//25            # 25-branch block of the longest file
keep=[i for i in range(len(lines)) if i//25!=big and (i%25)<8]
print(" ".join(map(str,keep)))
PY
NBR=$(wc -w < "$BRLIST"); echo "branch list ($LIVETIME): $NBR branches -> $BRLIST"

# --- 1) trace  2) co-shards (after trace ok)  3) sweep (after shards)
TID=$(sbatch --parsable --export=ALL,CKPT="$CKPT",CONFIG="$CONFIG",OUT="$OUT" $SVDIR/trace.slurm)
AID=$(sbatch --parsable --dependency=afterok:$TID --array=0-1 \
  --export=ALL,NSHARD=2,REPO="$REPO_REL",RESULTS="$RESULTS_REL",CONFIG="$INJECT",BRLIST="$BRLIST" \
  $SVDIR/co_shard.slurm)
SID=$(sbatch --parsable --dependency=afterany:$AID \
  --export=ALL,RESULTS="$RESULTS_REL",BRLIST="$BRLIST",METRICS="$METRICS",AGG="$AGG_REL" \
  $SVDIR/sweep.slurm)
echo "trace=$TID  coshards=$AID  sweep=$SID"
echo "SV: $TS/$AGG_REL/<metric>/<integration>/sensitive_volume.png"
echo "if sweep reports INCOMPLETE, just: sbatch --array=0-1 --export=... $SVDIR/co_shard.slurm  (skips done)"
