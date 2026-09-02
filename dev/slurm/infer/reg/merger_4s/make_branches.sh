#!/bin/bash
# Generate branches.txt = one "(background_file shifts)" pair per line.
# The infer array (merger_s4.slurm) reads line (SLURM_ARRAY_TASK_ID + 1).
#
#   shift [0,0]  -> zero-lag: injections recovered (foreground.hdf5)
#   shift [0,k]  -> time-slide: background-only for the FAR estimate
#
# "More shifts" = a bigger slide grid -> more background time -> lower FAR floor.
set -euo pipefail

# O3b to match the o3b injection set (else zero-lag finds no injections).
BG_DIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/DATA/O3b_H1_L1_2048Hz
OUT=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/slurm/infer/reg/merger_4s/branches.txt

# L1 time-slides in seconds (H1 fixed at 0). NON-ZERO only: shift 0 is real data
# (not a timeslide) and the injection set has no [0,0] injections. The o3b set
# has shifts up to 27 -- extend this list for more background + foreground.
SHIFTS=($(seq 1 25))   # 25 shifts -> ~1/month FAR floor (within the set's 27)

: > "$OUT"
for f in "$BG_DIR"/background-*.hdf5; do
    for s in "${SHIFTS[@]}"; do
        echo "$f [0,$s]" >> "$OUT"
    done
done

N=$(wc -l < "$OUT")
echo "wrote $OUT : $N pairs"
echo "set merger_s4.slurm  -->  #SBATCH --array=0-$((N - 1))"
