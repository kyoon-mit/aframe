#!/bin/bash
#SBATCH --job-name=rawcsv_plot_id2
#SBATCH --partition=shared             # CPU: plotting + verification only
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/slurm/wandb_logs/raw_csv/plot_%j.log
#
# Auto-runs (via --dependency=afterok:<dump job>) once the raw CSV dump finishes:
# verifies sig != bkg, then writes the sigma / chirp-relerr figures. Independent
# of any interactive session, so the plots get made even overnight.

set -euo pipefail

OUTDIR=/n/holystore01/LABS/iaifi_lab/Lab/kyoon/MODEL/aframe/reg-dev-latest/raw_csv/merger_4s_id2

cd /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/projects/train

# --- sanity: sig and bkg must NOT be identical ---
.venv/bin/python - "$OUTDIR" <<'PY'
import sys, pandas as pd
d = sys.argv[1]
s = pd.read_csv(f"{d}/raw_sig.csv"); b = pd.read_csv(f"{d}/raw_bkg.csv")
same = s[["pred_chirp_mass", "pred_sigma"]].equals(b[["pred_chirp_mass", "pred_sigma"]])
dmax = float((s.pred_sigma - b.pred_sigma).abs().max())
print(f"strains={s.strain_id.nunique()} rows={len(s)} sig==bkg={same} max|dsigma|={dmax:.4f}")
assert not same and dmax > 0, "sig and bkg are identical -- injection missing!"
print("OK: sig != bkg")
PY

.venv/bin/python /n/holystore01/LABS/iaifi_lab/Lab/kyoon/aframe/dev/infer/diagnostic/plot_outputs.py --outdir "$OUTDIR"
echo "PLOTS_DONE_OK"
