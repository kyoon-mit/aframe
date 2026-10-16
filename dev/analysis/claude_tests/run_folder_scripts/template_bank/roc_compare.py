"""Detection on the 50k test windows: signal rows against noise-only rows.

usage: roc_compare.py BANK_TEST_DIR DENREG_TEST_DIR
Bank statistics come from detection_scores.npz (logit, newsnr2, snr2); the
den-reg statistic is -sigma(chirp mass) from param_est_results{,_bkg}.csv.
Reports AUC and the fraction of signals found at fixed false-positive rates,
for all signals and for SNR 8-12.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

POPS = ("snr4_powerlaw", "snr8_powerlaw", "snr4_uniform")
FPRS = (1e-2, 1e-3, 1e-4)


def auc(sig, bkg):
    scores = np.concatenate([sig, bkg])
    ranks = scores.argsort().argsort() + 1.0
    return (ranks[: len(sig)].sum() - len(sig) * (len(sig) + 1) / 2) / (len(sig) * len(bkg))


def tpr_at(sig, bkg, fpr):
    if fpr * len(bkg) < 1:
        return float("nan")
    threshold = np.quantile(bkg, 1 - fpr)
    return (sig > threshold).mean()


def row(label, sig, bkg, snr):
    band = (snr >= 8) & (snr < 12)
    cells = [f"{auc(sig, bkg):.4f}", f"{auc(sig[band], bkg):.4f}"]
    cells += [f"{tpr_at(sig, bkg, f):.3f}" for f in FPRS]
    cells += [f"{tpr_at(sig[band], bkg, f):.3f}" for f in FPRS]
    print(f"{label:24s} " + " ".join(f"{c:>8s}" for c in cells))


bank_dir, denreg_dir = Path(sys.argv[1]), Path(sys.argv[2])
for pop in POPS:
    print(f"\n{pop}: AUC all / SNR 8-12; signals found at FPR {FPRS} (all), then (SNR 8-12)")
    f = bank_dir / pop / "detection_scores.npz"
    if f.exists():
        d = np.load(f)
        print(f"  {len(d['signal'])} signal, {len(d['background'])} noise-only windows")
        for key, label in (("", "bank logit"), ("newsnr2_", "bank chi2-reweighted SNR2"), ("snr2_", "bank network SNR2")):
            sig = d[f"{key}signal"] if key else d["signal"]
            bkg = d[f"{key}background"] if key else d["background"]
            row(label, sig, bkg, d["snr"])
    s = denreg_dir / pop / "param_est_results.csv"
    b = denreg_dir / pop / "param_est_results_bkg.csv"
    if s.exists() and b.exists():
        rs, rb = pd.read_csv(s), pd.read_csv(b)
        col = [c for c in rb.columns if c.startswith("sigma_chirp_mass")][0]
        row("den-reg -sigma", -rs[col].to_numpy(), -rb[col].to_numpy(), rs["snr"].to_numpy())
