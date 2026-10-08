import sys

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

d = sys.argv[1]
w = h5py.File(f"{d}/windows.h5")
m = h5py.File(f"{d}/mf.h5")
names = list(w.attrs["param_names"])
true = w["params/chirp_mass"][:]
snr = w["params/snr"][:]
sig = ~np.isnan(true)
ml = w["ml_mean"][:, names.index("chirp_mass")]
ml_sigma = w["ml_sigma"][:, names.index("chirp_mass")]
mf = m["mf_chirp_mass"][:]
mf_snr = m["mf_snr"][:]

print(f"{d.split('/')[-1]}: {sig.sum()} signal, {(~sig).sum()} background windows")
print(f"sanity: median mf_snr / injected snr (signal) = {np.median(mf_snr[sig] / snr[sig]):.3f}; "
      f"background mf_snr median {np.median(mf_snr[~sig]):.2f}, 99% {np.percentile(mf_snr[~sig], 99):.2f}")

def within(est):
    return np.abs(est[sig] - true[sig]) / true[sig] < 0.02

wm, wl = within(mf), within(ml)
s = snr[sig]
row = ["SNR band   MF    ML   n"]
for lo, hi in [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 50)]:
    k = (s >= lo) & (s < hi)
    if k.sum():
        row.append(f"{lo:>2}-{hi:<3}   {wm[k].mean():4.0%} {wl[k].mean():4.0%} {k.sum()}")
row.append(f"all      {wm.mean():4.0%} {wl.mean():4.0%} {sig.sum()}")
print("\n".join(row))

y = sig.astype(int)
print(f"AUC signal vs background: MF network SNR {roc_auc_score(y, mf_snr):.3f}   "
      f"ML -sigma {roc_auc_score(y, -ml_sigma):.3f}")
for lo, hi in [(8, 12)]:
    k = sig & (snr >= lo) & (snr < hi)
    yy = np.r_[np.ones(k.sum()), np.zeros((~sig).sum())]
    print(f"AUC SNR {lo}-{hi} vs background: MF {roc_auc_score(yy, np.r_[mf_snr[k], mf_snr[~sig]]):.3f}   "
          f"ML {roc_auc_score(yy, -np.r_[ml_sigma[k], ml_sigma[~sig]]):.3f}")
