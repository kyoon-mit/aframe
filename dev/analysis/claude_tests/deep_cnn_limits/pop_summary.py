"""Fractions within 1% from the population Monte Carlo (mc_pop.py),
overall and by SNR band and chirp-mass band, with binomial errors."""
import glob
import json

import numpy as np

L1 = np.log(1.01)
files = sorted(glob.glob("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/mc/pop_*.json"))
mc, snr, err = [], [], {}
for p in files:
    r = json.load(open(p))
    mc += r["mc0"]
    snr += r["snr"]
    for k, v in r["err"].items():
        err.setdefault(k, []).extend(v)
mc, snr = np.array(mc), np.array(snr)
models = list(err)
ok = {k: np.abs(np.array(v)) < L1 for k, v in err.items()}
n = mc.size
print(f"{len(files)} files, {n} realizations; SNR median {np.median(snr):.2f}, below 8: {np.mean(snr < 8):.3f}; "
      f"chirp mass median {np.median(mc):.3f}")


def row(mask, label):
    m = mask.sum()
    cells = []
    for k in models:
        p = ok[k][mask].mean()
        cells.append(f"{100 * p:.1f} ± {100 * np.sqrt(p * (1 - p) / m):.1f}")
    print(f"| {label} ({m}) | " + " | ".join(cells) + " |")


print("| events | " + " | ".join(models) + " |")
print("|---" * (len(models) + 1) + "|")
row(np.ones(n, bool), "all")
for a, b in ((4, 6), (6, 8), (8, 16), (16, 100)):
    row((snr >= a) & (snr < b), f"SNR {a} to {b}")
for a, b in ((0.85, 1.3), (1.3, 2.0), (2.0, 2.75)):
    row((mc >= a) & (mc < b), f"Mc {a} to {b}")
# paired differences against the coherent model
for k in models[1:]:
    d = ok["coherent"].astype(float) - ok[k].astype(float)
    print(f"coherent minus {k}: {100 * d.mean():.1f} ± {100 * d.std(ddof=1) / np.sqrt(n):.1f} points (paired)")
