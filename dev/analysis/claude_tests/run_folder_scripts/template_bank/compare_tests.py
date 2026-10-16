"""Chirp mass within 1/2% by SNR band from param_est_results.csv of each test.

usage: compare_tests.py LABEL=TEST_DIR [LABEL=TEST_DIR ...]
TEST_DIR holds snr4_powerlaw/, snr8_powerlaw/, snr4_uniform/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

POPS = ("snr4_powerlaw", "snr8_powerlaw", "snr4_uniform")
BANDS = ((4, 8), (8, 10), (10, 12), (8, 12), (12, 16), (16, 25), (25, 50))

runs = [a.split("=", 1) for a in sys.argv[1:]]
for pop in POPS:
    print(f"\n{pop}: within 1/2% of the true chirp mass")
    print(f"{'model':28s} {'n':>6s} " + " ".join(f"{a}-{b:<4d}" for a, b in BANDS) + "   all")
    for label, d in runs:
        f = Path(d) / pop / "param_est_results.csv"
        if not f.exists():
            print(f"{label:28s} (missing)")
            continue
        r = pd.read_csv(f)
        err = (r["chirp_mass_pred"] - r["chirp_mass_true"]).abs() / r["chirp_mass_true"]
        snr = r["snr"].to_numpy()
        cells = []
        for a, b in BANDS:
            s = (snr >= a) & (snr < b)
            cells.append("   --  " if s.sum() < 30 else f"{100*(err[s]<.01).mean():3.0f}/{100*(err[s]<.02).mean():<3.0f}")
        cells.append(f"{100*(err<.01).mean():3.0f}/{100*(err<.02).mean():<3.0f}")
        print(f"{label:28s} {len(r):6d} " + " ".join(f"{c:>7s}" for c in cells))
