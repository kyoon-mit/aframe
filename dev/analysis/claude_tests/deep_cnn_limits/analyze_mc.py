"""Summaries of mc_ml.py outputs: scatter of the ML estimate against the
discrete Fisher bound, and the fraction within 1% (|dlnMc| < ln 1.01)."""
import json
import sys

import numpy as np
from scipy import special, stats

from discrete_check import bounds

L1 = np.log(1.01)
for path in sys.argv[1:]:
    r = json.load(open(path))
    print(f"== {path.split('/')[-1]}: Mc {r['mc0']}, SNR {r['snr']}, {len(r['err']['coherent'])} realizations, "
          f"ln Mc search {r['grid'][0]:.4f}..{r['grid'][1]:.4f}")
    fb, _ = bounds(r["mc0"], r["snr"])
    for k, e in r["err"].items():
        e = np.array(e)
        n = e.size
        sf = fb[k]
        sd = e.std(ddof=1)
        # 95% interval of sd from the chi-square distribution
        lo, hi = sd * np.sqrt((n - 1) / stats.chi2.ppf([0.975, 0.025], n - 1))
        rob = 0.5 * (np.percentile(e, 84.13) - np.percentile(e, 15.87))
        w1 = np.mean(np.abs(e) < L1)
        w1_se = np.sqrt(w1 * (1 - w1) / n)
        pred = special.erf(L1 / (sf * np.sqrt(2)))
        out5 = np.mean(np.abs(e) > np.log(1.05))
        print(f"  {k:11s} Fisher {100 * sf:.4f}%  sd {100 * sd:.4f}% [{100 * lo:.4f}, {100 * hi:.4f}] "
              f"(sd/Fisher {sd / sf:.3f})  robust sd {100 * rob:.4f}%  mean {100 * e.mean():+.4f}%  "
              f"within 1%: {w1:.3f} +- {w1_se:.3f} (Gaussian at Fisher sd: {pred:.3f})  beyond 5%: {out5:.3f}")
