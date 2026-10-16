"""Small-R scaling of the segmented bounds: shared t_c should approach
sigma ~ 1/R, free time per segment sigma ~ 1/R^2."""
import numpy as np

from common import Noise
from fisher_seg import segmented

noise = Noise("o3")
for mc in (0.85, 1.2, 2.75):
    Rs = [0.4, 0.2, 0.1, 0.05]
    s = [segmented(mc, R, noise)["sig_lnmc"] for R in Rs]
    t = [segmented(mc, R, noise, per_segment_time=True)["sig_lnmc"] for R in Rs]
    es = [np.log(s[i + 1] / s[i]) / np.log(2) for i in range(3)]
    et = [np.log(t[i + 1] / t[i]) / np.log(2) for i in range(3)]
    print(f"Mc {mc}: halving R from 0.4 to 0.05 s, exponent of sigma (shared t_c) "
          + " ".join(f"{e:.3f}" for e in es) + "; (free time) " + " ".join(f"{e:.3f}" for e in et))
