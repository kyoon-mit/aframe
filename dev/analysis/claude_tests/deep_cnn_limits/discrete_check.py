"""Discrete-bin Fisher bounds used to judge the Monte Carlo (recomputed;
the per-segment-time value stored by runs started before the fix is
unreliable when a segment holds a single bin)."""
import numpy as np

from common import Noise, band, segment_edges_f
from mc_ml import DF, MODELS, SEGS, discrete_fisher


def bounds(mc0, snr):
    flo, fhi = band(mc0)
    j = np.arange(int(np.ceil(flo / DF)), int(np.floor(fhi / DF)) + 1)
    f = j * DF
    q = f ** (-7 / 3) / Noise("o3")(f)
    out, nb = {}, {}
    for mname, sname, pt in MODELS:
        fe = segment_edges_f(mc0, SEGS[sname])
        lab = np.clip(np.searchsorted(fe, f, side="right") - 1, 0, len(fe) - 2)
        out[mname] = discrete_fisher(f, q, lab, mc0, snr, pt)
        nb[sname] = np.bincount(lab).tolist()
    return out, nb


if __name__ == "__main__":
    for mc0 in (0.85, 1.2, 2.75):
        b8, nb = bounds(mc0, 8.0)
        print(mc0, {k: f"{100 * v:.4f}%" for k, v in b8.items()}, "bins per segment R0.44:", nb["R0.44"])
        b30, _ = bounds(mc0, 30.0)
        print("   scaling check SNR 30 x 30/8:", {k: f"{100 * v * 30 / 8:.4f}%" for k, v in b30.items()})
