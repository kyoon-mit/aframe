"""Section 3 explicit criterion: a lone resonator of ring time T = tau(f_n)
has group delay tau(f_n) at its centre, so its phase follows the chirp's
phase (linearized at f_n) up to x - arctan(x), x = 2 pi (f - f_n) tau.
Keeping that error below a quarter cycle (pi/2) fixes the share of the band
it can serve, 2 x* / (2 pi tau), hence c = pi / x*.
Also the chirp's own curvature over that share, and the lone resonator's
amplitude at the edge."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import brentq

from common import RES, tau_of_f

xs = brentq(lambda x: x - np.arctan(x) - np.pi / 2, 0.5, 10.0)
c = np.pi / xs
out = {"x_star": xs, "share_of_band_times_tau": xs / np.pi, "c": c, "k_3c_over_8": 3 * c / 8,
       "amplitude_at_edge": 1 / np.sqrt(1 + xs ** 2)}
# chirp curvature over the share: pi |dtau/df| df^2 with df = x*/(2 pi tau), |dtau/df| = (8/3) tau / f
rows = []
for mc in (0.85, 1.2, 1.8, 2.75):
    for f in (100.0, 200.0, 400.0, 600.0):
        tau = float(tau_of_f(f, mc))
        if tau > 4.0:
            continue
        dfe = xs / (2 * np.pi * tau)
        rows.append({"mc": mc, "f": f, "tau": tau, "curvature_phase_at_edge_rad": float(np.pi * (8 / 3) * tau / f * dfe ** 2)})
out["chirp_curvature_at_edge"] = rows
out["earlier_criterion_c_2pi"] = 2 * np.pi
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(RES, "c8_quarter.json"), "w"), indent=1)
