"""Check the closed-form Gram-matrix least squares in common.fit_match
against numpy lstsq on the explicit matrix of damped tones."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import RES, f_of_tau, fit_match, fit_match_direct, fit_tradeoff, kernel_of, logz, place, tau_of_f, toy_chirp

out = []
for mc in (1.2, 2.75):
    K = kernel_of(toy_chirp(mc))
    f1 = float(f_of_tau(4.0, mc))
    fn = np.linspace(f1, 900.0, 200001)
    for c, alpha in ((1.0, 0.5), (2.72, 0.5), (1.0, np.pi), (1.5, 1.0)):
        fr = place(fn, c * tau_of_f(fn, mc))
        lz = logz(alpha / tau_of_f(fr, mc), fr)
        r = {"mc": mc, "c": c, "alpha": alpha, "N": int(lz.size),
             "gram_match": fit_match(lz, K)["match"],
             "tradeoff_inf_match": fit_tradeoff(lz, K, caps=(np.inf,))["inf"]["match"],
             "lstsq_match": fit_match_direct(lz, K)}
        out.append(r)
        print(r, flush=True)
json.dump(out, open(os.path.join(RES, "c10_gram.json"), "w"), indent=1)
