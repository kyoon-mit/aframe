"""Check that fitting the complex (analytic) kernel with complex modes gives
the same match as fitting the real kernel Re K with real S4D-style terms
2 Re(c_n z_n^l) (2N real columns: Re z^l and Im z^l), unconstrained."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import RES, f_of_tau, fit_tradeoff, kernel_of, logz, place, tau_of_f, toy_chirp

out = []
for mc in (1.2, 2.75):
    K = kernel_of(toy_chirp(mc))
    f1 = float(f_of_tau(4.0, mc))
    fn = np.linspace(f1, 900.0, 100001)
    for c, alpha in ((1.1, 1.0), (1.5, 1.0), (2.0, 0.5)):
        fr = place(fn, c * np.minimum(tau_of_f(fn, mc), 4.0))
        lz = logz(alpha / np.minimum(tau_of_f(fr, mc), 4.0), fr)
        mc_cplx = fit_tradeoff(lz, K, caps=(np.inf,))["inf"]["match"]
        l = np.arange(K.size, dtype=float)
        V = np.exp(l[:, None] * lz[None, :])
        Vr = np.concatenate([V.real, V.imag], axis=1)
        y = K.real
        coef, *_ = np.linalg.lstsq(Vr, y, rcond=1e-12)
        yh = Vr @ coef
        m_real = float(np.dot(yh, y) / (np.linalg.norm(yh) * np.linalg.norm(y)))
        out.append({"mc": mc, "c": c, "alpha": alpha, "N": int(lz.size),
                    "match_complex": mc_cplx, "match_real": m_real})
        print(out[-1], flush=True)
json.dump(out, open(os.path.join(RES, "c9_real.json"), "w"), indent=1)
