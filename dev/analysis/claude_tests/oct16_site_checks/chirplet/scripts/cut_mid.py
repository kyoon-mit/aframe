"""Cut pieces with the tangent taken at the midpoint in time of each piece (instead of in count)."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
sys.argv = [sys.argv[0], "/dev/null"]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "construct_check.py")).read().split("res = {")[0])


def cut_pieces_time_mid(mc, fmax, kind):
    h, keep = chirp(mc, fmax)
    n, _, _ = count(TAU, mc, kind)
    n_end = float(count(tau_of_f(fmax, mc), mc, kind)[0])
    idx = np.floor(n).astype(int)
    approx = np.zeros(NS, complex)
    for j in np.unique(idx[keep]):
        sel = keep & (idx == j)
        tau_lo, tau_hi = tau_at_count(j, mc, kind), tau_at_count(min(j + 1, n_end), mc, kind)
        tauc = 0.5 * (tau_lo + tau_hi)
        s = T[sel] + tauc
        ph = phase_of_tau(tauc, mc) + 2 * np.pi * f_of_tau(tauc, mc) * s
        if kind == "lin":
            ph = ph + np.pi * fdot(tauc, mc) * s**2
        approx[sel] = np.abs(h[sel]) * np.exp(1j * ph)
    return match(approx, h)


for fmax in (512.0, 900.0, 1024.0):
    print(fmax, [(mc, round(cut_pieces_time_mid(mc, fmax, "lin"), 4), round(cut_pieces_time_mid(mc, fmax, "tone"), 4)) for mc in MASSES])
