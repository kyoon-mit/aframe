"""Fine mass grid (0.05% steps, +-10%): exact leading-order chirp and 12-chirplet chain against the kernels."""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[v] = "1"
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location("sa", os.path.join(os.path.dirname(os.path.abspath(__file__)), "start_check_adv_lib.py"))
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)
for name in ("1.2", "1.5", "2.0"):
    K = S.KER[f"K_{name}"]
    mc0 = float(name)
    grid = mc0 * np.exp(np.linspace(np.log(0.9), np.log(1.1), 401))
    best_e = (0, None); best_c = (0, None)
    for m in grid:
        f = S.f_tau(S.tau, m)
        keep = (f > 20) & (f < 1024)
        h = np.where(keep, S.tau ** -0.25 * np.exp(1j * S.phi(S.tau, m)), 0)
        kc = S.to_kernel(S.chain_atoms(m).sum(1))
        ke = S.to_kernel(h)
        for kk, which in ((ke, "e"), (np.conj(ke), "e"), (kc, "c"), (np.conj(kc), "c")):
            v, lag = S.xcorr_best(kk, K)
            if which == "e" and v > best_e[0]:
                best_e = (v, m)
            if which == "c" and v > best_c[0]:
                best_c = (v, m)
    print(name, "exact LO chirp %.4f at %.4f" % best_e, "chain12 %.4f at %.4f" % best_c, flush=True)
