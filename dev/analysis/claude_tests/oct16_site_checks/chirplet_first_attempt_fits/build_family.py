"""Whitened kernels of the 1.2 Msun row (24496) with both masses scaled by (1 + delta).

Only the chirp mass changes (mass ratio, spins, angles, distance fixed).
Checks: the stored row's kernel against kernels.npz, and the regenerated
delta = 0 kernel against the stored one.
"""

import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402

import kernelgen as KG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROW = 24496
DELTAS = [0.0] + [s * d for d in (1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3, 1.5e-3, 2e-3, 3e-3, 4e-3, 5e-3,
                                  7e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2, 1e-1) for s in (1, -1)]


def main():
    t0 = time.time()
    f_psd, S = KG.load_psd()
    par, plus, cross = KG.row_params(ROW)
    stored = np.load("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/fits/kernels.npz")["K_1.2"]
    _, hw = KG.whitened_window(plus, cross, f_psd, S)
    K_row = KG.kernel_from(hw)
    chk = {"stored_row_kernel_vs_kernels_npz_max_abs_diff": float(np.max(np.abs(K_row - stored)))}
    mc0 = KG.chirp_mass(par["mass_1"], par["mass_2"])
    deltas = sorted(DELTAS)
    Ks, mcs, merger = [], [], []
    for d in deltas:
        hp, hc = KG.generate(par, 1.0 + d)
        m, hw = KG.whitened_window(hp, hc, f_psd, S)
        K = KG.kernel_from(hw)
        Ks.append(K)
        mcs.append(mc0 * (1.0 + d))
        merger.append(m)
        if d == 0.0:
            chk["regenerated_delta0_vs_stored_match"] = float(np.abs(np.vdot(K, stored)))
            chk["regenerated_delta0_vs_stored_max_abs_diff"] = float(np.max(np.abs(K - stored)))
        print(f"delta {d:+.4f} Mc {mc0 * (1 + d):.5f} merger idx {m} ({time.time() - t0:.0f}s)", flush=True)
    np.savez(os.path.join(HERE, "family.npz"), K=np.array(Ks), delta=np.array(deltas), mc=np.array(mcs),
             merger_index=np.array(merger))
    out = {"row": ROW, "chirp_mass_row": mc0, "mass_ratio": par["mass_2"] / par["mass_1"], "checks": chk,
           "deltas": deltas, "wall_s": time.time() - t0}
    json.dump(out, open(os.path.join(HERE, "family.json"), "w"), indent=1)
    print(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
