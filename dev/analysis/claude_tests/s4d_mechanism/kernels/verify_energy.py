"""End-to-end check: the module's own peak energy (LongConvEnergyNet.peak_energy, one detector)
on noiseless whitened templates equals SNR^2 * match_net^2 from analyze.py."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

torch.set_num_threads(8)
out = {}
for name in ("noisy_s4d_d64", "noisy_s4d_d512", "noisy_free"):
    m, ck, cfg = kc.build(name)
    M = np.load(os.path.join(kc.HERE, f"{name}_match.npz"))
    x2048, mc = kc.templates(2048, 0)
    x1024, _ = kc.templates(1024, 0)
    rho2 = (x1024 ** 2).sum(-1)
    js = [0, 60, 117, 180, 234]
    if m.kernel_type == "s4d" and m.kernel.A_imag.shape[1] > 32:
        z = np.load(os.path.join(kc.HERE, f"{name}_modes.npz"))
        k = kc.kernel_from_modes(z["z"], z["c"], m.length)
        k = torch.tensor(k / np.linalg.norm(k, axis=-1, keepdims=True))
        m.kernels = lambda: k
        note = "kernels from the resonances (checked against the module on 8 channels)"
    else:
        note = "module kernels()"
    with torch.no_grad():
        E = m.peak_energy(torch.tensor(x2048[js])[:, None]).numpy()          # (5, C)
    pred = rho2[js, None] * M["match_net"][js].astype(float) ** 2
    rel = np.abs(E - pred) / np.maximum(pred, 1e-12 * pred.max())
    big = pred > 1e-3 * pred.max()
    out[name] = {"kernels": note, "max_rel_diff_where_energy_above_1e-3_of_max": float(rel[big].max()),
                 "median_rel_diff": float(np.median(rel))}
    print(name, out[name])
with open(os.path.join(kc.HERE, "verify_energy.json"), "w") as fh:
    json.dump(out, fh, indent=1)
