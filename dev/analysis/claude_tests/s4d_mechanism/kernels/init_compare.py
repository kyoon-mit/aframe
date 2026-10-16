"""Resonances at initialization (a fresh draw from each network's own init code) against the trained
ones: share of in-band resonances whose ring time lies between tau(f; 2.75) and tau(f; 0.85)."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

torch.set_num_threads(8)
out = {}
for name in ("noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "clean_s4d_layer1"):
    cfg = kc.load_cfg(name)
    arch = cfg["model"]["init_args"]["arch"]
    import importlib
    mod, cls = arch["class_path"].rsplit(".", 1)
    Cls = getattr(importlib.import_module(mod), cls)
    torch.manual_seed(1)
    m = Cls(**arch["init_args"]).double()
    if name.startswith("clean"):
        kern, fs = m.net.s4_layers[0].kernel, 2048.0
    else:
        kern, fs = m.kernel, 1024.0
    init = kc.s4d_modes(kern, fs)
    L = 8192 if fs == 2048.0 else 4096
    E = kc.mode_energy(init["z"], init["c"], L)
    w_init = E / E.sum(1, keepdims=True)
    tr = np.load(os.path.join(kc.HERE, f"{name}_modes.npz"))
    res = {}
    for tag, f, ring, w in (("init", init["f"], init["ring"], w_init), ("trained", tr["f"], tr["ring"], tr["weight"])):
        fa = np.abs(f)
        inb = (fa >= 20) & (fa < fs / 2)
        lo, hi = kc.tau_of_f(fa, 2.75), kc.tau_of_f(fa, 0.85)
        inside = inb & (ring >= lo) & (ring <= hi) & (lo <= kc.MERGE / 2048)   # tau band, chirp in the window
        res[tag] = {"inband": int(inb.sum()),
                    "ring_in_tau_band": int(inside.sum()),
                    "share_of_inband": round(float(inside.sum() / inb.sum()), 4),
                    "energy_weighted_share_of_inband": round(float((w * inside).sum() / (w * inb).sum()), 4),
                    "inband_ring_median_s": round(float(np.median(ring[inb])), 4)}
    out[name] = res
    np.savez(os.path.join(kc.HERE, f"{name}_init_modes.npz"), f=init["f"], ring=init["ring"], weight=w_init)
    print(name, res)
with open(os.path.join(kc.HERE, "init_compare.json"), "w") as fh:
    json.dump(out, fh, indent=1)
