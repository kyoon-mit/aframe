"""Kernel matches in the network-energy convention of kernels/ (match_net = sqrt(max_t 2|k * x|^2 /
||x||^2) against the 235 whitened grid templates at 1024 Hz, H1): the four trained networks (read
from kernels/<tag>_match.npz) and the 256-template bank (computed here with kernels/kcommon.py's
peak_energy_ratio). Per network: the best channel's match for each template chirp mass."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
K = HERE.parent / "kernels"
sys.path.insert(0, str(K))
import kcommon as kc  # noqa: E402

torch.set_num_threads(8)
TAGS = {"d64": "noisy_s4d_d64", "d512": "noisy_s4d_d512", "band": "noisy_s4d_d512_band", "free": "noisy_free"}


def stats(m):
    best = m.max(1)
    return {"best_channel_match_median": float(np.median(best)), "best_channel_match_p10": float(np.quantile(best, 0.1)),
            "best_channel_match_p90": float(np.quantile(best, 0.9)), "best_channel_match_min": float(best.min()),
            "channels_with_best_match_over_0.5": int((m.max(0) > 0.5).sum()),
            "channels_with_best_match_over_0.7": int((m.max(0) > 0.7).sum()),
            "best_by_chirp_mass_quartile": [float(np.median(q)) for q in np.array_split(best, 4)]}


x, mc = kc.templates(rate=1024, ifo=0)
bank = np.load(HERE / "data" / "bank256.npz")
E, _ = kc.peak_energy_ratio(bank["kernels"].astype(np.complex128), x)
m_bank = np.sqrt(E)
print("templates", x.shape, "bank", bank["kernels"].shape, "match", m_bank.shape, flush=True)
out = {"bank": stats(m_bank)}
# the bank template nearest in chirp mass to each grid template, and its match
near = np.abs(np.log(bank["chirp_mass"])[None] - np.log(mc)[:, None]).argmin(1)
out["bank"]["nearest_template_match_median"] = float(np.median(m_bank[np.arange(len(mc)), near]))
out["bank"]["argmax_template_is_within_1pct_chirp_mass"] = float(
    (np.abs(bank["chirp_mass"][m_bank.argmax(1)] / mc - 1) < 0.01).mean())
for tag, name in TAGS.items():
    z = np.load(K / f"{name}_match.npz")
    assert np.allclose(z["chirp_mass"], mc)
    out[tag] = stats(z["match_net"])
    out[tag]["inband_best_channel_match_median"] = float(np.median(np.nanmax(z["inband_match"], 1)))
np.savez(HERE / "results" / "match_bank.npz", match_net=m_bank.astype(np.float32), chirp_mass=mc)
json.dump(out, open(HERE / "results" / "match_compare.json", "w"), indent=1)
print(json.dumps(out, indent=1))
