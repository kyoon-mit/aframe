"""Per-window offset of the dropout-off estimate from the dropout average, real max-pool
S4D with dropout 0.2 at epoch 232, first 512 cached clean windows with a signal.
Writes offset_check.json and offset_check_arrays.npz next to this script."""
import json
import os
import shutil
import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/projects/train")
from train.experimental.oracle.arch import S4Regressor, _blocks  # noqa: E402

torch.set_num_threads(4)
HERE = os.path.dirname(os.path.abspath(__file__))
H5 = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
CK = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/sig_only_reg_s4d_max_64x32x4/CLAUDE-TESTS/"
      "sig_only_reg_s4d_max_64x32x4/checkpoints/last.ckpt")
NSUB, K, BATCH = 512, 16, 32

with h5py.File(H5, "r") as f:
    mc_all = f["params/chirp_mass"][:4096]
    keep = np.flatnonzero(np.isfinite(mc_all))[:NSUB]
    X = f["X_clean"][:4096][keep].astype(np.float32)
mc = mc_all[keep].astype(np.float64)

local = f"{HERE}/ckpt_copy_offset.ckpt"
shutil.copy(CK, local)
ckpt = torch.load(local, map_location="cpu", weights_only=False)
os.remove(local)
model = S4Regressor(num_ifos=2, num_params=1, d_model=64, d_state=32, n_layers=4, dropout=0.2,
                    prenorm=True, num_groups=32, dt_min=1e-4, dt_max=10.0, pool="max")
model.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")})


def feats():
    out = []
    with torch.no_grad():
        for i in range(0, NSUB, BATCH):
            out.append(_blocks(model.net, torch.from_numpy(X[i:i + BATCH])).amax(-1))
    return torch.cat(out).numpy().astype(np.float64)


model.eval()
m_off = feats()
model.train()
m_draws = []
for k in range(K):
    torch.manual_seed(1000 + k)
    m_draws.append(feats())
m_draws = np.stack(m_draws)
m_mc = m_draws.mean(0)
W = model.net.decoder.weight.detach().numpy().astype(np.float64)
b = model.net.decoder.bias.detach().numpy().astype(np.float64)
mu_off = m_off @ W[0] + b[0]
mu_draws = m_draws @ W[0] + b[0]
mu_mc = mu_draws.mean(0)
pred_off, pred_mc, pred_one = 1.8 + 0.55 * mu_off, 1.8 + 0.55 * mu_mc, 1.8 + 0.55 * mu_draws
d_mu = mu_off - mu_mc
rel_shift = (pred_off - pred_mc) / mc
res = {
    "epoch": int(ckpt["epoch"]), "n_windows": NSUB, "draws": K,
    "decoder_bias_mu": float(b[0]),
    "mean_feature_off": float(m_off.mean()), "mean_feature_mc": float(m_mc.mean()),
    "median_channel_ratio_mc_over_off": float(np.median(m_mc / m_off)),
    "frac_feature_gap_positive": float(np.mean(m_mc > m_off)),
    "d_mu_mean": float(d_mu.mean()), "d_mu_median": float(np.median(d_mu)), "d_mu_sd": float(d_mu.std()),
    "d_mc_mean_msun": float(0.55 * d_mu.mean()),
    "rel_shift_mean": float(rel_shift.mean()), "rel_shift_median": float(np.median(rel_shift)),
    "rel_shift_abs_median": float(np.median(np.abs(rel_shift))),
    "frac_rel_shift_abs_gt_1pct": float(np.mean(np.abs(rel_shift) > 0.01)),
    "frac_rel_shift_negative": float(np.mean(rel_shift < 0)),
    "within_1pct_off": float(np.mean(np.abs(pred_off - mc) / mc < 0.01)),
    "within_1pct_mc": float(np.mean(np.abs(pred_mc - mc) / mc < 0.01)),
    "within_1pct_single_draw_mean": float(np.mean(np.abs(pred_one - mc[None]) / mc[None] < 0.01)),
    "rel_err_mc_median_abs": float(np.median(np.abs(pred_mc - mc) / mc)),
    "rel_err_off_median_abs": float(np.median(np.abs(pred_off - mc) / mc)),
    "budget_d_mu_at_min_mc": float(0.01 * mc.min() / 0.55), "budget_d_mu_at_max_mc": float(0.01 * mc.max() / 0.55),
    "mc_min": float(mc.min()), "mc_max": float(mc.max()),
}
se = m_draws.std(0, ddof=1) / np.sqrt(K)
gap = m_mc - m_off
res["frac_gap_below_minus_3se"] = float(np.mean(gap < -3 * se))
res["frac_gap_above_plus_3se"] = float(np.mean(gap > 3 * se))
res["frac_gap_zero_spread"] = float(np.mean(se == 0))
np.savez_compressed(f"{HERE}/offset_check_arrays.npz", m_off=m_off, m_draws=m_draws, mc=mc,
                    W=W, b=b)
json.dump(res, open(f"{HERE}/offset_check.json", "w"), indent=1)
print(json.dumps(res, indent=1))
