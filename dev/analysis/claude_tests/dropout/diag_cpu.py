"""Dropout on and off with the same weights, on cached clean training-pipeline windows (CPU).

Windows: the first 4,096 of s4d_energy_diag/train_windows_snr8_powerlaw.h5, keeping the
3,915 with a signal (finite chirp mass). For each checkpoint: within 1% and the mean
pooled feature with dropout off (eval mode) and on (train mode, 3 seeds); on the first
NSUB windows, the average over K dropout draws of the pooled features and of the
estimate (Monte Carlo dropout). Writes diag_cpu.json next to this script.

Usage: python diag_cpu.py [tag ...]   (default: all checkpoints in CKS)
"""
import json
import os
import shutil
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/projects/train")
from train.experimental.oracle.arch import ComplexS4Regressor, S4Regressor, _blocks  # noqa: E402

torch.set_num_threads(8)
HERE = os.path.dirname(os.path.abspath(__file__))
H5 = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
RUNS = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
N_FIRST, BATCH, SEEDS, NSUB, K = 4096, 32, (0, 1, 2), 512, 16
Y_MEAN, Y_STD = 1.8, 0.55
ARCH = dict(num_ifos=2, num_params=1, d_model=64, d_state=32, n_layers=4, prenorm=True,
            num_groups=32, dt_min=1e-4, dt_max=10.0)


def ck(run, name):
    return f"{RUNS}/{run}/CLAUDE-TESTS/{run}/checkpoints/{name}"


CKS = {
    "max_p02_ep63": ("max", 0.2, ck("sig_only_reg_s4d_max_64x32x4", "ck_63.ckpt")),
    "max_p02_last": ("max", 0.2, ck("sig_only_reg_s4d_max_64x32x4", "last.ckpt")),
    "max_p0_last": ("max", 0.0, ck("sig_only_reg_s4d_max_dropout0_64x32x4", "last.ckpt")),
    "max_p0_latest": ("max", 0.0, ck("sig_only_reg_s4d_max_dropout0_64x32x4", "last.ckpt")),
    "mean_p02_ep19": ("mean", 0.2, ck("sig_only_reg_s4d_64x32x4", "ck_19.ckpt")),
    "mean_p02_last": ("mean", 0.2, ck("sig_only_reg_s4d_64x32x4", "last.ckpt")),
    "emax_p02_ep55": ("energy_max", 0.2, ck("sig_only_reg_s4d_complex_energy_max_64x32x4", "ck_55.ckpt")),
    "emax_p02_last": ("energy_max", 0.2, ck("sig_only_reg_s4d_complex_energy_max_64x32x4", "last.ckpt")),
}


def build(kind, p):
    if kind == "energy_max":
        return ComplexS4Regressor(**ARCH, dropout=p, readout="energy_max")
    return S4Regressor(**ARCH, dropout=p, pool=kind)


def features(model, kind, x):
    """Pooled features (B, 64) and network output (B, 2)."""
    net = model.net
    if kind in ("max", "mean"):
        h = _blocks(net, x)
        f = h.amax(-1) if kind == "max" else h.mean(-1)
        return f, net.decoder(f)
    z = x.transpose(-1, -2)
    z = net.encoder(z).transpose(-1, -2)
    for layer, norm, dropout in list(zip(net.s4_layers, net.norms, net.dropouts))[:-1]:
        z = z + dropout(layer(net._apply_norm(norm, z)))
    z = net._apply_norm(net.norms[-1], z)
    f = torch.log1p(net._energy(z)).amax(-1)
    return f, net.decoder(f)


def run_all(model, kind, X):
    fs, outs = [], []
    with torch.no_grad():
        for i in range(0, len(X), BATCH):
            f, o = features(model, kind, torch.from_numpy(X[i:i + BATCH]))
            fs.append(f)
            outs.append(o)
    return torch.cat(fs).numpy(), torch.cat(outs).numpy()


def summary(out, mc):
    pred = out[:, 0] * Y_STD + Y_MEAN
    rel = (pred - mc) / mc
    return pred, rel


t0 = time.time()
with h5py.File(H5, "r") as f:
    mc_all = f["params/chirp_mass"][:N_FIRST]
    snr_all = f["params/snr"][:N_FIRST]
    keep = np.isfinite(mc_all)
    X = f["X_clean"][:N_FIRST][keep].astype(np.float32)
mc, snr = mc_all[keep].astype(np.float64), snr_all[keep]
print(f"windows {len(X)} (of first {N_FIRST}); SNR {snr.min():.2f} to {snr.max():.2f}; "
      f"chirp mass {mc.min():.3f} to {mc.max():.3f}; load {time.time() - t0:.1f} s", flush=True)

out_path = f"{HERE}/diag_cpu.json"
results = json.load(open(out_path)) if os.path.exists(out_path) else {}
tags = sys.argv[1:] or list(CKS)
for tag in tags:
    kind, p, path = CKS[tag]
    local = f"{HERE}/ckpt_copy_{tag}.ckpt"
    shutil.copy(path, local)
    ckpt = torch.load(local, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model = build(kind, p)
    model.load_state_dict(sd)
    r = {"kind": kind, "dropout": p, "checkpoint": path, "epoch": int(ckpt["epoch"]),
         "n_windows": int(len(X))}
    t1 = time.time()
    model.eval()
    f_off, o_off = run_all(model, kind, X)
    pred_off, rel_off = summary(o_off, mc)
    r["off"] = {"within_1pct": float(np.mean(np.abs(rel_off) < 0.01)),
                "median_abs_rel": float(np.median(np.abs(rel_off))),
                "mean_rel": float(np.mean(rel_off)),
                "mean_feature": float(f_off.mean())}
    r["on"] = []
    model.train()
    for s in SEEDS:
        torch.manual_seed(s)
        f_on, o_on = run_all(model, kind, X)
        pred_on, rel_on = summary(o_on, mc)
        r["on"].append({"seed": s, "within_1pct": float(np.mean(np.abs(rel_on) < 0.01)),
                        "median_abs_rel": float(np.median(np.abs(rel_on))),
                        "mean_rel": float(np.mean(rel_on)),
                        "mean_feature": float(f_on.mean())})
    # Monte Carlo average over K dropout draws on the first NSUB windows
    Xs, mcs = X[:NSUB], mc[:NSUB]
    fsum, osum, f2sum = 0.0, 0.0, 0.0
    for k in range(K):
        torch.manual_seed(1000 + k)
        f_k, o_k = run_all(model, kind, Xs)
        fsum = fsum + f_k
        f2sum = f2sum + f_k ** 2
        osum = osum + o_k
    f_mc, o_mc = fsum / K, osum / K
    f_sd = np.sqrt(np.maximum(f2sum / K - f_mc ** 2, 0.0))
    pred_mc, rel_mc = summary(o_mc, mcs)
    model.eval()
    f_off_s, o_off_s = f_off[:NSUB], o_off[:NSUB]
    pred_off_s, rel_off_s = summary(o_off_s, mcs)
    w = model.net.decoder.weight.detach().numpy()[0]
    r["mc"] = {"n_windows": NSUB, "draws": K,
               "mean_feature_mc": float(f_mc.mean()), "mean_feature_off": float(f_off_s.mean()),
               "ratio_mean_feature": float(f_mc.mean() / f_off_s.mean()),
               "mean_feature_gap": float((f_mc - f_off_s).mean()),
               "frac_channels_gap_positive": float(np.mean(f_mc - f_off_s > 0)),
               "mean_feature_sd_over_draws": float(f_sd.mean()),
               "within_1pct_mc": float(np.mean(np.abs(rel_mc) < 0.01)),
               "within_1pct_off_subset": float(np.mean(np.abs(rel_off_s) < 0.01)),
               "mean_rel_off_minus_mc": float(np.mean((pred_off_s - pred_mc) / mcs)),
               "median_abs_rel_off_minus_mc": float(np.median(np.abs(pred_off_s - pred_mc) / mcs)),
               "decoder_mu_weight_norm": float(np.linalg.norm(w)),
               "mu_shift_from_features": float(np.mean((f_off_s - f_mc) @ w))}
    r["seconds"] = round(time.time() - t1, 1)
    results[tag] = r
    json.dump(results, open(out_path, "w"), indent=1)
    on = r["on"]
    print(f"{tag:15s} ep {r['epoch']:3d} p {p}: within1% off {100*r['off']['within_1pct']:.1f} | on "
          + " ".join(f"{100*o['within_1pct']:.1f}" for o in on)
          + f" | feature off {r['off']['mean_feature']:.3f} on "
          + " ".join(f"{o['mean_feature']:.3f}" for o in on)
          + f" | MC({K}) feature {r['mc']['mean_feature_mc']:.3f} vs off {r['mc']['mean_feature_off']:.3f}"
          + f" within1% MC {100*r['mc']['within_1pct_mc']:.1f} vs off {100*r['mc']['within_1pct_off_subset']:.1f}"
          + f" | {r['seconds']} s", flush=True)
    os.remove(local)
