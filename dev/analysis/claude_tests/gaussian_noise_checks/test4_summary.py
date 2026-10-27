"""Numbers for Test 4: the S4D and Mamba-2 den-reg trained in Gaussian noise.

Prints, from the eval_by_snr.py output, the checkpoints, the runs' Weights
and Biases records and the configs:
- chirp-mass accuracy by SNR band on the same validation windows;
- validation and training accuracy at matched epochs;
- the size and the filters of each network;
- the Mamba-2 denoiser's memory per head;
- how often each training signal has been seen, with 4 and with 10 repeats;
- the share of each validation population below SNR 6 and 8.

usage: test4_summary.py <eval.npz>
"""

import math
import sys

import numpy as np
import torch
import wandb
from scipy.stats import binom

RUNS = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
NAME = "den_reg_{}_gaussian_c_rho_5_lambda_amp_3_train_snr8_prob0.96"
IDS = {"s4d": "xubc87f1", "mamba2": "y2mc1f8r"}
ENTITY = "kyoon-mit-massachusetts-institute-of-technology"
BANDS = [(4, 5), (5, 6), (6, 8), (8, 12), (12, 100), (8, 100), (4, 100)]
EPOCHS = [9, 49, 99, 199, 299, 399, 411]
SAMPLE_RATE, BATCH, PROB, POOL = 2048, 128, 0.96, 200_000


def ckpt(arch):
    path = f"{RUNS}/{NAME.format(arch)}/CLAUDE-TESTS/{IDS[arch]}/checkpoints/last.ckpt"
    return torch.load(path, map_location="cpu", weights_only=False)


def accuracy(rows):
    for tag, mode in (("", "eval mode"), ("_dropout", "dropout on, one draw")):
        print(f"Chirp-mass accuracy on the same validation windows, {mode}:"
              " share within 1% / 2% / 5%")
        for lo, hi in BANDS:
            k = (rows["snr"] >= lo) & (rows["snr"] < hi)
            line = f"  SNR {lo} to {hi}, {k.sum()} windows"
            for name in ("s4d", "mamba"):
                e = np.abs(rows[name + tag][k] - rows["true"][k]) / rows["true"][k]
                line += f" | {name} " + " / ".join(f"{(e < t).mean():.3f}" for t in (0.01, 0.02, 0.05))
            print(line)


def history():
    api = wandb.Api()
    keys = ("val/within_1pct_chirp_mass", "train/within_1pct_chirp_mass", "val/rho")
    table = {}
    for arch, rid in IDS.items():
        rows = {}
        for row in api.run(f"{ENTITY}/CLAUDE-TESTS/{rid}").scan_history(page_size=5000):
            if row.get("epoch") is None:
                continue
            d = rows.setdefault(int(row["epoch"]), {})
            d.update({k: row[k] for k in keys if row.get(k) is not None})
        table[arch] = rows
    print("Weights and Biases record, by epoch: validation within 1%, training within 1%"
          " (dropout on, SNR 8 to 100), validation overlap rho")
    for e in EPOCHS:
        line = f"  epoch {e}"
        for arch in IDS:
            d = table[arch].get(e, {})
            line += f" | {arch} " + " / ".join(
                f"{d[k]:.3f}" if k in d else "-" for k in keys)
        print(line)
    for arch in IDS:
        val = {e: d[keys[0]] for e, d in table[arch].items() if keys[0] in d}
        best = max(val, key=val.get)
        last = max(val)
        print(f"  {arch}: last epoch {last}, validation within 1% {val[last]:.3f};"
              f" best {val[best]:.3f} at epoch {best}")
        train = {e: d[keys[1]] for e, d in table[arch].items() if keys[1] in d}
        for lo in (330, last - 19):
            span = [val[e] for e in range(lo, lo + 20) if e in val]
            tspan = [train[e] for e in range(lo, lo + 20) if e in train]
            print(f"    epochs {lo} to {lo + 19}: mean within 1%, validation {np.mean(span):.3f}"
                  f" ({len(span)} epochs), training {np.mean(tspan):.3f} ({len(tspan)} epochs)")


def networks():
    s4d, mamba = ckpt("s4d"), ckpt("mamba2")
    for arch, c in (("s4d", s4d), ("mamba2", mamba)):
        s = c["state_dict"]
        counts = {part: sum(v.numel() for k, v in s.items() if f".{part}." in k)
                  for part in ("denoiser", "regressor")}
        print(f"{arch}: checkpoint epoch {c['epoch']}, denoiser {counts['denoiser']:,} numbers,"
              f" regressor {counts['regressor']:,}")
    s = s4d["state_dict"]
    pre = "model.model.denoiser.s4_layers.{}.kernel."
    print("S4D denoiser, per layer: modes (channels x tones), share with frequency above 1 Hz,"
          " share with 1/e memory above 1 s")
    for i in range(6):
        dt = s[pre.format(i) + "log_dt"].exp()[:, None]
        decay = dt * s[pre.format(i) + "log_A_real"].exp()
        freq = (dt * s[pre.format(i) + "A_imag"]).abs() / (2 * math.pi) * SAMPLE_RATE
        print(f"  layer {i}: {decay.numel()} modes, {(freq > 1).float().mean():.3f},"
              f" {(1 / decay > SAMPLE_RATE).float().mean():.3f}")
    s = mamba["state_dict"]
    print("Mamba-2 denoiser, per layer: heads (one real decay rate each), shared B and C width")
    for i in range(6):
        A = s[f"model.model.denoiser.layers.{i}.A_log"]
        proj = s[f"model.model.denoiser.layers.{i}.in_proj.weight"].shape[0]
        d_inner = s[f"model.model.denoiser.layers.{i}.out_proj.weight"].shape[1]
        d_state = (proj - 2 * d_inner - A.numel()) // 2
        print(f"  layer {i}: {A.numel()} heads, inner width {d_inner}, one B and one C of size {d_state}")


def memory(m):
    print("Mamba-2 denoiser 1/e memory 1 / (dt A) on the first validation batch, seconds:"
          " median and 99th percentile over samples")
    for i, layer in enumerate(m):
        print(f"  layer {i}: " + ", ".join(
            f"head {h} {np.median(layer[:, h]):.4f} / {np.percentile(layer[:, h], 99):.4f}"
            for h in range(layer.shape[1])))


def repeats():
    n = np.arange(BATCH + 1)
    p = binom.pmf(n, BATCH, PROB)
    print("Distinct signals: N ~ Binomial(128, 0.96) samples carry a signal, ceil(N / R) distinct ones")
    for R, batches in ((4, 391), (10, 978)):
        per_batch = (p * np.ceil(n / R)).sum()
        print(f"  R = {R}: {per_batch:.3f} per batch, {per_batch * batches:,.0f} per epoch"
              f" of {batches} batches")
        if R == 4:
            for arch in IDS:
                epochs = ckpt(arch)["epoch"] + 1
                print(f"    {arch}, {epochs} epochs: {per_batch * batches * epochs / POOL:.1f} draws"
                      f" per training waveform, {per_batch * batches * epochs * R / POOL:.1f}"
                      f" noisy copies")


def population():
    def below(x, alpha, a=4.0, b=100.0):
        return (x ** (alpha + 1) - a ** (alpha + 1)) / (b ** (alpha + 1) - a ** (alpha + 1))

    for alpha in (-3, -4):
        print(f"Validation SNR density x^{alpha} on [4, 100]: below 6 {below(6, alpha):.4f},"
              f" below 8 {below(8, alpha):.4f}")


def main():
    rows = dict(np.load(sys.argv[1]))
    accuracy(rows)
    history()
    networks()
    memory(rows["memory"])
    repeats()
    population()


if __name__ == "__main__":
    main()
