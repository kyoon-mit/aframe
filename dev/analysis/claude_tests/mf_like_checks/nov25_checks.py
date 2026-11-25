"""Numbers on the 25 November 2026 MNet-Shallow update.

1. Logistic loss of a constant answer when half the labels are 1.
2. Product of a template with a phase-shifted copy of itself.
3. Training waveform file: chirp-mass range and spread, spin and angle ranges.
4. Default nn.Linear start and the one-cycle learning rates of the runs.
5. First-pass results from wandb (CLAUDE-TESTS).

usage: nov25_checks.py <training waveform file>
"""

import math
import sys

import h5py
import numpy as np
import torch

RUNS = "kyoon-mit-massachusetts-institute-of-technology/CLAUDE-TESTS/"
TEMPLATES = [1, 4, 16, 64, 256, 1024, 4096]


def item1():
    p = np.linspace(1e-4, 1 - 1e-4, 999_999)
    loss = -0.5 * np.log(p) - 0.5 * np.log(1 - p)
    i = loss.argmin()
    print(f"1. ln 2 = {math.log(2):.6f}; on a grid the smallest loss is "
          f"{loss[i]:.6f} at p = {p[i]:.6f}")


def item2():
    fs, n = 2048, 8192
    t = np.arange(n) / fs
    f0, f1 = 30.0, 300.0
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / t[-1] * t**2)
    h = np.sin(phase) * np.hanning(n)
    H = np.fft.rfft(h)
    norm = h @ h
    print("2. phase shift (deg), <h_phi, h>/<h, h>, cos(phi):")
    for deg in (0, 30, 60, 90, 120, 180):
        phi = math.radians(deg)
        h_phi = np.fft.irfft(H * np.exp(1j * phi), n)
        print(f"   {deg:4d}  {h_phi @ h / norm:+.6f}  {math.cos(phi):+.6f}")


def item3(path):
    with h5py.File(path, "r") as f:
        p = {k: f["parameters"][k][...] for k in f["parameters"]}
    m1, m2 = p["mass_1"], p["mass_2"]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    counts, _ = np.histogram(mc, 10)
    print(f"3. {len(mc)} waveforms; source-frame chirp mass "
          f"{mc.min():.4f} to {mc.max():.4f}; counts in ten equal bins: "
          f"{counts.tolist()}")
    for k in ("a_1", "a_2", "tilt_1", "tilt_2", "phase", "psi", "theta_jn",
              "dec", "ra"):
        print(f"   {k}: {p[k].min():.4g} to {p[k].max():.4g}")


def item4():
    lin = torch.nn.Linear(8192, 4096, bias=False)
    w = lin.weight.detach()
    print(f"4. 1/sqrt(8192) = {1 / math.sqrt(8192):.6f}; default nn.Linear "
          f"weights span {w.min():.6f} to {w.max():.6f}")
    steps = 200 * 782
    opt = torch.optim.AdamW(lin.parameters(), lr=1e-5, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-5, pct_start=0.115, total_steps=steps
    )
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    lrs = np.array(lrs)
    print(f"   one-cycle over {steps} steps: first {lrs[0]:.3e}, largest "
          f"{lrs.max():.3e} at step {lrs.argmax()} "
          f"({lrs.argmax() / steps:.4f} of the steps), last {lrs[-1]:.3e}; "
          f"samples {steps * 64}")


def item5():
    import wandb

    api = wandb.Api()
    print("5. templates, best AUROC (epoch), loss epoch 0, lowest loss, "
          "loss epoch 199, lowest AUROC")
    for k in TEMPLATES:
        run = api.run(RUNS + f"mnet_shallow_h1_t{k}_train_snr4_prob0.5")
        a = np.array([r["validation/TimeSlideAUROC_epoch"] for r in
                      run.scan_history(keys=["validation/TimeSlideAUROC_epoch"])])
        by_epoch = {r["epoch"]: r["train/loss_epoch"] for r in
                    run.scan_history(keys=["train/loss_epoch", "epoch"])}
        loss = np.array([by_epoch[e] for e in sorted(by_epoch)])
        print(f"   {k:5d}  {a.max():.4f} ({a.argmax()})  {loss[0]:.4f}  "
              f"{loss.min():.4f}  {loss[-1]:.4f}  {a.min():.4f}  "
              f"[{len(a)} validations, {len(loss)} epochs]")


if __name__ == "__main__":
    item1()
    item2()
    item3(sys.argv[1])
    item4()
    item5()
