"""Test 3: the same injections in ideal noise, independent unit Gaussian
samples, with a perfect next-sample noise model.

Windows: the data config's test loader (as test2_detection.py), keeping only
the whitened signal h (X_clean). The noise n is drawn independently, N(0, 1)
in every sample of both detectors, so the SNR of a window is exactly |h|.
The perfect noise model predicts N(0, 1) for every sample whatever came
before, so its best guess of the noise is 0 and data minus predicted noise is
the data itself. Per window, with d = h + n:

    snr          |h|, the true SNR in this noise
    snr_label    aframe's sampled SNR
    peak         largest |h| in one sample, in noise widths
    sig_frac     share of the leftover's power that is signal, |h|^2 / |d|^2
    mf_noise     n h / |h|, matched filter with no signal
    mf_signal    d h / |h|
    draw_noise   (n - m) h / |h|, subtracting a random draw m ~ N(0, 1)
    draw_signal  (d - m) h / |h|
    draw_frac    sig_frac of the leftover d - m
    draw_amp     (d - m) h / |h|^2
    draw_shrink  shrink for the leftover d - m, a = |h|^2 / (|h|^2 + 2 samples)
    amp          d h / |h|^2, fitted amplitude of the known template
    shrink       share of the signal's energy recovered by the best guess
                 a d with no template, a = |h|^2 / (|h|^2 + samples), given
                 the signal's energy: 1 - |a d - h|^2 / |h|^2

usage: test3_white_noise.py <config.yaml> <out dir> [windows]
"""

import sys
import time

import lightning.pytorch as pl
import numpy as np
import torch
import yaml

from train.experimental.slic.dump_windows import build_data_module


def white_noise_windows(config, count, seed):
    """(h, snr_label, d, n, m) batches: the config's test injections in
    independent unit Gaussian noise, with a second draw m for the random
    subtraction."""
    with open(config) as f:
        pl.seed_everything(yaml.safe_load(f)["seed_everything"])
    dm = build_data_module(config, "test")
    dm.inverter.prob = 0.0
    dm.reverser.prob = 0.0
    gen = torch.Generator().manual_seed(seed)
    total = 0
    for batch in dm.test_dataloader():
        batch = dm.on_before_batch_transfer(batch, 0)
        _, X_clean, y, params = dm.on_after_batch_transfer(batch, 0)
        if not (y[:, 0] > 0).all():
            raise ValueError("expected a signal in every window")
        h = X_clean.cpu().double()
        n = torch.randn(h.shape, generator=gen, dtype=torch.float64)
        m = torch.randn(h.shape, generator=gen, dtype=torch.float64)
        yield h, params["snr"].cpu().double(), h + n, n, m
        total += len(h)
        if total >= count:
            return


def main():
    config, out = sys.argv[1:3]
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    keys = ("snr", "snr_label", "peak", "sig_frac", "mf_noise", "mf_signal",
            "draw_noise", "draw_signal", "amp", "shrink", "draw_frac",
            "draw_amp", "draw_shrink")
    rows = {k: [] for k in keys}
    nll, samples = 0.0, 0
    t0 = time.time()
    for h, label, d, n, m in white_noise_windows(config, count, 0):
        size = h[0].numel()
        e = h.square().sum((1, 2))
        norm = e.sqrt()
        a = (e / (e + size))[:, None, None]
        rows["snr"].append(norm)
        rows["snr_label"].append(label)
        rows["peak"].append(h.abs().amax((1, 2)))
        rows["sig_frac"].append(e / d.square().sum((1, 2)))
        rows["mf_noise"].append((n * h).sum((1, 2)) / norm)
        rows["mf_signal"].append((d * h).sum((1, 2)) / norm)
        rows["draw_noise"].append(((n - m) * h).sum((1, 2)) / norm)
        rows["draw_signal"].append(((d - m) * h).sum((1, 2)) / norm)
        rows["amp"].append((d * h).sum((1, 2)) / e)
        rows["shrink"].append(1 - (a * d - h).square().sum((1, 2)) / e)
        r = d - m
        a2 = (e / (e + 2 * size))[:, None, None]
        rows["draw_frac"].append(e / r.square().sum((1, 2)))
        rows["draw_amp"].append((r * h).sum((1, 2)) / e)
        rows["draw_shrink"].append(1 - (a2 * r - h).square().sum((1, 2)) / e)
        nll += float((0.5 * n.square() + 0.5 * np.log(2 * np.pi)).sum())
        samples += n.numel()
        done = sum(len(v) for v in rows["snr"])
        if done // 4000 != (done - len(h)) // 4000:
            print(f"{done} windows {time.time() - t0:.0f} s", flush=True)
    save = {k: torch.cat(v).numpy()[:count] for k, v in rows.items()}
    save["samples_per_window"] = size
    save["noise_nll_per_sample"] = nll / samples
    np.savez_compressed(f"{out}/test3.npz", **save)
    print("noise NLL per sample under N(0, 1):", nll / samples)
    print("->", f"{out}/test3.npz")


if __name__ == "__main__":
    main()
