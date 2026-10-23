"""Test 2: detection with the true template, matched filter against the
likelihood ratio under the next-sample network.

Windows: the data config's test loader (O3b) and injection, with its random
sign flip and time reversal switched off. Each window gives the whitened
signal h (X_clean) and noise n = X - h. With h known, for d = n (no signal)
and d = n + h (signal):

    matched filter     sum over detectors and samples of d h, / |h|
    network            log p(d - h) - log p(d), summed over detectors
    Gaussian           the same under the Gaussian with the typical
                       spectrum of each detector's training windows, exact
                       covariance (test1_likelihood.py)

usage: test2_detection.py <config.yaml> <ckpt> <out dir> [windows]
"""

import glob
import math
import sys
import time

import numpy as np
import torch
import yaml

from train.experimental.next_sample.arch import CausalS4Noise
from train.experimental.next_sample.task import mixture_log_prob
from train.experimental.noise_model.sources import Whitening
from train.experimental.slic.dump_windows import build_data_module

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test1_likelihood import LOUD, N, files, levinson, windows  # noqa: E402

dev = "cuda"


def gaussian_models():
    """Per detector (L, E) of the full-order predictor for the typical
    spectrum of the O3a training windows."""
    w = Whitening(2048.0, 4.0, 20.0, 1.0, 20.0, fftlength=2.0).to(dev)
    power = []
    for f in files("O3a")[:-2]:
        for _, x in windows(f, w):
            power.append((torch.fft.rfft(x, n=2 * N).abs().square() / N).float().cpu())
    power = torch.cat(power)
    out = []
    for i in range(2):
        S = power[:, i].median(0).values.double() / math.log(2)
        r = torch.fft.irfft(S, n=2 * N)[:N].numpy()
        preds, _ = levinson(r, (N - 1,))
        out.append(preds[N - 1])
    return out


def main():
    config, ckpt, out = sys.argv[1:4]
    count = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    torch.set_num_threads(16)
    with open(config) as f:
        seed = yaml.safe_load(f)["seed_everything"]
    import lightning.pytorch as pl

    pl.seed_everything(seed)
    dm = build_data_module(config, "test")
    dm.inverter.prob = 0.0
    dm.reverser.prob = 0.0
    loader = dm.test_dataloader()

    gauss = gaussian_models()
    net = CausalS4Noise(num_components=3, d_model=64, d_state=32, n_layers=4,
                        dropout=0.0, dt_min=1e-4, dt_max=10.0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
    net = net.to(dev).eval()

    def log_p(x):
        """(B, 2, N) -> (B, 2) log-likelihood under the network."""
        flat = x.reshape(-1, 1, N).float()
        with torch.no_grad():
            mean, log_sigma, logits = net(flat)
            lp = mixture_log_prob(flat[:, 0], mean, log_sigma, logits).double()
        return lp.sum(-1).reshape(len(x), 2)

    keys = ("snr", "h_norm", "maxabs_noise", "mf_noise", "mf_signal",
            "net_noise", "net_signal", "gauss_noise", "gauss_signal")
    rows = {k: [] for k in keys}
    total = 0
    t0 = time.time()
    for batch in loader:
        batch = dm.on_before_batch_transfer(batch, 0)
        X, X_clean, y, params = dm.on_after_batch_transfer(batch, 0)
        if not (y[:, 0] > 0).all():
            raise ValueError("expected a signal in every window")
        h = X_clean.to(dev).double()
        n = X.to(dev).double() - h
        norm = h.square().sum((1, 2)).sqrt()
        rows["snr"].append(params["snr"].double())
        rows["h_norm"].append(norm.cpu())
        rows["maxabs_noise"].append(n.abs().amax(-1).cpu())
        rows["mf_noise"].append(((n * h).sum((1, 2)) / norm).cpu())
        rows["mf_signal"].append((((n + h) * h).sum((1, 2)) / norm).cpu())
        lp_n, lp_nmh, lp_nph = log_p(n), log_p(n - h), log_p(n + h)
        rows["net_noise"].append((lp_nmh - lp_n).cpu())
        rows["net_signal"].append((lp_n - lp_nph).cpu())
        g_noise, g_signal = [], []
        for i, (L, E) in enumerate(gauss):
            eh, en = h[:, i] @ L.T, n[:, i] @ L.T
            cross = (eh * en / E).sum(-1)
            hh = (eh.square() / E).sum(-1)
            g_noise.append(cross - 0.5 * hh)
            g_signal.append(cross + 0.5 * hh)
        rows["gauss_noise"].append(torch.stack(g_noise, 1).cpu())
        rows["gauss_signal"].append(torch.stack(g_signal, 1).cpu())
        total += len(X)
        if total // 2000 != (total - len(X)) // 2000:
            print(f"{total} windows {time.time() - t0:.0f} s", flush=True)
        if total >= count:
            break
    save = {k: torch.cat(v).numpy()[:count] for k, v in rows.items()}
    save["loud"] = LOUD
    np.savez_compressed(f"{out}/test2.npz", **save)
    print("->", f"{out}/test2.npz")


if __name__ == "__main__":
    main()
