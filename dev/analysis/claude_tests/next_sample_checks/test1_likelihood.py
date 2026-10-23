"""Test 1: log-likelihood of held-out noise under the next-sample network and
under Gaussians with the spectrum of the training noise.

Windows: every back-to-back 4 s window of each file, whitened as aframe's
training data (noise_model.sources.Whitening, as the 19 October noise study).
Training spectra: the 21 O3a training files of k3_shared_mean_asinh.yaml.
Held out: its 2 O3a validation files, and all O3b files.

Gaussian models. Spectrum S at 2N frequencies from periodograms of the
training windows zero-padded to 2N, either the mean over windows or the
median / ln 2 (typical window). Its inverse FFT gives lags 0..N-1 of a
positive definite autocovariance. Levinson-Durbin gives the best linear
predictor of each order; the likelihood of a window is the chain rule over
its samples, sample t predicted from min(t, p) earlier samples. p = N - 1 is
the exact Gaussian with that covariance (checked against a Cholesky log
determinant). Spectra pooled over H1 and L1, as the network is, and per
detector.

usage: test1_likelihood.py <ckpt> <out dir>
"""

import glob
import math
import sys
import time

import h5py
import numpy as np
import torch

from train.experimental.next_sample.arch import CausalS4Noise
from train.experimental.next_sample.task import mixture_log_prob
from train.experimental.noise_model.sources import Whitening

DATA = "/home/kyoon/SSM-BNS/DATA"
N = 8192
ORDERS = (0, 32, 256, 2048, N - 1)
LOUD = 8.0
HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)
dev = "cuda"


def files(run):
    return sorted(glob.glob(f"{DATA}/{run}_H1_L1_2048Hz/background/*.hdf5"))


def windows(path, w, batch=128):
    """Whitened back-to-back windows of one file, (B, 2, N) float64 batches."""
    with h5py.File(path) as h:
        data = torch.from_numpy(np.stack([h["H1"][:], h["L1"][:]])).to(dev)
    starts = list(range(0, data.shape[1] - w.stretch_size + 1, w.window_size))
    for b in range(0, len(starts), batch):
        s = starts[b : b + batch]
        stretches = torch.stack([data[:, k : k + w.stretch_size] for k in s]).float()
        with torch.no_grad():
            yield s, w(stretches).double()


def levinson(r, orders):
    """Unit lower-triangular L (N, N) per order with L x = prediction errors,
    and the error variances E (N,) per order."""
    n = len(r)
    a = np.zeros(0)
    err = r[0]
    full_a, full_e = [a], [err]
    for m in range(1, n):
        k = (r[m] - a @ r[m - 1 : 0 : -1]) / err if m > 1 else r[1] / err
        a = np.concatenate([a - k * a[::-1], [k]])
        err = err * (1 - k * k)
        if not 0 < err:
            raise ValueError(f"not positive definite at order {m}")
        full_a.append(a)
        full_e.append(err)
    out = {}
    for p in orders:
        L = np.eye(n)
        E = np.empty(n)
        for t in range(n):
            q = min(t, p)
            if q:
                L[t, t - q : t] = -full_a[q][::-1]
            E[t] = full_e[q]
        out[p] = (torch.from_numpy(L).to(dev), torch.from_numpy(E).to(dev))
    return out, np.array(full_e)


def main():
    ckpt, out = sys.argv[1:3]
    torch.manual_seed(0)
    w = Whitening(2048.0, 4.0, 20.0, 1.0, 20.0, fftlength=2.0).to(dev)
    o3a = files("O3a")
    train_files, val_files = o3a[:-2], o3a[-2:]
    sets = {"O3a_val": val_files, "O3b": files("O3b")}

    # training periodograms, zero-padded to 2N
    t0 = time.time()
    power = []
    loud_train = 0
    for f in train_files:
        for _, x in windows(f, w):
            loud_train += int((x.abs().amax(-1) > LOUD).sum())
            power.append((torch.fft.rfft(x, n=2 * N).abs().square() / N).float().cpu())
    power = torch.cat(power)  # (W, 2, N + 1)
    print(f"training windows {len(power)} per detector, loud {loud_train} "
          f"detector-windows, {time.time() - t0:.0f} s", flush=True)
    spectra = {
        "mean_pooled": power.double().mean((0, 1)),
        "typical_pooled": power.flatten(0, 1).median(0).values.double() / math.log(2),
    }
    for i, ifo in enumerate(("H1", "L1")):
        spectra[f"mean_{ifo}"] = power[:, i].double().mean(0)
        spectra[f"typical_{ifo}"] = power[:, i].median(0).values.double() / math.log(2)
    del power

    models = {}
    checks = {}
    for name, S in spectra.items():
        r = torch.fft.irfft(S, n=2 * N)[:N].numpy()
        preds, full_e = levinson(r, ORDERS)
        R = torch.from_numpy(r[np.abs(np.arange(N)[:, None] - np.arange(N)[None])]).to(dev)
        chol = torch.linalg.cholesky(R)
        logdet_chol = 2 * chol.diagonal().log().sum().item()
        logdet_lev = np.log(full_e).sum()
        checks[name] = (logdet_chol, logdet_lev)
        print(f"{name}: r0 {r[0]:.4f} logdet Cholesky {logdet_chol:.6f} "
              f"Levinson {logdet_lev:.6f}", flush=True)
        del R, chol
        models[name] = preds

    net = CausalS4Noise(num_components=3, d_model=64, d_state=32, n_layers=4,
                        dropout=0.0, dt_min=1e-4, dt_max=10.0)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.removeprefix("model."): v for k, v in sd.items()})
    net = net.to(dev).eval()

    save = {"orders": np.array(ORDERS), "spectra": np.array(list(spectra))}
    for name, (a, b) in checks.items():
        save[f"logdet_{name}"] = np.array([a, b])
    for split, paths in sets.items():
        rows = {"file": [], "start": [], "maxabs": [], "network": []}
        prof_net = torch.zeros(2, N, dtype=torch.float64, device=dev)
        prof_gauss = {}
        prof_count = torch.zeros(2, device=dev, dtype=torch.float64)
        for name in models:
            for p in ORDERS:
                rows[f"{name}_p{p}"] = []
        t0 = time.time()
        for fi, f in enumerate(paths):
            for s, x in windows(f, w):
                B = len(x)
                flat = x.reshape(-1, N)
                with torch.no_grad():
                    mean, log_sigma, logits = net(flat.float()[:, None])
                    lp = mixture_log_prob(flat.float(), mean, log_sigma, logits).double()
                nll_net = -lp.reshape(B, 2, N)
                rows["network"].append(nll_net.sum(-1).cpu())
                clean = (x.abs().amax(-1) <= LOUD)
                prof_net += (nll_net * clean[..., None]).sum(0)
                prof_count += clean.sum(0)
                for name, preds in models.items():
                    for p, (L, E) in preds.items():
                        e = flat @ L.T
                        nll = 0.5 * e.square() / E + 0.5 * E.log() + HALF_LOG_2PI
                        nll = nll.reshape(B, 2, N)
                        rows[f"{name}_p{p}"].append(nll.sum(-1).cpu())
                        if p == N - 1:
                            prof_gauss.setdefault(name, torch.zeros_like(prof_net))
                            prof_gauss[name] += (nll * clean[..., None]).sum(0)
                rows["file"] += [fi] * B
                rows["start"] += s
                rows["maxabs"].append(x.abs().amax(-1).cpu())
            print(f"{split} {f.split('/')[-1]} {time.time() - t0:.0f} s", flush=True)
        for k, v in rows.items():
            save[f"{split}_{k}"] = torch.cat(v).numpy() if isinstance(v[0], torch.Tensor) else np.array(v)
        save[f"{split}_profile_network"] = (prof_net / prof_count[:, None]).cpu().numpy()
        for name, v in prof_gauss.items():
            save[f"{split}_profile_{name}"] = (v / prof_count[:, None]).cpu().numpy()
        save[f"{split}_files"] = np.array(paths)
    np.savez_compressed(f"{out}/test1.npz", **save)
    print("->", f"{out}/test1.npz")


if __name__ == "__main__":
    main()
