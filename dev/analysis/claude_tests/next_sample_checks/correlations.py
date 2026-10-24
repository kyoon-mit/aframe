"""How much each noise sample is correlated with earlier ones.

Raw strain and whitened 4 s windows (as test1_likelihood.py) of the 21 O3a
training files, per detector; whitened windows with largest |value| > 8 are
left out. Prints the correlation with the sample 1, 2, 3, 10 and 100 steps
back, the share of a sample's variance predictable from the one before
(lag-1 correlation squared), and from the whole past of a 4 s window
(1 - final Levinson-Durbin error / variance, on the mean autocovariance),
and from the infinite past by the Kolmogorov-Szego formula on the mean
periodogram, 1 - exp(mean log S) / mean S, over all frequencies and over
20 Hz and above only.

usage: correlations.py
"""

import sys

import h5py
import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test1_likelihood import LOUD, N, files, levinson, windows  # noqa: E402
from train.experimental.noise_model.sources import Whitening  # noqa: E402

LAGS = (1, 2, 3, 10, 100)


def main():
    paths = files("O3a")[:-2]
    raw = np.zeros((2, max(LAGS) + 1))
    for f in paths:
        with h5py.File(f) as h:
            for i, ifo in enumerate(("H1", "L1")):
                x = h[ifo][:].astype(np.float64)
                x -= x.mean()
                raw[i] += [x @ x if k == 0 else x[k:] @ x[:-k] for k in range(max(LAGS) + 1)]
    w = Whitening(2048.0, 4.0, 20.0, 1.0, 20.0, fftlength=2.0).to("cuda")
    acov = torch.zeros(2, N, dtype=torch.float64, device="cuda")
    count = torch.zeros(2, dtype=torch.float64, device="cuda")
    power = torch.zeros(2, N // 2 + 1, dtype=torch.float64, device="cuda")
    for f in paths:
        for _, x in windows(f, w):
            keep = (x.abs().amax(-1) <= LOUD).double()
            p = torch.fft.rfft(x, n=2 * N).abs().square()
            lagged = torch.fft.irfft(p, n=2 * N)[..., :N] / N
            acov += (lagged * keep[..., None]).sum(0)
            power += (torch.fft.rfft(x).abs().square() / N * keep[..., None]).sum(0)
            count += keep.sum(0)
    acov = (acov / count[:, None]).cpu().numpy()
    power = (power / count[:, None]).cpu().numpy()
    freqs = np.fft.rfftfreq(N, 1 / 2048.0)
    print(f"{len(paths)} files; clean whitened windows per detector: {count.cpu().numpy()}")
    for i, ifo in enumerate(("H1", "L1")):
        rr = raw[i] / raw[i][0]
        rw = acov[i] / acov[i][0]
        _, full_e = levinson(acov[i], ())
        print(f"{ifo} raw strain: " + ", ".join(f"lag {k} {rr[k]:.6f}" for k in LAGS)
              + f"; predictable from previous sample {rr[1] ** 2:.6f}")
        print(f"{ifo} whitened: " + ", ".join(f"lag {k} {rw[k]:+.4f}" for k in LAGS)
              + f"; variance {acov[i][0]:.4f}; predictable from previous sample "
              f"{rw[1] ** 2:.4f}, from whole past {1 - full_e[-1] / acov[i][0]:.4f}")
        for name, band in (("all", freqs >= 0), ("20 Hz and above", freqs >= 20)):
            S = power[i][band]
            print(f"  {name}: infinite past {1 - np.exp(np.log(S).mean()) / S.mean():.4f}, "
                  f"share of variance {S.sum() / power[i].sum():.4f}")


if __name__ == "__main__":
    main()
