"""Strongest lines of the typical whitened window: per file and detector,
median over the clean windows (largest value at most 8) among the first 600
back-to-back windows of each O3a and O3b file of the periodogram,
divided by ln 2; power relative to white (1).

usage: test1_lines.py
"""

import glob

import h5py
import numpy as np
import torch

from train.experimental.noise_model.sources import Whitening

DATA = "/home/kyoon/SSM-BNS/DATA"
N = 8192


def typical_power(path, w, count=600):
    with h5py.File(path) as h:
        data = torch.from_numpy(np.stack([h["H1"][:], h["L1"][:]]))
    starts = list(range(0, data.shape[1] - w.stretch_size + 1, w.window_size))[:count]
    power = []
    for b in range(0, len(starts), 64):
        s = starts[b : b + 64]
        x = w(torch.stack([data[:, k : k + w.stretch_size] for k in s]).float()).double()
        p = torch.fft.rfft(x).abs().square() / N
        p[x.abs().amax(-1) > 8] = float("nan")
        power.append(p)
    return torch.cat(power).nanmedian(0).values.numpy() / np.log(2)


def main():
    torch.set_num_threads(16)
    w = Whitening(2048.0, 4.0, 20.0, 1.0, 20.0, fftlength=2.0)
    freqs = np.fft.rfftfreq(N, 1 / 2048)
    for run in ("O3a", "O3b"):
        for path in sorted(glob.glob(f"{DATA}/{run}_H1_L1_2048Hz/background/*.hdf5")):
            med = typical_power(path, w)
            row = f"{run} {path.split('/')[-1]}"
            for i, ifo in enumerate(("H1", "L1")):
                top = np.argsort(med[i])[::-1][:3]
                row += f" | {ifo} mean {med[i].mean():.2f}: " + ", ".join(
                    f"{freqs[k]:.2f} Hz {med[i, k]:.0f}" for k in top)
            print(row, flush=True)


if __name__ == "__main__":
    main()
