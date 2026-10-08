"""Matched-filter benchmark on dumped test windows.

Bank: waveforms drawn from the injection file, on a chirp-mass grid
log-spaced at 0.5% over the prior, at mass ratios 0.5, 0.7, 0.9, lowest
spin among candidates. Each template is whitened by each window's own
whitening response; complex SNR per detector; network SNR^2 = rho_H^2 +
max over +-10 ms of rho_L^2; lags restricted to where the merger can sit.

Outputs per window: best-template chirp mass, max network SNR.
"""

import sys

import h5py
import numpy as np
import torch
from ml4gw.spectral import truncate_inverse_power_spectrum

WAVE = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
FS = 2048
L = 8192              # 4 s window
N = 2 * L             # linear correlation length
MERGE_T = 7680        # template merger index inside its 4 s frame (3.75 s)
MAX_LAG = 600         # samples, covers merger in [3.5, 4.0] s plus margin
COINC = 20            # samples, 10 ms
FDURATION, HIGHPASS = 1.0, 20.0
STEP, QS = 0.005, (0.5, 0.7, 0.9)

win_file, out_file = sys.argv[1], sys.argv[2]
dev = torch.device("cuda")


def build_bank():
    f = h5py.File(WAVE)
    p = f["parameters"]
    m1, m2 = p["mass_1"][:], p["mass_2"][:]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    spin = p["a_1"][:] + p["a_2"][:]
    grid = np.exp(np.arange(np.log(mc.min()), np.log(mc.max()), STEP))
    chosen, labels = [], []
    for g in grid:
        near = np.where(np.abs(np.log(mc / g)) < STEP / 4)[0]
        if not len(near):
            continue
        for qq in QS:
            cand = near[np.abs(q[near] - qq) < 0.05]
            if not len(cand):
                continue
            i = cand[np.argmin(spin[cand])]
            if i not in chosen:
                chosen.append(i)
    chosen = np.sort(np.array(chosen))
    hp = f["waveforms/plus"][chosen]
    hc = f["waveforms/cross"][chosen]
    # align each template on its own amplitude peak
    bank = np.zeros((len(chosen), L), np.float32)
    taper = np.ones(L, np.float32)
    ramp = int(0.25 * FS)
    taper[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    for j in range(len(chosen)):
        peak = int(np.argmax(hp[j] ** 2 + hc[j] ** 2))
        padded = np.pad(hp[j], (L, L))
        start = peak + L - MERGE_T
        bank[j] = padded[start : start + L] * taper
    return torch.tensor(bank, device=dev), mc[chosen], q[chosen]


bank, bank_mc, bank_q = build_bank()
print(f"bank: {len(bank)} templates, chirp mass {bank_mc.min():.3f} to {bank_mc.max():.3f}")
H = torch.fft.rfft(bank, n=N)                      # (T, F)
F = H.shape[-1]
w_parseval = torch.full((F,), 2.0, device=dev)
w_parseval[0] = w_parseval[-1] = 1.0

wf = h5py.File(win_file)
X_all = wf["X"]
psd_all = wf["psds"]
n = X_all.shape[0]
best_mc = np.zeros(n, np.float32)
best_q = np.zeros(n, np.float32)
max_snr = np.zeros(n, np.float32)
lag_idx = torch.cat([torch.arange(N - MAX_LAG, N), torch.arange(0, MAX_LAG + 1)]).to(dev)

B = 4
for a in range(0, n, B):
    b = min(a + B, n)
    X = torch.tensor(X_all[a:b], device=dev)                      # (b, 2, L)
    psd = torch.tensor(psd_all[a:b], device=dev).double()         # (b, 2, f)
    psd = torch.nn.functional.interpolate(psd, size=(F,), mode="linear")
    W = torch.nan_to_num(
        truncate_inverse_power_spectrum(psd, FDURATION, FS, HIGHPASS) ** -0.5
    ).float()                                                      # (b, 2, F)
    D = torch.fft.rfft(X, n=N)                                     # (b, 2, F)
    HW = H[None, None] * W[:, :, None]                             # (b, 2, T, F)
    sigma = ((HW.abs() ** 2) * w_parseval).sum(-1).div(N).sqrt()  # (b, 2, T)
    prod = D[:, :, None] * HW.conj()
    spec = torch.zeros(*prod.shape[:-1], N, dtype=prod.dtype, device=dev)
    spec[..., :F] = prod
    spec[..., 1 : F - 1] *= 2
    z = torch.fft.ifft(spec, n=N)[..., lag_idx]                    # (b, 2, T, lags)
    rho = z.abs() / sigma[..., None].clamp_min(1e-30)
    rhoL = torch.nn.functional.max_pool1d(
        rho[:, 1].reshape(-1, 1, rho.shape[-1]) ** 2, 2 * COINC + 1, stride=1, padding=COINC
    ).reshape(rho[:, 1].shape)
    net = (rho[:, 0] ** 2 + rhoL).amax(-1).sqrt()                  # (b, T)
    val, j = net.max(-1)
    j = j.cpu().numpy()
    best_mc[a:b], best_q[a:b] = bank_mc[j], bank_q[j]
    max_snr[a:b] = val.cpu().numpy()
    if a % 2000 == 0:
        print(f"{a}/{n}", flush=True)

with h5py.File(out_file, "w") as f:
    f["mf_chirp_mass"] = best_mc
    f["mf_mass_ratio"] = best_q
    f["mf_snr"] = max_snr
print(f"DONE {out_file}")
