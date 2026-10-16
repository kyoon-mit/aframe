"""Real-waveform template bank on the benchmark windows: peak network SNR^2
of every template in every window, for offline argmax/interpolation studies.

Bank: injection-file waveforms (h_plus), merger at sample 7680 of a 4 s frame,
first 0.25 s tapered. Two sub-banks:
  fine: log-Mc grid at 0.25%, mass ratio near 0.8
  q3:   log-Mc grid at 0.5%, mass ratio near 0.5, 0.7, 0.9 (the benchmark's)
Whitening of the templates: each window's own PSD ("window") or one average
PSD per detector from O3a training windows ("avg").
Network SNR^2 = rho_H^2 + max over +-10 ms of rho_L^2, max over lags where
the merger can sit.

usage: bank_check.py POP WHITEN FS OUT [NMAX]
"""

import sys
import time

import h5py
import numpy as np
import torch
from ml4gw.spectral import truncate_inverse_power_spectrum

WAVE = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
TRAIN = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5"
BENCH = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_"
         "c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")
FS0, L0, MERGE0 = 2048, 8192, 7680
FDURATION, HIGHPASS = 1.0, 20.0

pop, whiten, fs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
nmax = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] != "all" else None
CHIS = (-0.2, -0.1, 0.0, 0.1, 0.2)
dev = torch.device("cuda")
dec = FS0 // fs
L, MERGE = L0 // dec, MERGE0 // dec
MAX_LAG, COINC = 600 // dec, 20 // dec
N = L + 1024 // dec                 # >= L + MAX_LAG: no wraparound at kept lags
F = N // 2 + 1


def pick(mc, q, spin, grid, qs, step):
    chosen = []
    for g in grid:
        near = np.where(np.abs(np.log(mc / g)) < step / 4)[0]
        for qq in qs:
            cand = near[np.abs(q[near] - qq) < 0.05]
            if len(cand):
                chosen.append(cand[np.argmin(spin[cand])])
    return np.array(chosen)


def pick_chi(mc, q, chi, inplane, grid, step):
    chosen, chis = [], []
    for g in grid:
        near = np.where((np.abs(np.log(mc / g)) < step / 4) & (np.abs(q - 0.8) < 0.1))[0]
        for c in CHIS:
            cand = near[np.abs(chi[near] - c) < 0.03]
            if len(cand):
                chosen.append(cand[np.argmin(np.abs(chi[cand] - c) / 0.03 + inplane[cand])])
                chis.append(c)
    return np.array(chosen), np.array(chis)


def build_bank():
    f = h5py.File(WAVE)
    p = f["parameters"]
    m1, m2 = p["mass_1"][:], p["mass_2"][:]
    mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
    q = np.minimum(m1, m2) / np.maximum(m1, m2)
    a1, a2, t1, t2 = p["a_1"][:], p["a_2"][:], p["tilt_1"][:], p["tilt_2"][:]
    chi = (m1 * a1 * np.cos(t1) + m2 * a2 * np.cos(t2)) / (m1 + m2)
    inplane = a1 * np.abs(np.sin(t1)) + a2 * np.abs(np.sin(t2))
    lo, hi = np.log(0.85), np.log(2.75)
    rows, chis = pick_chi(mc, q, chi, inplane, np.exp(np.arange(lo, hi + 1e-9, 0.005)), 0.005)
    tag = np.round((chis + 0.2) / 0.1).astype(int) + 10   # 10..14 = chi -0.2..0.2
    order = np.argsort(rows)
    hp = np.empty((len(rows), 65536))
    hp[order] = f["waveforms/plus"][rows[order]]
    bank = np.zeros((len(rows), L0), np.float32)
    taper = np.ones(L0, np.float32)
    ramp = int(0.25 * FS0)
    taper[:ramp] = 0.5 * (1 - np.cos(np.pi * np.arange(ramp) / ramp))
    hc = np.empty_like(hp)
    hc[order] = f["waveforms/cross"][rows[order]]
    for j in range(len(rows)):
        peak = int(np.argmax(hp[j] ** 2 + hc[j] ** 2))
        padded = np.pad(hp[j], (L0, L0))
        start = peak + L0 - MERGE0
        bank[j] = padded[start : start + L0] * taper
    return bank, mc[rows], q[rows], tag, rows


def decimate(x):
    """Band-limit to fs/2 and resample by FFT truncation (last axis)."""
    if dec == 1:
        return x
    n = x.shape[-1]
    X = torch.fft.rfft(x, dim=-1)[..., : n // dec // 2 + 1]
    return torch.fft.irfft(X, n=n // dec, dim=-1) / dec


def whitening(psd):
    """Inverse ASD of the aframe whitener on this script's F bins. psd: (b, 2, f)."""
    psd = torch.nn.functional.interpolate(psd, size=(N // 2 * dec + 1,), mode="linear")
    W = truncate_inverse_power_spectrum(psd, FDURATION, FS0, HIGHPASS) ** -0.5
    return torch.nan_to_num(W)[..., :F].float()


t0 = time.time()
bank, bmc, bq, tag, rows = build_bank()
chi_of_rows = (tag - 12) * 0.1
print(f"bank {len(bank)} templates, per chi", [int(np.sum(tag == t)) for t in range(10, 15)], flush=True)
H = torch.fft.rfft(decimate(torch.tensor(bank, device=dev)), n=N)       # (T, F)
w_parseval = torch.full((F,), 2.0, device=dev)
w_parseval[0] = w_parseval[-1] = 1.0
lag_idx = torch.cat([torch.arange(N - MAX_LAG, N), torch.arange(0, MAX_LAG + 1)]).to(dev)

if whiten == "avg":
    psd_avg = torch.tensor(h5py.File(TRAIN)["psds"][:4000].mean(0), device=dev)[None]  # (1, 2, f)
    W_avg = whitening(psd_avg)                                             # (1, 2, F)
    HW_avg = H[None] * W_avg[0][:, None]                                   # (2, T, F)
    sigma_avg = ((HW_avg.abs() ** 2) * w_parseval).sum(-1).div(N).sqrt()  # (2, T)

wf = h5py.File(f"{BENCH}/{pop}/windows.h5")
n = wf["X"].shape[0] if nmax is None else min(nmax, wf["X"].shape[0])
X_all = wf["X"][:n]
psd_all = wf["psds"][:n]
peak = np.zeros((n, len(bank)), np.float32)
lag = np.zeros((n, len(bank)), np.int16)
B = 16
for a in range(0, n, B):
    b = min(a + B, n)
    X = decimate(torch.tensor(X_all[a:b], device=dev))
    D = torch.fft.rfft(X, n=N)                                             # (b, 2, F)
    if whiten == "window":
        W = whitening(torch.tensor(psd_all[a:b], device=dev))
        HW = H[None, None] * W[:, :, None]                                 # (b, 2, T, F)
        sigma = ((HW.abs() ** 2) * w_parseval).sum(-1).div(N).sqrt()
        prod = D[:, :, None] * HW.conj()
    else:
        sigma = sigma_avg[None].expand(b - a, -1, -1)
        prod = D[:, :, None] * HW_avg.conj()[None]
    spec = torch.zeros(*prod.shape[:-1], N, dtype=prod.dtype, device=dev)
    spec[..., :F] = prod
    spec[..., 1 : F - 1] *= 2
    z = torch.fft.ifft(spec, n=N)[..., lag_idx]
    del spec, prod
    rho2 = z.abs() ** 2 / sigma[..., None].clamp_min(1e-30) ** 2           # (b, 2, T, lags)
    rhoL = torch.nn.functional.max_pool1d(
        rho2[:, 1].reshape(-1, 1, rho2.shape[-1]), 2 * COINC + 1, stride=1, padding=COINC
    ).reshape(rho2[:, 1].shape)
    net = rho2[:, 0] + rhoL                                                # (b, T, lags)
    v, i = net.max(-1)
    peak[a:b] = v.cpu().numpy()
    lag[a:b] = i.cpu().numpy().astype(np.int16)
    if a % 1600 == 0:
        print(f"{a}/{n} {time.time() - t0:.0f}s", flush=True)

np.savez(out, bank_chi=chi_of_rows, peak=peak, lag=lag, bank_mc=bmc, bank_q=bq, tag=tag, rows=rows,
         true_mc=wf["params/chirp_mass"][:n], true_q=wf["params/mass_ratio"][:n],
         snr=wf["params/snr"][:n])
print(f"DONE {out} {time.time() - t0:.0f}s", flush=True)
