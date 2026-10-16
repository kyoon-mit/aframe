"""Whitened 4 s templates from the injection waveform file, bank rows (one per grid chirp mass).

Merger (peak of h+^2 + hx^2) at 3.75 s of the window (bank convention, inside the data's
3.5-4.0 s range). h_plus is whitened with ml4gw.spectral.whiten (average O3 PSD of the bank,
fduration 1 s, highpass 20 Hz) on a 5 s stretch and cropped to 4 s, as the data module does.
Checks: unit variance of whitened PSD-colored noise; SNR kept by the long_conv decimation;
time before merger of each frequency against the Newtonian formula.
"""
import json
import math
import os
import sys

import h5py
import numpy as np
import torch
from ml4gw.spectral import whiten

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kcommon import BANK, HERE, MERGE, f_of_tau, tau_of_f  # noqa: E402

torch.set_num_threads(8)
FS = 2048
L = 8192
z = np.load(BANK)
rows = z["rows"]
order = np.argsort(rows)
with h5py.File(str(z["waveform_file"]), "r") as f:
    hp = np.empty((len(rows), f["waveforms/plus"].shape[-1]))
    hc = np.empty_like(hp)
    hp[order] = f["waveforms/plus"][rows[order]]
    hc[order] = f["waveforms/cross"][rows[order]]
peak = np.argmax(hp ** 2 + hc ** 2, axis=1)
start = peak - MERGE - FS // 2                 # 0.5 s whitening margin each side
seg = np.stack([hp[j, s:s + L + FS] for j, s in enumerate(start)])
segc = np.stack([hc[j, s:s + L + FS] for j, s in enumerate(start)])
psd = torch.tensor(z["psd"], dtype=torch.float64)                 # (2, 2049), 0.5 Hz
X = torch.tensor(np.stack([seg, seg], 1))                        # (J, 2, 10240)
W = whiten(X, psd, 1.0, FS, highpass=20.0).numpy()               # (J, 2, 8192)
Xc = torch.tensor(np.stack([segc, segc], 1))
Wc = whiten(Xc, psd, 1.0, FS, highpass=20.0).numpy()

checks = {}
# 1. whitened PSD-colored Gaussian noise has unit variance
g = torch.Generator().manual_seed(1)
n = L + FS
white = torch.randn(64, 2, n, generator=g, dtype=torch.float64)
fr = torch.fft.rfftfreq(n, 1 / FS)
S = torch.nn.functional.interpolate(psd[None], size=(n // 2 + 1,), mode="linear")[0]   # same interpolation as whiten
col = torch.fft.irfft(torch.fft.rfft(white) * (S * FS / 2).sqrt(), n=n)
wn = whiten(col, psd, 1.0, FS, highpass=20.0)
checks["whitened_noise_std_H1_L1"] = [float(wn[:, i].std()) for i in range(2)]
checks["whitened_noise_std_expected_from_band"] = math.sqrt((1024 - 20) / 1024)
Pw = (torch.fft.rfft(wn).abs() ** 2).mean(0) / wn.shape[-1]        # per-bin power, unit for white
fw = torch.fft.rfftfreq(wn.shape[-1], 1 / FS)
checks["whitened_noise_per_bin_power_20_1024Hz_median_H1_L1"] = [float(Pw[i][fw >= 20].median()) for i in range(2)]
checks["whitened_noise_power_share_in_loudest_1pct_bins_H1_L1"] = [
    float(Pw[i][fw >= 20].sort(descending=True).values[: int((fw >= 20).sum()) // 100].sum() / Pw[i][fw >= 20].sum())
    for i in range(2)]
# 2. SNR kept by the decimation of LongConvEnergyNet (band below 512 Hz)
x = torch.tensor(W[:, 0])
spec = torch.fft.rfft(x, n=L)[..., : L // 4 + 1]
xd = torch.fft.irfft(spec, n=L // 2) / math.sqrt(2)
lo = torch.fft.irfft(torch.fft.rfft(x, n=L) * (torch.fft.rfftfreq(L, 1 / FS) < 512), n=L)
r = (xd.pow(2).sum(-1) / lo.pow(2).sum(-1)).numpy()
checks["decimated_snr2_over_below512_snr2_minmax"] = [float(r.min()), float(r.max())]
checks["below512_share_of_snr2_minmax"] = [float((lo.pow(2).sum(-1) / x.pow(2).sum(-1)).min()),
                                         float((lo.pow(2).sum(-1) / x.pow(2).sum(-1)).max())]
# 3. time before merger of 50/100/200/400 Hz from the waveform phase vs Newtonian
mc = z["chirp_mass"]
# analytic signal of h_plus (one-sided spectrum), any inclination
Nw = hp.shape[1]
Hs = np.fft.fft(hp, axis=1)
wgt = np.zeros(Nw); wgt[0] = 1; wgt[1:Nw // 2] = 2; wgt[Nw // 2] = 1
hpa = np.fft.ifft(Hs * wgt, axis=1)
ph = np.unwrap(np.angle(hpa), axis=1)
finst = np.gradient(ph, axis=1) * FS / (2 * np.pi)
k = FS // 32                                   # 1/32 s running mean
finst = np.apply_along_axis(lambda v: np.convolve(v, np.ones(k) / k, mode="same"), 1, finst)
tab = []
for j in (0, int(np.argmin(np.abs(mc - 1.2))), len(mc) - 1):
    row = {"chirp_mass": float(mc[j])}
    for f0 in (50, 100, 200, 400):
        # last time before the peak at which the frequency is below f0
        pre = finst[j, FS: peak[j]]             # skip the first second (start taper)
        i = np.where(pre < f0)[0][-1] + FS
        t = (peak[j] - i) / FS
        row[f"{f0}Hz_waveform_s"] = round(float(t), 4)
        row[f"{f0}Hz_newtonian_s"] = round(float(tau_of_f(f0, mc[j])), 4)
    row["window_start_freq_newtonian_Hz"] = round(float(f_of_tau(MERGE / FS, mc[j])), 2)
    tab.append(row)
checks["time_before_merger"] = tab
snr2 = (W ** 2).sum(-1)
checks["template_snr2_H1_over_L1_median"] = float(np.median(snr2[:, 0] / snr2[:, 1]))
np.savez(os.path.join(HERE, "templates_2048.npz"), white=W, white_cross=Wc, chirp_mass=mc,
         grid_log_mc=z["grid_log_mc"], merge_index=MERGE, sample_rate=FS, rows=rows, peak_in_file=peak)
with open(os.path.join(HERE, "templates_checks.json"), "w") as fh:
    json.dump(checks, fh, indent=1)
print(json.dumps(checks, indent=1))
