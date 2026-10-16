"""Amplitude peak of stored training polarizations against the coalescence index 61440."""
import h5py
import numpy as np

fn = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
fs = 2048
coal = 65536 - 2 * fs
rng = np.random.default_rng(20261016)
with h5py.File(fn, "r") as f:
    n = f["waveforms/plus"].shape[0]
    starts = np.sort(rng.choice(n // 64, size=4, replace=False)) * 64
    offs, post_amp, m1s, m2s = [], [], [], []
    for s in starts:
        hp = f["waveforms/plus"][s:s + 64]
        hc = f["waveforms/cross"][s:s + 64]
        m1 = f["parameters/mass_1"][s:s + 64]
        m2 = f["parameters/mass_2"][s:s + 64]
        amp = np.sqrt(hp ** 2 + hc ** 2)
        pk = amp.argmax(-1)
        offs.append(pk - coal)
        peak = amp.max(-1)
        # largest amplitude from 0.1 s to 2 s after the coalescence, relative to the peak
        post_amp.append(amp[:, coal + int(0.1 * fs):].max(-1) / peak)
        m1s.append(m1); m2s.append(m2)
offs = np.concatenate(offs); post_amp = np.concatenate(post_amp)
m1 = np.concatenate(m1s); m2 = np.concatenate(m2s)
mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
print("rows", offs.size, "chirp mass range %.3f-%.3f" % (mc.min(), mc.max()))
print("peak index - 61440 (samples): min", offs.min(), "median", np.median(offs), "max", offs.max())
print("in ms: min %.2f median %.2f max %.2f" % (offs.min() / fs * 1e3, np.median(offs) / fs * 1e3, offs.max() / fs * 1e3))
print("max amplitude 0.1-2 s after coalescence / peak: median %.2e max %.2e" % (np.median(post_amp), post_amp.max()))
