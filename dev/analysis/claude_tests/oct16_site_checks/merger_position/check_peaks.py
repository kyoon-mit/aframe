"""Amplitude peak of stored training waveforms relative to their coalescence sample.

Reads three storage chunks (64 rows each) of the training file; amplitude is
sqrt(h_plus^2 + h_cross^2); instantaneous frequency from the phase of h_plus + i h_cross.
"""
import sys

import h5py
import numpy as np

fn = sys.argv[1]
fs = 2048
offs, fr_at, mcs = [], [], []
with h5py.File(fn, "r") as f:
    n = f["waveforms/plus"].shape[0]
    right_pad = float(f.attrs["right_pad"])
    for start in (0, n // 2, n - 64):
        hp = f["waveforms/plus"][start:start + 64]
        hc = f["waveforms/cross"][start:start + 64]
        m1 = f["parameters/mass_1"][start:start + 64]
        m2 = f["parameters/mass_2"][start:start + 64]
        sig = hp.shape[-1] - int(right_pad * fs)
        amp = np.hypot(hp, hc)
        k = amp.argmax(-1)
        offs.append(k - sig)
        ph = np.unwrap(np.arctan2(hc, hp), axis=-1)
        fr = np.abs(np.diff(ph, axis=-1)) * fs / (2 * np.pi)
        fr_at.append(np.median(fr[:, sig - 4:sig + 4], axis=-1))
        mcs.append((m1 * m2) ** 0.6 / (m1 + m2) ** 0.2)
offs = np.concatenate(offs)
fr_at = np.concatenate(fr_at)
mcs = np.concatenate(mcs)
print(f"{offs.size} waveforms, detector-frame chirp mass {mcs.min():.3f} to {mcs.max():.3f}")
print(f"coalescence sample {sig}; amplitude peak at {offs.min()} to {offs.max()} samples from it "
      f"({offs.min() / fs * 1e3:.2f} to {offs.max() / fs * 1e3:.2f} ms), median {np.median(offs):.0f}")
print(f"instantaneous frequency at the coalescence sample: {fr_at.min():.1f} to {fr_at.max():.1f} Hz (Nyquist {fs / 2} Hz)")
