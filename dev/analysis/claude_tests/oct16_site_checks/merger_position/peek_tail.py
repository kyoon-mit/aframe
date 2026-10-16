"""Envelope and frequency of stored waveforms around and after the coalescence index."""
import sys
import h5py
import numpy as np

fn = sys.argv[1]
fs = 2048
with h5py.File(fn, "r") as f:
    for r in [0, 100000, 199999]:
        hp = f["waveforms/plus"][r]
        hc = f["waveforms/cross"][r]
        n = len(hp)
        sig = n - 2 * fs
        amp = np.sqrt(hp**2 + hc**2)
        pk = amp.max()
        ph = np.unwrap(np.arctan2(hc, hp))
        fr = np.abs(np.diff(ph)) * fs / (2 * np.pi)
        print(f"row {r}: peak at {np.argmax(amp) - sig} samples")
        for ms in [-50, -20, -10, -5, -2, 0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 500, 1000, 1500, 1990]:
            i = sig + int(round(ms * fs / 1000))
            seg = slice(max(i - 4, 0), i + 5)
            print(f"   t {ms:6d} ms: amp/peak {amp[i] / pk:.3e}, inst freq {np.median(fr[seg]):8.1f} Hz")
