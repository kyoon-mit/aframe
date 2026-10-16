"""Amplitude peak and instantaneous frequency near the coalescence sample, 192 stored waveforms."""
import time, h5py, numpy as np
t0 = time.time()
fn = '/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5'
lo, hi = 61440 - 400, 61440 + 100
offs, freqs, mcs = [], [], []
with h5py.File(fn, 'r') as f:
    print('chunks', f['waveforms/plus'].chunks)
    m1 = f['parameters/mass_1']; m2 = f['parameters/mass_2']
    for a, b in [(0, 64), (100000, 100064), (199936, 200000)]:
        hp = f['waveforms/plus'][a:b, lo:hi]; hc = f['waveforms/cross'][a:b, lo:hi]
        M1 = m1[a:b]; M2 = m2[a:b]
        amp = np.hypot(hp, hc)
        pk = amp.argmax(axis=1) + lo
        offs.extend(61440 - pk)
        ph = np.unwrap(np.angle(hp - 1j * hc), axis=1)
        fr = np.diff(ph, axis=1) * 2048 / (2 * np.pi)
        i = 61440 - lo
        freqs.extend(np.abs(0.5 * (fr[:, i - 1] + fr[:, i])))
        mcs.extend((M1 * M2) ** 0.6 / (M1 + M2) ** 0.2)
offs = np.array(offs); freqs = np.array(freqs); mcs = np.array(mcs)
print(f"n {offs.size}; chirp mass {mcs.min():.3f}..{mcs.max():.3f}")
print(f"peak before 61440 by {offs.min()}..{offs.max()} samples = {offs.min()/2.048:.2f}..{offs.max()/2.048:.2f} ms")
print(f"instantaneous frequency at 61440: {freqs.min():.1f}..{freqs.max():.1f} Hz")
print(f"{time.time()-t0:.1f} s")
