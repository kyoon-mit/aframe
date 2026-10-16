"""Where is the amplitude peak of stored training waveforms, and what follows it."""
import sys
import h5py
import numpy as np

fn = sys.argv[1]
fs = 2048
with h5py.File(fn, "r") as f:
    print("chunks", f["waveforms/plus"].chunks, "compression", f["waveforms/plus"].compression)
    rows = [0, 1, 2, 3, 100000, 199999]
    for r in rows:
        hp = f["waveforms/plus"][r]
        hc = f["waveforms/cross"][r]
        m1 = f["parameters/mass_1"][r]; m2 = f["parameters/mass_2"][r]
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        amp = np.sqrt(hp**2 + hc**2)
        k = int(np.argmax(amp))
        n = len(hp)
        sig = n - 2 * fs
        # last nonzero-amplitude stretch after peak
        post = amp[k:]
        # first index after peak where amp falls below 1e-3 of peak
        below = np.where(post < 1e-3 * amp[k])[0]
        end_ring = k + (below[0] if len(below) else len(post))
        # content in the 0.5 s after merger beyond ringdown
        win_post = amp[sig + 1 : sig + 1024]
        tail = amp[end_ring : sig + 1024]
        print(f"row {r}: Mc_det {mc:.4f}, n {n}, peak idx {k} (sig idx {sig}, diff {k - sig} samples = {(k - sig) / fs * 1e3:.2f} ms), "
              f"amp drops below 1e-3 peak at idx {end_ring} ({(end_ring - sig) / fs * 1e3:.1f} ms after sig idx); "
              f"max amp in [end_ring, sig+1024) relative to peak: {tail.max() / amp[k] if len(tail) else 0:.3e}; "
              f"max amp last 2 s after ringdown: {amp[end_ring:].max() / amp[k]:.3e}; amp at idx 0: {amp[0] / amp[k]:.3e}")
        # where does nonzero wrapped content start/end
        nz = np.where(amp[end_ring:] > 1e-6 * amp[k])[0]
        if len(nz):
            print(f"    nonzero after ringdown: first at {(end_ring + nz[0] - sig) / fs * 1e3:.1f} ms after merger, last at {(end_ring + nz[-1] - sig) / fs * 1e3:.1f} ms")
