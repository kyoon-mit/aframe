"""Per-sample frequency of h+ - i hx just before and after the coalescence sample (no averaging)."""
import h5py, numpy as np
fn = '/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5'
lo, hi = 61440 - 400, 61440 + 100
res = []
with h5py.File(fn, 'r') as f:
    for a, b in [(0, 64), (100000, 100064), (199936, 200000)]:
        hp = f['waveforms/plus'][a:b, lo:hi]; hc = f['waveforms/cross'][a:b, lo:hi]
        h = hp - 1j * hc
        i = 61440 - lo
        d_before = np.abs(np.angle(h[:, i] * np.conj(h[:, i - 1]))) * 2048 / (2 * np.pi)   # step into 61440
        d_after = np.abs(np.angle(h[:, i + 1] * np.conj(h[:, i]))) * 2048 / (2 * np.pi)
        # frequency at the amplitude peak
        amp = np.abs(h); pk = amp.argmax(axis=1)
        d_pk = np.abs(np.angle(h[np.arange(b-a), pk] * np.conj(h[np.arange(b-a), pk - 1]))) * 2048 / (2 * np.pi)
        res.append((d_before, d_after, d_pk))
db = np.concatenate([r[0] for r in res]); da = np.concatenate([r[1] for r in res]); dp = np.concatenate([r[2] for r in res])
print(f"step into 61440: {db.min():.1f}..{db.max():.1f} Hz; step out of 61440: {da.min():.1f}..{da.max():.1f} Hz; at amplitude peak: {dp.min():.1f}..{dp.max():.1f} Hz")
# median of the eight per-sample frequencies from 61436 to 61444
meds = []
with h5py.File(fn, 'r') as f:
    for a, b in [(0, 64), (100000, 100064), (199936, 200000)]:
        hp = f['waveforms/plus'][a:b, lo:hi]; hc = f['waveforms/cross'][a:b, lo:hi]
        h = hp + 1j * hc
        i = 61440 - lo
        steps = np.abs(np.angle(h[:, i - 3:i + 5] * np.conj(h[:, i - 4:i + 4]))) * 2048 / (2 * np.pi)
        meds.append(np.median(steps, axis=1))
meds = np.concatenate(meds)
print(f"median of 8 steps around 61440: {meds.min():.1f}..{meds.max():.1f} Hz")
