"""What sits after the coalescence in stored polarizations: amplitude by time, and its frequency."""
import h5py
import numpy as np

fn = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
fs = 2048
coal = 65536 - 2 * fs
with h5py.File(fn, "r") as f:
    hp = f["waveforms/plus"][1000:1008]
    hc = f["waveforms/cross"][1000:1008]
for r in range(4):
    a = np.sqrt(hp[r] ** 2 + hc[r] ** 2)
    pk = a.max()
    seg = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.1), (0.1, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0)]
    out = []
    for t0, t1 in seg:
        x = a[coal + int(t0 * fs): coal + int(t1 * fs)]
        out.append(f"{t0}-{t1}s:{x.max() / pk:.1e}")
    # frequency of hp in 1.0-2.0 s after coalescence from zero crossings
    x = hp[r, coal + fs: coal + 2 * fs]
    zc = np.sum(np.diff(np.sign(x)) != 0) / 2.0
    # first sample of the stored array, and where the array starts being nonzero
    nz = np.nonzero(a)[0]
    print(r, " ".join(out), f"| zero-cross freq 1-2 s after: {zc:.1f} Hz | first nonzero index {nz[0]} (t={(nz[0]-coal)/fs:.2f} s), a[0]/pk={a[0]/pk:.1e}")
