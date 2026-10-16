"""Stored-waveform amplitude around and after the coalescence sample, and before it (start)."""
import h5py, numpy as np
fn = '/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5'
with h5py.File(fn, 'r') as f:
    rows = [0, 1, 2, 100000, 199999]
    m1 = f['parameters/mass_1'][:]; m2 = f['parameters/mass_2'][:]; z = f['parameters/redshift'][:]
    for r in rows:
        hp = f['waveforms/plus'][r]; hc = f['waveforms/cross'][r]
        a = np.hypot(hp, hc)
        mc = (m1[r]*m2[r])**0.6/(m1[r]+m2[r])**0.2
        pk = int(a.argmax())
        print(f"row {r}: chirp mass (stored masses) {mc:.4f}, z {z[r]:.4f}, peak index {pk} ({pk-61440:+d} vs 61440), peak amp {a[pk]:.3e}")
        for lo, hi in [(0, 2048), (2048, 4096), (61440+41, 61440+205), (61440+205, 61440+1024), (61440+1024, 61440+2048), (61440+2048, 65536)]:
            seg = a[lo:hi]
            # dominant frequency in that segment from zero crossings of hp
            s = hp[lo:hi]
            zc = np.count_nonzero(np.diff(np.signbit(s)))
            f_est = zc / 2 / ((hi - lo) / 2048)
            print(f"   samples {lo}-{hi}: max amp / peak {seg.max()/a[pk]:.4f}, median {np.median(seg)/a[pk]:.2e}, zero-crossing freq {f_est:.1f} Hz")
