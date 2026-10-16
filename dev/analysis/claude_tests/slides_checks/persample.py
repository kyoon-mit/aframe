import h5py, numpy as np
f = h5py.File('/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/s4d_energy_diag/train_windows_snr8_powerlaw.h5', 'r')
mc = f['params/chirp_mass'][:]
print('chirp mass: n', mc.size, 'nan', np.isnan(mc).sum(), 'min %.3f max %.3f median %.3f' % (np.nanmin(mc), np.nanmax(mc), np.nanmedian(mc)))
snr = f['params/snr'][:]
print('snr: nan', np.isnan(snr).sum(), 'min %.2f median %.2f' % (np.nanmin(snr), np.nanmedian(snr)))
ok = ~np.isnan(snr[:440]); idx = np.where(ok)[0]
Xc = f["X_clean"][:440][ok]
s = snr[:440][ok]; print("rows", ok.sum())
tot = np.sqrt((Xc.astype(np.float64) ** 2).sum(axis=(1, 2)))
print('sqrt(sum X_clean^2)/snr: median %.4f (5-95%%: %.4f-%.4f)' % (np.median(tot / s), *np.percentile(tot / s, [5, 95])))
# per-sample amplitude scaled to network SNR 8
scale = 8.0 / tot
Y = Xc * scale[:, None, None]
nz = np.abs(Y) > 0
rms_all = np.sqrt((Y ** 2).mean(axis=(1, 2)))
peak = np.abs(Y).max(axis=(1, 2))
# merger position: argmax of |X_clean| summed over detectors
pos = np.argmax((Xc ** 2).sum(axis=1), axis=1) / 2048.0
print('at network SNR 8: per-sample rms over both detectors x 8192: median %.4f; max |sample| median %.3f (5-95%%: %.3f-%.3f)' % (np.median(rms_all), np.median(peak), *np.percentile(peak, [5, 95])))
print('peak-power time in window [s]: min %.3f median %.3f max %.3f' % (pos.min(), np.median(pos), pos.max()))
# rms per sample over first 3 s (inspiral body)
rms3 = np.sqrt((Y[:, :, :6144] ** 2).mean(axis=(1, 2)))
print('rms per sample over first 3 s: median %.4f' % np.median(rms3))
