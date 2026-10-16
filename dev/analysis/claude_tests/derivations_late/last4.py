import h5py, numpy as np
W = "/home/kyoon/SSM-BNS/DATA/aframe/train/end_o3_ratesandpops_bns_uniform_chirp.hdf5"
psd = np.load("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz")["psd"].mean(0)
f = h5py.File(W); p = f["parameters"]
m1, m2 = p["mass_1"][:2000], p["mass_2"][:2000]
mc = (m1*m2)**0.6/(m1+m2)**0.2
fs = 2048
def snr2(seg):
    n = len(seg); win = np.ones(n); r = int(0.1 * fs); win[:r] = 0.5*(1-np.cos(np.pi*np.arange(r)/r))
    H = np.fft.rfft(seg * win) / fs; fr = np.fft.rfftfreq(n, 1/fs)
    S = np.interp(fr, np.arange(2049) * 0.5, psd); ok = (fr >= 20) & (fr < 1000)
    return 4 * np.sum(np.abs(H[ok])**2 / S[ok]) * (fs / n)
for target in (1.0, 1.2, 1.5, 2.0, 2.5):
    i = int(np.argmin(np.abs(mc - target)))
    h = f["waveforms/plus"][i]
    peak = int(np.argmax(f["waveforms/plus"][i]**2 + f["waveforms/cross"][i]**2))
    full = snr2(h[: peak + 512]); last4 = snr2(h[peak + 512 - 8192: peak + 512])
    # where does the waveform start (first nonzero sample)
    nz = np.flatnonzero(np.abs(h) > 0)[0]
    print(f"Mc {mc[i]:.3f}: starts {(peak-nz)/fs:.1f} s before peak; SNR ratio last4/all = {np.sqrt(last4/full):.3f}; SNR^2 ratio {last4/full:.3f}; volume factor (1/r)^3 = {(full/last4)**1.5:.2f}")
