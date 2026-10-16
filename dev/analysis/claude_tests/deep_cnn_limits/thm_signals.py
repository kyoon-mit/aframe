"""Unit-norm test signals in a 4 s window at 2048 Hz (N = 8192), for white
unit-variance noise.

newtonian(mc, tc, f_cut): leading-order chirp, phase -2 (tau / (5 G Mc / c^3))^(5/8),
    amplitude proportional to tau^(-1/4) (frequency^(2/3)), tau = tc - t; it starts
    abruptly at t = 0 (as a window cut from a longer stream does) and is
    switched off by a 5 ms half-cosine just before the frequency reaches f_cut.
whitened_bns(m1, m2, end_after_peak): IMRPhenomD plus polarization from 20 Hz,
    whitened over its whole length with the aLIGO O3-low design PSD
    (LALSimulation SimNoisePSDaLIGOaLIGOO3LowT1800545, LIGO-T1800545), band
    20 Hz to 1024 Hz, then cut to the 4 s window that ends end_after_peak seconds
    after the peak of the whitened amplitude. A stand-in for our whitened data.
"""
import numpy as np

FS = 2048.0
N = 8192
TSUN = 4.925490947641267e-06     # G Msun / c^3 in s (lal.MTSUN_SI)


def newtonian(mc=1.2, tc=3.75, f_cut=800.0, taper=0.005):
    t = np.arange(N) / FS
    tau = tc - t
    gm = mc * TSUN
    on = tau > 0
    f = np.zeros(N)
    f[on] = (5.0 / (256.0 * tau[on])) ** 0.375 * gm ** -0.625 / np.pi
    tau_cut = 5.0 / 256.0 * (np.pi * f_cut) ** (-8.0 / 3.0) * gm ** (-5.0 / 3.0)
    phase = np.zeros(N)
    phase[on] = -2.0 * (tau[on] / (5.0 * gm)) ** 0.625
    amp = np.zeros(N)
    amp[on] = tau[on] ** -0.25
    w = np.clip((tau - tau_cut) / taper, 0.0, 1.0)
    w = 0.5 - 0.5 * np.cos(np.pi * w)
    s = amp * w * np.cos(phase)
    s[tau <= tau_cut] = 0.0
    return s / np.linalg.norm(s), dict(f_start=f[0], tau_cut=tau_cut)


def aligo_o3low_psd(freqs):
    """aLIGO O3-low PSD (LIGO-T1800545) on a uniform grid starting at 0 Hz;
    infinite below 20 Hz."""
    import lal
    import lalsimulation as ls
    df = freqs[1] - freqs[0]
    ser = lal.CreateREAL8FrequencySeries("psd", lal.LIGOTimeGPS(0), 0.0, df, lal.DimensionlessUnit, len(freqs))
    ls.SimNoisePSDaLIGOaLIGOO3LowT1800545(ser, 20.0)
    out = ser.data.data.copy()
    out[(freqs < 20.0) | (out <= 0)] = np.inf
    return out


def whitened_bns(m1=1.4, m2=1.4, end_after_peak=0.25, f_min=20.0):
    import lal
    import lalsimulation as ls
    dt = 1.0 / FS
    hp, hc = ls.SimInspiralChooseTDWaveform(
        m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, 0, 0, 0, 0,
        1e6 * lal.PC_SI * 100, 0.0, 0.0, 0.0, 0.0, 0.0, dt, f_min, f_min,
        lal.CreateDict(), ls.GetApproximantFromString("IMRPhenomD"))
    h = hp.data.data.copy()
    pad = 8 * N
    L = len(h) + 2 * pad
    L = int(2 ** np.ceil(np.log2(L)))
    x = np.zeros(L)
    x[pad:pad + len(h)] = h
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(L, dt)
    psd = aligo_o3low_psd(freqs)
    hw = np.fft.irfft(X / np.sqrt(psd), n=L)
    env = np.abs(hw)
    ipk = int(np.argmax(env))
    end = ipk + int(round(end_after_peak * FS))
    s = hw[end - N:end].copy()
    return s / np.linalg.norm(s), dict(peak_index=N - int(round(end_after_peak * FS)))


def mc_of(m1, m2):
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


if __name__ == "__main__":
    s, info = newtonian()
    print("newtonian: f_start %.2f Hz, tau_cut %.4f s, max|s| at SNR 8: %.3f, sum s: %.3e" % (
        info["f_start"], info["tau_cut"], 8 * np.abs(s).max(), s.sum()))
    sw, info2 = whitened_bns()
    print("whitened IMRPhenomD 1.4+1.4: max|s| at SNR 8: %.3f, sum s: %.3e, peak index %d" % (
        8 * np.abs(sw).max(), sw.sum(), info2["peak_index"]))
    np.save("out/sig_newtonian.npy", s)
    np.save("out/sig_whitened_bns.npy", sw)
