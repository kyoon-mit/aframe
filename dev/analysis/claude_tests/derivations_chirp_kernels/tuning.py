"""Half-maximum widths of the tuning curves (same code as the figure) and the 1%-off peak shift."""
import numpy as np

FS, L, TS = 2048, 8192, 4.925491e-6


def chirp(mc, quad=False):
    tau = (L - np.arange(L)) / FS
    f = (1 / (8 * np.pi)) * (5 / tau) ** (3 / 8) * (TS * mc) ** (-5 / 8)
    phi = -2 * (tau / (5 * TS * mc)) ** (5 / 8)
    a = tau ** -0.25 * (f < 900) * (f > 20)
    h = a * (np.sin(phi) if quad else np.cos(phi))
    return h / np.linalg.norm(h)


def energy(sig, mck):
    out = 0
    for k in (chirp(mck), chirp(mck, True)):
        y = np.fft.irfft(np.fft.rfft(sig, 2 * L) * np.fft.rfft(k[::-1], 2 * L), 2 * L)[:L]
        out = out + y ** 2
    return out


def peak(sig, mck):
    return np.sqrt(energy(sig, mck).max())


for mk in (1.15, 1.20, 1.25):
    ev = mk * np.exp(np.linspace(-0.03, 0.01, 4001))
    r = np.array([peak(chirp(m), mk) for m in ev])
    i0 = np.argmax(r)
    lo = ev[:i0][r[:i0] < 0.5].max()
    hi = ev[i0:][r[i0:] < 0.5].min()
    print(f"kernel {mk}: peak {r[i0]:.4f} at {100*(ev[i0]/mk-1):+.3f}%; below 0.5 at {100*(lo/mk-1):+.3f}% and {100*(hi/mk-1):+.3f}%;"
          f" response at -3%: {r[0]:.3f}; max beyond +0.5%: {r[ev > mk*1.005].max():.3f}")

# 1%-off kernel on a noise-free signal merging at the last sample
s = chirp(1.2)
for mk in (1.2, 1.2 * 1.01):
    e = energy(s, mk)
    i = int(np.argmax(e))
    print(f"kernel {mk:.4f}: peak energy {e[i]:.4f} (unit signal) at sample {i} = {(i-(L-1))/FS*1e3:+.1f} ms from merger")

# match_vs_mc range: +-3% around 1.2
mcs = 1.2 * np.exp(np.linspace(-0.03, 0.03, 241))
m = np.array([peak(s, x) for x in mcs])
print("match at -3%", m[0].round(3), "+3%", m[-1].round(3))
