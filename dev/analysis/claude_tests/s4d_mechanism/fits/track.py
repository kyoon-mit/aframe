"""Hold time tau(f) of each kernel: group delay Re[DFT(l K) / DFT(K)], smoothed with energy weights."""

import json

import numpy as np

import common as C

G_MSUN_S = 4.925490947641267e-6  # G Msun / c^3 in seconds
SMOOTH_BINS = 9  # energy-weighted boxcar over 9 DFT bins (2.3 Hz)
E_QUANT = (0.001, 0.999)  # band edges: 0.1% and 99.9% of the sign's energy


def newtonian_tau(f, mc_det):
    """Newtonian time to coalescence (s) at GW frequency f (Hz)."""
    return (5.0 / 256.0) * (mc_det * G_MSUN_S) ** (-5.0 / 3.0) * (np.pi * f) ** (-8.0 / 3.0)


def group_delay(K):
    """Per-bin frequency (Hz), energy and group delay (s) of K."""
    lags = np.arange(C.L)
    F = np.fft.fft(K)
    Fl = np.fft.fft(lags * K)
    E = np.abs(F) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        gd = np.real(Fl / F) * C.DT
    f = np.fft.fftfreq(C.L, C.DT)
    return f, E, gd


def sign_track(f, E, gd, sign):
    """Sorted |f|, energy, smoothed tau for one frequency sign."""
    sel = sign * f > 0
    af = sign * f[sel]
    order = np.argsort(af)
    af, e, g = af[order], E[sel][order], gd[sel][order]
    w = np.ones(SMOOTH_BINS)
    num = np.convolve(e * g, w, mode="same")
    den = np.convolve(e, w, mode="same")
    tau = num / np.maximum(den, 1e-300)
    tau = np.clip(tau, 2 * C.DT, C.L * C.DT)
    return af, e, tau


def pava_decreasing(y, w):
    """Weighted least-squares non-increasing fit (pool adjacent violators)."""
    vals, wts, lens = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); lens.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v2, w2, n2 = vals.pop(), wts.pop(), lens.pop()
            v1, w1, n1 = vals.pop(), wts.pop(), lens.pop()
            ww = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / ww if ww > 0 else 0.5 * (v1 + v2))
            wts.append(ww); lens.append(n1 + n2)
    return np.repeat(vals, lens)


def band_and_count(af, e, tau):
    """Energy band and the time-bandwidth count n(f) = int tau df over it."""
    cum = np.cumsum(e) / e.sum()
    lo = af[np.searchsorted(cum, E_QUANT[0])]
    hi = af[min(np.searchsorted(cum, E_QUANT[1]), af.size - 1)]
    inb = (af >= lo) & (af <= hi)
    df = af[1] - af[0]
    n = np.cumsum(tau * inb) * df
    return float(lo), float(hi), n, inb


def main():
    ker = np.load(C.OUT_DIR / "kernels.npz")
    rep = json.load(open(C.OUT_DIR / "repro.json"))
    out = {}
    save = {}
    for Mc in C.MCS:
        K = ker[f"K_{Mc}"]
        info = rep["templates"][str(Mc)]
        mc_det = info["chirp_mass"]
        f, E, gd = group_delay(K)
        res = {"energy_frac": {}}
        for sign in (1, -1):
            af, e, tau_s = sign_track(f, E, gd, sign)
            tau = np.exp(pava_decreasing(np.log(tau_s), e + 1e-30 * e.max()))
            save[f"tau_smooth_{Mc}_{'pos' if sign > 0 else 'neg'}"] = tau_s
            lo, hi, n, inb = band_and_count(af, e, tau)
            key = "pos" if sign > 0 else "neg"
            res["energy_frac"][key] = float(e.sum() / E.sum())
            save[f"af_{Mc}_{key}"] = af
            save[f"e_{Mc}_{key}"] = e
            save[f"tau_{Mc}_{key}"] = tau
            save[f"n_{Mc}_{key}"] = n
            tn = newtonian_tau(af, mc_det)
            chk = {}
            for fq in (80, 100, 150, 200, 300, 400, 500, 600, 700):
                i = np.searchsorted(af, fq)
                chk[fq] = [round(float(tau[i]), 4), round(float(tn[i]), 4)]
            # Newtonian time-bandwidth count over the same band, and cycles in the window
            nt = float(np.sum(np.clip(tn, 0, C.L * C.DT) * inb) * (af[1] - af[0]))
            res[key] = {
                "band_hz": [lo, hi],
                "int_tau_df": float(n[-1]),
                "two_pi_int_tau_df": float(2 * np.pi * n[-1]),
                "int_tau_df_newtonian_same_band_capped_at_T": nt,
                "tau_measured_vs_newtonian_s_at_hz": chk,
            }
        out[str(Mc)] = res
        dom = "pos" if info["grid_sign"] > 0 else "neg"
        r = res[dom]
        print(f"Mc={Mc} dominant={dom} energy {res['energy_frac']} band {r['band_hz'][0]:.1f}-{r['band_hz'][1]:.1f} Hz "
              f"int tau df {r['int_tau_df']:.1f} (2pi x = {r['two_pi_int_tau_df']:.0f}; Newtonian {r['int_tau_df_newtonian_same_band_capped_at_T']:.1f}); "
              f"cycles {info['cycles_window_dominant_sign_part']:.1f}, 3/8 cycles = {0.375 * info['cycles_window_dominant_sign_part']:.1f}")
        print("   tau measured vs Newtonian (s):", r["tau_measured_vs_newtonian_s_at_hz"])
    np.savez(C.OUT_DIR / "tracks.npz", **save)
    json.dump(out, open(C.OUT_DIR / "tracks.json", "w"), indent=1)


if __name__ == "__main__":
    main()
