"""Tuning width: match of the 1.2 Msun chirplet fits and of the exact 1.2 Msun kernel
against kernels of the same source at chirp mass 1.2009 (1 + delta).

Match at lag 0 (mergers aligned) and maximized over lag (|cross-correlation|,
both kernels zero outside 0..L-1). Kernels unit norm, so the match is the
fraction of the matched filter's SNR kept.
"""

import json

import numpy as np

import collect as CL
import core as C

L = C.L
MC = 1.2
LEVELS = (0.9, 0.5)


def corr_max(A, B):
    n = 2 * L
    c = np.fft.ifft(np.conj(np.fft.fft(A, n)) * np.fft.fft(B, n))
    j = int(np.argmax(np.abs(c)))
    lag = j if j < L else j - n
    return float(np.abs(c[j])), int(lag)


def crossing(delta, rel, level):
    """Smallest |delta| on each side where rel falls below level (linear interpolation)."""
    out = {}
    for side, sel in (("plus", delta >= 0), ("minus", delta <= 0)):
        d, r = np.abs(delta[sel]), rel[sel]
        o = np.argsort(d)
        d, r = d[o], r[o]
        below = np.flatnonzero(r < level)
        if below.size == 0:
            out[side] = None
            continue
        i = int(below[0])
        if i == 0:
            out[side] = 0.0
            continue
        d0, d1, r0, r1 = d[i - 1], d[i], r[i - 1], r[i]
        out[side] = float(d0 + (r0 - level) * (d1 - d0) / (r0 - r1))
    return out


def curve(Kref, fam, deltas):
    lag0 = np.array([float(np.abs(np.vdot(Kref, K))) for K in fam])
    mx = [corr_max(Kref, K) for K in fam]
    return lag0, np.array([m for m, _ in mx]), np.array([g for _, g in mx])


def summarize(name, deltas, lag0, mxm, lags, peak_ref):
    i0 = int(np.flatnonzero(deltas == 0.0)[0])
    res = {"name": name, "match_at_delta0": float(mxm[i0]),
           "match_lag0": lag0.tolist(), "match_max_over_lag": mxm.tolist(), "best_lag_samples": lags.tolist(),
           "delta_of_peak_max_over_lag": float(deltas[int(np.argmax(mxm))])}
    for lab, arr in (("max_over_lag", mxm), ("lag0", lag0)):
        rel = arr / arr[i0]
        res[f"half_width_{lab}"] = {f"{lv:.1f}": crossing(deltas, rel, lv) for lv in LEVELS}
    for d in (-0.01, -0.005, -0.001, 0.001, 0.005, 0.01):
        j = int(np.argmin(np.abs(deltas - d)))
        res[f"match_max_over_lag_at_{d:+.3f}"] = float(mxm[j])
        res[f"relative_to_delta0_at_{d:+.3f}"] = float(mxm[j] / mxm[i0])
        res[f"relative_to_exact_curve_at_{d:+.3f}"] = float(mxm[j] / peak_ref[j]) if peak_ref is not None else None
    return res


def main():
    z = np.load(C.HERE / "family.npz")
    fam, deltas = z["K"], z["delta"]
    Kex = C.load_kernels()[MC]
    lag0, mxm, lags = curve(Kex, fam, deltas)
    out = {"deltas": deltas.tolist(), "chirp_mass_center": float(json.load(open(C.HERE / "family.json"))["chirp_mass_row"])}
    out["exact"] = summarize("exact 1.2 kernel", deltas, lag0, mxm, lags, None)
    recs = CL.load_runs()
    best = CL.best_by(recs, filt=lambda r: r["Mc"] == MC)
    out["chirplet"] = {}
    for (Mc, M), r in sorted(best.items()):
        Kf = CL.fitted_kernel(r)
        l0, mm, lg = curve(Kf, fam, deltas)
        s = summarize(f"chirplet M={M} ({r['start']})", deltas, l0, mm, lg, mxm)
        s["fit_match_to_stored_kernel"] = r["match_final"]
        out["chirplet"][str(M)] = s
        hw = s["half_width_max_over_lag"]
        print(f"M={M:4d} fit {r['match_final']:.4f} at delta0 {s['match_at_delta0']:.4f} "
              f"rel-0.9 width +{hw['0.9']['plus']} -{hw['0.9']['minus']}  rel-0.5 width +{hw['0.5']['plus']} -{hw['0.5']['minus']} "
              f"peak at {s['delta_of_peak_max_over_lag']:+.4f}", flush=True)
    hw = out["exact"]["half_width_max_over_lag"]
    print(f"exact rel-0.9 width +{hw['0.9']['plus']} -{hw['0.9']['minus']}  rel-0.5 width +{hw['0.5']['plus']} -{hw['0.5']['minus']}")
    json.dump(out, open(C.HERE / "tuning.json", "w"), indent=1)


if __name__ == "__main__":
    main()
