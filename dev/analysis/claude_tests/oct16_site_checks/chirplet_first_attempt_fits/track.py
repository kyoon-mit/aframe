"""Time-frequency track of each kernel, quarter-cycle times and chirplet-count estimates.

Per kernel: analytic part of each frequency sign (zero-padded to 2L), raw unwrapped
phase, and f(l), df/dt(l) from local weighted quadratic fits of that phase.
Quarter-cycle time at l0: length of the interval around l0 where the true phase
stays within pi/2 of the tangent model (straight-line frequency f + fdot (t - t0),
i.e. a chirplet; or a constant frequency f, i.e. a tone).
"""

import json

import numpy as np

import core as C

FS, L = C.FS, C.L
G_MSUN_S = 4.925490947641267e-6
GRID_STEP = 2
QUARTER = np.pi / 2


def analytic_part(K, sign):
    n = 2 * L
    F = np.fft.fft(K, n)
    f = np.fft.fftfreq(n, 1 / FS)
    return np.fft.ifft(F * (sign * f > 0))[:L]


def half_width(l0):
    """Fit window half-width in samples: 8 + 10% of the lag."""
    return np.maximum(8, (0.1 * l0 + 8)).astype(int)


def local_quadratic(ph, w2, l0):
    """Weighted LS fit ph(l) ~ b0 + b1 x + b2 x^2, x = l - l0, triangular window x weights."""
    h = int(half_width(np.array([l0]))[0])
    lo, hi = max(0, l0 - h), min(L, l0 + h + 1)
    x = np.arange(lo, hi) - l0
    w = w2[lo:hi] * (1.0 - np.abs(x) / (h + 1))
    X = np.stack([np.ones_like(x, dtype=float), x, x.astype(float) ** 2], 1)
    A = X.T @ (w[:, None] * X)
    b = X.T @ (w * ph[lo:hi])
    return np.linalg.solve(A, b)


def smooth_track(Kp):
    """f (Hz, signed), fdot (Hz/s), phase constant on the grid, and raw phase/amplitude."""
    ph = np.unwrap(np.angle(Kp))
    A2 = np.abs(Kp) ** 2
    grid = np.arange(0, L, GRID_STEP)
    coef = np.array([local_quadratic(ph, A2, int(l0)) for l0 in grid])
    f = coef[:, 1] * FS / (2 * np.pi)
    fdot = coef[:, 2] * FS**2 / np.pi
    return grid, coef[:, 0], f, fdot, ph, A2


def quarter_cycle(ph, A2, l0, phi0, f0, c0):
    """(left, right) distances in samples until |true - model phase| >= pi/2; model chirp f0 + c0 t."""
    x = np.arange(L) - l0
    model = phi0 + 2 * np.pi * f0 * x / FS + np.pi * c0 * (x / FS) ** 2
    e = np.abs(ph - model)
    ok = A2 > 1e-3 * np.median(A2)
    bad = (e >= QUARTER) & ok
    right = np.flatnonzero(bad[l0:])
    left = np.flatnonzero(bad[: l0 + 1][::-1])
    dr = int(right[0]) if right.size else L - 1 - l0
    dl = int(left[0]) if left.size else l0
    return dl, dr


def newtonian(mc_det, tau):
    """Leading-order f, fdot, fddot at time tau before merger."""
    a = (G_MSUN_S * mc_det) ** (-5.0 / 8.0) * (5.0 / 256.0) ** (3.0 / 8.0) / np.pi
    f = a * tau ** (-3.0 / 8.0)
    fdot = (3.0 / 8.0) * f / tau
    fddot = (3.0 / 8.0) * (11.0 / 8.0) * f / tau**2
    return f, fdot, fddot


def segments(n_of_l, A2_sign):
    """Energy of each unit of the tiling count n(l) for one sign."""
    nseg = int(np.ceil(n_of_l[-1]))
    idx = np.minimum(np.floor(n_of_l).astype(int), nseg - 1)
    return np.bincount(idx, weights=A2_sign, minlength=nseg)


def count_for_energy(seg_e, total, levels):
    """Fewest segments (largest first) whose energy reaches level^2 of the total."""
    srt = np.sort(seg_e)[::-1]
    cum = np.cumsum(srt) / total
    out = {}
    for m in levels:
        k = int(np.searchsorted(cum, m**2) + 1)
        out[f"{m:.2f}"] = k if k <= srt.size else None
    return out


def main():
    ker = C.load_kernels()
    tmpl = json.load(open(C.S4D / "repro.json"))["templates"]
    out, save = {}, {}
    levels = [0.9, 0.95, 0.99]
    for Mc in C.MCS:
        K = ker[Mc]
        info = tmpl[str(Mc)]
        dom = int(info["grid_sign"])
        res = {"dominant_sign": dom}
        parts = {s: analytic_part(K, s) for s in (1, -1)}
        energy = {s: float(np.sum(np.abs(parts[s]) ** 2)) for s in (1, -1)}
        res["energy_by_sign"] = {"pos": energy[1], "neg": energy[-1]}
        grid, phi0, f, fdot, ph, A2 = smooth_track(parts[dom])
        # check: the other sign's track mirrors the dominant one (same orbital phase)
        g2, _, f_o, _, _, A2_o = smooth_track(parts[-dom])
        strong = A2[grid] > 0.2 * np.median(A2)
        res["other_sign_track_check"] = {
            "median_abs_rel_diff_f_where_dominant_strong": float(np.median(np.abs(f_o[strong] + f[strong]) / np.abs(f[strong]))),
        }
        Tq = np.zeros(grid.size)
        Tt = np.zeros(grid.size)
        dlr = np.zeros((grid.size, 2), dtype=int)
        for j, l0 in enumerate(grid):
            a, b = quarter_cycle(ph, A2, int(l0), phi0[j], f[j], fdot[j])
            dlr[j] = (a, b)
            Tq[j] = a + b
            a2, b2 = quarter_cycle(ph, A2, int(l0), phi0[j], f[j], 0.0)
            Tt[j] = a2 + b2
        lag = np.arange(L)
        Tq_l = np.interp(lag, grid, np.maximum(Tq, 1.0))
        Tt_l = np.interp(lag, grid, np.maximum(Tt, 1.0))
        n_q = np.cumsum(1.0 / Tq_l)
        n_t = np.cumsum(1.0 / Tt_l)
        res["tiling_count_chirplet_one_sign"] = float(n_q[-1])
        res["tiling_count_tone_one_sign"] = float(n_t[-1])
        # energy-ranked segment counts, both signs share the track (counts double for two signs)
        tot = energy[1] + energy[-1]
        segq = np.concatenate([segments(n_q, np.abs(parts[s]) ** 2) for s in (1, -1)])
        segt = np.concatenate([segments(n_t, np.abs(parts[s]) ** 2) for s in (1, -1)])
        res["segments_needed_chirplet"] = count_for_energy(segq, tot, levels)
        res["segments_needed_tone"] = count_for_energy(segt, tot, levels)
        res["segments_needed_chirplet_dominant_sign_only"] = count_for_energy(
            segments(n_q, np.abs(parts[dom]) ** 2), tot, levels)
        # Newtonian cross-check over the window, tau from 1/FS to L/FS
        tau = (np.arange(L) + 0.5) / FS
        _, fd_n, fdd_n = newtonian(info["chirp_mass"], tau)
        Tq_n = 2 * (3.0 / (2.0 * fdd_n)) ** (1.0 / 3.0) * FS  # samples, Taylor cubic term
        Tt_n = np.sqrt(2.0 / fd_n) * FS
        res["newtonian_tiling_count_chirplet"] = float(np.sum(1.0 / Tq_n))
        res["newtonian_tiling_count_tone"] = float(np.sum(1.0 / Tt_n))
        # summary of the track at a few lags
        res["track_at_lag"] = {}
        for l0 in (16, 64, 256, 1024, 2048, 4096, 6144):
            j = int(np.argmin(np.abs(grid - l0)))
            fn, fdn, _ = newtonian(info["chirp_mass"], (l0 + 0.5) / FS)
            res["track_at_lag"][str(l0)] = {
                "f_hz": float(f[j]), "fdot_hz_per_s": float(fdot[j]),
                "f_newtonian": float(dom * fn), "fdot_newtonian": float(-dom * fdn),
                "quarter_cycle_chirplet_samples": float(Tq[j]), "quarter_cycle_tone_samples": float(Tt[j]),
                "newtonian_quarter_cycle_chirplet_samples": float(Tq_n[l0]),
                "newtonian_quarter_cycle_tone_samples": float(Tt_n[l0]),
            }
        out[str(Mc)] = res
        for nm, v in (("grid", grid), ("phi0", phi0), ("f", f), ("fdot", fdot), ("Tq", Tq), ("Tt", Tt),
                      ("dl", dlr[:, 0]), ("dr", dlr[:, 1]), ("ph", ph), ("A2_dom", A2),
                      ("A2_pos", np.abs(parts[1]) ** 2), ("A2_neg", np.abs(parts[-1]) ** 2),
                      ("n_q", n_q), ("n_t", n_t)):
            save[f"{nm}_{Mc}"] = v
        print(f"Mc={Mc} dom={dom:+d} E={res['energy_by_sign']} tiling chirplet {n_q[-1]:.1f} tone {n_t[-1]:.1f} "
              f"(Newtonian {res['newtonian_tiling_count_chirplet']:.1f}, {res['newtonian_tiling_count_tone']:.1f}) "
              f"segments chirplet {res['segments_needed_chirplet']} tone {res['segments_needed_tone']} "
              f"other-sign f check {res['other_sign_track_check']}", flush=True)
        for l0, v in res["track_at_lag"].items():
            print(f"   l={l0}: f {v['f_hz']:.1f} (N {v['f_newtonian']:.1f}) fdot {v['fdot_hz_per_s']:.1f} (N {v['fdot_newtonian']:.1f}) "
                  f"Tq {v['quarter_cycle_chirplet_samples']:.0f} (N {v['newtonian_quarter_cycle_chirplet_samples']:.0f}) "
                  f"Tt {v['quarter_cycle_tone_samples']:.0f} (N {v['newtonian_quarter_cycle_tone_samples']:.0f})")
    np.savez(C.HERE / "track.npz", **save)
    json.dump(out, open(C.HERE / "track.json", "w"), indent=1)


if __name__ == "__main__":
    main()
