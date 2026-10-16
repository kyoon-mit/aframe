"""Reproduce tone_check kernels and matrix-pencil numbers; save kernels, tracks, pencil poles."""

import json
import time

import numpy as np

import common as C

T0 = time.time()


def kernel_track(K, sign):
    """Instantaneous frequency (Hz, signed) and envelope of the dominant-sign part of K vs lag."""
    F = np.fft.fft(K)
    f = np.fft.fftfreq(C.L, C.DT)
    F[sign * f <= 0] = 0
    Ks = np.fft.ifft(F)
    ph = np.unwrap(np.angle(Ks))
    finst = np.gradient(ph) / (2 * np.pi * C.DT)
    return finst, np.abs(Ks)


def main():
    ref = json.load(open(C.TONE_DIR / "results.json"))
    ker = C.build_kernels()
    out = {"templates": {}, "match": {}, "check_vs_tone_check": {}, "pencil_raw_stats": {}}
    save = {}
    poles = {}
    for Mc in C.MCS:
        K, E, info = ker[Mc]["K"], ker[Mc]["E"], ker[Mc]["info"]
        r = ref["templates"][str(Mc)]
        diffs = {k: abs(float(info[k]) - float(r[k])) for k in r}
        out["templates"][str(Mc)] = info
        out["check_vs_tone_check"][str(Mc)] = {"template_max_abs_diff": max(diffs.values())}
        print(f"Mc={Mc} row={info['row']} sign={info['grid_sign']} "
              f"pos={info['frac_energy_pos_freq']:.4f} cycles={info['cycles_window_dominant_sign_part']:.1f} "
              f"template max diff vs tone_check {max(diffs.values()):.2e}", flush=True)
        save[f"K_{Mc}"] = K
        finst, env = kernel_track(K, info["grid_sign"])
        save[f"finst_{Mc}"] = finst
        save[f"env_{Mc}"] = env

        m = {"A_best_fourier": {}, "B_fixed_grid": {}, "C_pencil_reflected": {},
             "C_pencil_raw_growing": {}, "C_pencil_clipped": {}}
        cum = np.cumsum(np.sort(E)[::-1]) / E.sum()
        for N in C.NS:
            m["A_best_fourier"][N] = float(np.sqrt(cum[N - 1]))
            m["B_fixed_grid"][N] = C.lstsq_match(C.grid_logz(N, info["grid_sign"]), K)[0]
        t = time.time()
        shape, s, Vh = C.hankel_svd(K)
        print(f"  SVD {shape} {time.time() - t:.1f}s", flush=True)
        save[f"hankel_sv_{Mc}"] = s
        st = {}
        for N in C.NS:
            z = C.pencil_raw(Vh, N)
            zr, nref = C.reflect(z)
            zc, nclip = C.clip(z)
            m["C_pencil_reflected"][N] = C.lstsq_match(C.safe_log(zr), K)[0]
            m["C_pencil_raw_growing"][N] = C.lstsq_match(C.safe_log(z), K)[0]
            m["C_pencil_clipped"][N] = C.lstsq_match(C.safe_log(zc), K)[0]
            growth = C.L * np.log(np.abs(z))  # e-folds over the kernel length
            st[N] = {
                "n_growing": nref,
                "max_efolds_growth_over_L": float(growth.max()),
                "median_efolds_growth_of_growing": float(np.median(growth[growth > 0])) if nref else 0.0,
            }
            poles[f"z_{Mc}_{N}"] = z
        del Vh
        out["pencil_raw_stats"][str(Mc)] = st
        for k in m:
            out["match"].setdefault(k, {})[str(Mc)] = m[k]
        # comparison with tone_check
        cmp = {}
        for mine, theirs in (("A_best_fourier", "A_best_fourier"), ("B_fixed_grid", "B_fixed_grid"),
                             ("C_pencil_reflected", "C_matrix_pencil")):
            cmp[mine] = max(abs(m[mine][N] - ref["match"][theirs][str(Mc)][str(N)]) for N in C.NS)
        out["check_vs_tone_check"][str(Mc)].update({f"max_abs_diff_{k}": v for k, v in cmp.items()})
        print("  max |diff| vs tone_check:", {k: f"{v:.2e}" for k, v in cmp.items()}, flush=True)
        for k in m:
            print(f"  {k:22s}", " ".join(f"{m[k][N]:.4f}" for N in C.NS), flush=True)
        print("  growing poles:", {N: st[N]["n_growing"] for N in C.NS},
              "max e-folds:", {N: round(st[N]["max_efolds_growth_over_L"], 2) for N in C.NS}, flush=True)

    np.savez(C.OUT_DIR / "kernels.npz", **save)
    np.savez(C.OUT_DIR / "pencil_poles.npz", **poles)
    out["wall_s"] = time.time() - T0
    json.dump(out, open(C.OUT_DIR / "repro.json", "w"), indent=1)
    print(f"wall {time.time() - T0:.1f}s")


if __name__ == "__main__":
    main()
