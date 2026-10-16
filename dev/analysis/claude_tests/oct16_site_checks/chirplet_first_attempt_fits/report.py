"""Collect chirplet fits, compare with resonator and Fourier counts, write results.json and figures."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import collect as CL  # noqa: E402
import core as C  # noqa: E402

MS = [1, 2, 4, 8, 16, 32, 64, 128]
TH = [0.9, 0.95, 0.99]
FREE_KERNEL_COMPLEX = 4016  # positive-frequency bins 20 Hz <= f < 1024 Hz of an 8192-sample spectrum
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def needed(curve, levels, grid):
    out = {}
    for lv in levels:
        hit = [n for n in grid if curve.get(n) is not None and curve[n] >= lv]
        if hit:
            n = min(hit)
            prev = [m for m in grid if m < n]
            out[f"{lv:.2f}"] = {"smallest_tested": n, "previous_tested": max(prev) if prev else None}
        else:
            out[f"{lv:.2f}"] = {"smallest_tested": None, "previous_tested": max(grid)}
    return out


def resonator_curves():
    s = json.load(open(C.S4D / "summary.json"))["tables"]
    out = {}
    for Mc in C.MCS:
        rows = s[str(Mc)]["rows"]
        ps = sorted(int(p) for p in rows)
        raw = {p: rows[str(p)].get("best_float32_valid_decaying") for p in ps}
        env, run = {}, 0.0
        for p in ps:
            if raw[p] is not None:
                run = max(run, raw[p])
            env[p] = run
        out[Mc] = {"tested_p": ps, "best_f32_valid": raw, "running_max": env,
                   "thresholds_from_file": s[str(Mc)]["thresholds_best_f32_valid"],
                   "int_tau_df_dominant": s[str(Mc)]["int_tau_df_dominant"],
                   "int_tau_df_both_signs": s[str(Mc)]["int_tau_df_both_signs"]}
    return out


def fourier_curves(ker):
    out = {}
    for Mc in C.MCS:
        E = np.abs(np.fft.fft(ker[Mc])) ** 2
        cum = np.sqrt(np.cumsum(np.sort(E)[::-1]) / E.sum())
        grid = [2**k for k in range(13)]
        out[Mc] = {"match_at_N": {n: float(cum[n - 1]) for n in grid},
                   "N_needed": {f"{lv:.2f}": int(np.searchsorted(cum, lv) + 1) for lv in TH}}
    return out


def main():
    ker = C.load_kernels()
    tj = json.load(open(C.HERE / "track.json"))
    recs = CL.load_runs()
    res_c = resonator_curves()
    four = fourier_curves(ker)
    out = {"config": {
        "kernels": str(C.KERNELS), "L": C.L, "fs": C.FS,
        "model": "K(l) = sum_k a_k exp(-(l-t_k)^2/(2 s_k^2)) exp(i(2 pi f_k (l-t_k)/fs + pi c_k ((l-t_k)/fs)^2))",
        "amplitudes": "least squares at every evaluation (column-scaled Cholesky, 2 refinement steps)",
        "optimizer": "scipy L-BFGS-B on scaled (t, log s, f, c); rescaled restarts of up to 5000 iterations; "
                     "stop when the match gains < 1e-6 over 500 iterations, or a wall-time cap",
        "bounds": "t in [0, L-1], s in [2, 4L] samples, f in [-1024, 1024] Hz, c free",
        "float32_valid_rule": f"float64 match minus float32 match <= {CL.F32_TOL}",
        "starts": {
            "track_n": "M points uniform in the quarter-cycle tiling count n(l) of the kernel's own track; f, df/dt from local "
                       "quadratic fits of the analytic phase; sigma = half the quarter-cycle interval",
            "track_e": "same, M points at equal steps of the sign's energy from the merger",
            "track_n_jJ": "track_n with jitter: t += 0.25 s N(0,1), f += 0.1 fs/(2 pi s) N(0,1), c *= 1 + 0.1 N(0,1), "
                          "s *= exp(0.2 N(0,1))",
            "random_J": "t ~ U(0, L), |f| ~ U(20, 800) Hz, c = 0, log s ~ U(log 32, log 1024) samples",
            "*_dom": "all M chirplets on the dominant frequency sign (default: split by the two signs' energies)",
            "sign_split": "n_dominant = round(M x dominant energy fraction), at least 1",
        },
    }, "per_mc": {}}
    best = CL.best_by(recs)
    for Mc in C.MCS:
        rs = [r for r in recs if r["Mc"] == Mc]
        d = {"curve_best": {}, "best_start": {}, "track_best": {}, "random": {}, "start_match": {},
             "dominant_sign_only_best": {}, "best_fit_diagnostics": {}, "per_start": {}}
        for M in MS:
            rm = [r for r in rs if r["M"] == M and CL.valid(r)]
            if not rm:
                continue
            b = best.get((Mc, M))
            d["curve_best"][M] = b["match_final"]
            d["best_start"][M] = b["start"]
            tr = [r["match_final"] for r in rm if r["start"].startswith("track") and not r["start"].endswith("_dom")]
            ra = [r["match_final"] for r in rm if r["start"].startswith("random")]
            do = [r["match_final"] for r in rm if r["start"].endswith("_dom")]
            d["track_best"][M] = max(tr) if tr else None
            d["track_spread"] = d.get("track_spread", {})
            d["track_spread"][M] = (max(tr) - min(tr)) if tr else None
            if ra:
                d["random"][M] = {"n": len(ra), "best": max(ra), "median": float(np.median(ra)), "worst": min(ra),
                                  "n_within_0.01_of_best_overall": int(sum(m >= b["match_final"] - 0.01 for m in ra)),
                                  "n_within_0.05_of_best_overall": int(sum(m >= b["match_final"] - 0.05 for m in ra))}
            if do:
                d["dominant_sign_only_best"][M] = max(do)
            d["start_match"][M] = {r["start"]: r["match_start"] for r in rm if r["start"] in ("track_n", "track_e")}
            d["per_start"][M] = {r["start"]: {"start": r["match_start"], "final": r["match_final"],
                                              "f32": r["match_final_float32"], "iters": r["n_iter"],
                                              "wall_s": r["wall_s"], "stop": r["messages"][-1] if r["messages"] else ""}
                                 for r in rs if r["M"] == M}
            dead = sum(c["energy_share"] < 1e-4 for c in b["chirplets"])
            d["best_fit_diagnostics"][M] = {"start": b["start"], "match_float32": b["match_final_float32"],
                                            "cancel_ratio": b["cancel_ratio"], "cond_colnorm_G": b["cond_colnorm_G"],
                                            "n_iter": b["n_iter"], "wall_s": b["wall_s"], "dead_chirplets": int(dead),
                                            "n_invalid_float32_fits_at_this_M": int(sum(1 for r in rs if r["M"] == M and not CL.valid(r)))}
        d["M_needed"] = needed(d["curve_best"], TH, MS)
        rc = res_c[Mc]
        d["resonators_from_file"] = {"running_max_curve": rc["running_max"], "thresholds": rc["thresholds_from_file"],
                                     "int_tau_df_dominant": rc["int_tau_df_dominant"],
                                     "int_tau_df_both_signs": rc["int_tau_df_both_signs"]}
        d["fourier_recomputed"] = four[Mc]
        nums = {}
        for lv in TH:
            key = f"{lv:.2f}"
            m = d["M_needed"][key]["smallest_tested"]
            p = rc["thresholds_from_file"].get(key, {}).get("smallest_tested_p") if key != "0.95" else rc["thresholds_from_file"]["0.95"]["smallest_tested_p"]
            nums[key] = {"chirplets_M": m, "chirplet_real_numbers": 6 * m if m else None,
                         "resonators_N": p, "resonator_real_numbers": 4 * p if p else None,
                         "fourier_tones_N": four[Mc]["N_needed"][key], "fourier_real_amplitudes": 2 * four[Mc]["N_needed"][key],
                         "free_kernel_complex": FREE_KERNEL_COMPLEX, "free_kernel_real_numbers": 2 * FREE_KERNEL_COMPLEX}
        d["learned_numbers"] = nums
        t = tj[str(Mc)]
        d["estimate"] = {"tiling_count_chirplet_one_sign": t["tiling_count_chirplet_one_sign"],
                         "tiling_count_tone_one_sign": t["tiling_count_tone_one_sign"],
                         "newtonian_tiling_count_chirplet": t["newtonian_tiling_count_chirplet"],
                         "newtonian_tiling_count_tone": t["newtonian_tiling_count_tone"],
                         "segments_needed_chirplet": t["segments_needed_chirplet"],
                         "segments_needed_tone": t["segments_needed_tone"],
                         "segments_needed_chirplet_dominant_sign_only": t["segments_needed_chirplet_dominant_sign_only"],
                         "energy_by_sign": t["energy_by_sign"]}
        out["per_mc"][str(Mc)] = d
    try:
        out["tuning"] = json.load(open(C.HERE / "tuning.json"))
    except FileNotFoundError:
        out["tuning"] = None

    def keys(o):
        if isinstance(o, dict):
            return {str(k): keys(v) for k, v in o.items()}
        if isinstance(o, list):
            return [keys(v) for v in o]
        return o

    json.dump(keys(out), open(C.HERE / "results.json", "w"), indent=1)
    figure(out, res_c, four)
    if out["tuning"]:
        tuning_figure(out["tuning"])
    for Mc in C.MCS:
        d = out["per_mc"][str(Mc)]
        print(f"Mc={Mc}: " + " ".join(f"{M}:{d['curve_best'][M]:.4f}" for M in MS if M in d["curve_best"]))
        print("   needed", {k: v["smallest_tested"] for k, v in d["M_needed"].items()},
              "estimate tiling", round(d["estimate"]["tiling_count_chirplet_one_sign"], 1),
              "segments", d["estimate"]["segments_needed_chirplet"])


def style(ax):
    ax.set_facecolor(SURF)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figure(out, res_c, four):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True, sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, Mc in zip(axes.flat, C.MCS):
        style(ax)
        d = out["per_mc"][str(Mc)]
        for lv in TH:
            ax.axhline(lv, color=GRID, lw=1.0, ls=(0, (4, 3)), zorder=0)
        Ms = sorted(d["curve_best"])
        ax.plot(Ms, [d["curve_best"][M] for M in Ms], color=BLUE, lw=2, marker="o", ms=6, label="chirplets (best fit)")
        if d["dominant_sign_only_best"]:
            Md = sorted(d["dominant_sign_only_best"])
            ax.plot(Md, [d["dominant_sign_only_best"][M] for M in Md], color=BLUE, lw=1.5, ls=(0, (2, 2)),
                    marker="o", ms=5, mfc="white", label="chirplets, one sign only")
        rc = res_c[Mc]["running_max"]
        ps = sorted(rc)
        ax.plot(ps, [rc[p] for p in ps], color=ORANGE, lw=2, ls="--", marker="s", ms=4.5, label="resonators (earlier fits)")
        fm = four[Mc]["match_at_N"]
        Ns = sorted(fm)
        ax.plot(Ns, [fm[n] for n in Ns], color=AQUA, lw=2, ls=":", marker="^", ms=5, label="Fourier tones (strongest N)")
        ax.set_xscale("log", base=2)
        ax.set_xlim(0.8, 5000)
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096])
        ax.set_xticklabels(["1", "4", "16", "64", "256", "1024", "4096"])
        ax.set_title(f"chirp mass {Mc} M$_\\odot$", color=INK, fontsize=11, loc="left")
        e = out["per_mc"][str(Mc)]["estimate"]["tiling_count_chirplet_one_sign"]
        ax.axvline(e, color=INK2, lw=1, ls=(0, (1, 2)))
        ax.text(e * 1.08, 0.04, f"quarter-cycle\ntiling {e:.1f}", color=INK2, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("number of components per kernel (chirplets, resonators or Fourier tones)", color=INK, fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("match (fraction of matched-filter SNR kept)", color=INK, fontsize=10)
    axes[0, 0].legend(loc="center right", fontsize=8.5, frameon=False, labelcolor=INK)
    fig.suptitle("Whitened BNS chirp kernels (4 s, 2048 Hz): match against number of components", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(C.HERE / "match_vs_M.png", dpi=130, facecolor=SURF)
    plt.close(fig)


def tuning_figure(tu):
    deltas = np.array(tu["deltas"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor(SURF)
    picks = sorted({int(k) for k in tu["chirplet"]} & {4, 8, 16, 32})
    for ax, (lo, hi) in zip(axes, ((-0.5, 0.5), (-10, 10))):
        style(ax)
        sel = (deltas >= lo) & (deltas <= hi)
        for col, M in zip(RAMP, picks):
            c = tu["chirplet"][str(M)]
            ax.plot(deltas[sel], np.array(c["match_max_over_lag"])[sel], color=col, lw=2, marker="o", ms=3.5,
                    label=f"{M} chirplets (fit match {c['fit_match_to_stored_kernel']:.3f})")
        ax.plot(deltas[sel], np.array(tu["exact"]["match_max_over_lag"])[sel], color=INK, lw=1.6, ls="--",
                label="exact 1.2 M$_\\odot$ kernel")
        ax.set_xlabel("true chirp mass offset from 1.2009 M$_\\odot$ (%)", color=INK, fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("match, maximized over time shift", color=INK, fontsize=10)
    axes[0].legend(loc="lower center", fontsize=8, frameon=False, labelcolor=INK)
    axes[0].set_title("near the fitted mass", color=INK, fontsize=10, loc="left")
    axes[1].set_title("wide range", color=INK, fontsize=10, loc="left")
    fig.suptitle("Tuning: kernels fitted to 1.2 M$_\\odot$ against the same source at other chirp masses", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(C.HERE / "tuning.png", dpi=130, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    main()
