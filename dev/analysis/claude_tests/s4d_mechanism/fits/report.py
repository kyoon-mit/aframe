"""Tables, best float32-valid match per p, threshold p, pole statistics, figures."""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "3"

import glob  # noqa: E402
import json  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

D = C.OUT_DIR
STARTS = ["chirp", "pencil_clipped", "pencil_reflected"]
THRESH = [0.5, 0.9, 0.95, 0.97, 0.99]
T_WIN = C.L * C.DT
UNDAMPED_ALPHA = 1e-12  # decay per sample below this counts as undamped (bound alpha = 0)

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"


def load():
    rep = json.load(open(D / "repro.json"))
    gro = json.load(open(D / "growing.json"))
    scan = json.load(open(D / "scan_chirp_placement.json"))
    tj = json.load(open(D / "tracks.json"))
    runs = {}
    for fn in glob.glob(str(D / "runs" / "*.path.json")):
        pj = json.load(open(fn))
        rj = json.load(open(fn.replace(".path.json", ".json")))
        runs[(float(rj["Mc"]), rj["start"], int(rj["p"]))] = {"run": rj, "path": pj, "npz": fn.replace(".path.json", ".npz")}
    rnd = {}
    for fn in glob.glob(str(D / "restarts" / "*.path.json")):
        pj = json.load(open(fn))
        rj = json.load(open(fn.replace(".path.json", ".json")))
        rnd.setdefault((float(rj["Mc"]), int(rj["p"])), []).append(
            {"run": rj, "path": pj, "npz": fn.replace(".path.json", ".npz")})
    return rep, gro, scan, tj, runs, rnd


def tables(rep, gro, scan, tj, runs, rnd):
    out = {}
    for Mc in C.MCS:
        m = str(Mc)
        sc = {r["p"]: r for r in scan[m]}
        rows = {}
        for p in sorted(set(C.NS) | set(sc)):
            row = {"p": p}
            if p in C.NS:
                row["fourier_bestN_undamped"] = rep["match"]["A_best_fourier"][m][str(p)]
                row["pencil_reflected_tone_check"] = rep["match"]["C_pencil_reflected"][m][str(p)]
                row["pencil_clipped_no_fit"] = rep["match"]["C_pencil_clipped"][m][str(p)]
                row["pencil_growing_allowed_not_s4d"] = gro[m][str(p)]["match"]
            if p in sc:
                row["chirp_placement_no_fit_f64"] = sc[p]["match"]
                row["chirp_placement_no_fit_f32"] = sc[p]["match_float32"]
            vp = {}
            for st in STARTS:
                r = runs.get((Mc, st, p))
                if r is None:
                    continue
                rj, pj = r["run"], r["path"]
                vp[st] = {
                    "start_f64": rj["match_start_lstsq"],
                    "final_f64_pure_ls": rj["match_final_lstsq"],
                    "final_f32": rj["match_final_float32"],
                    "best_f32_on_path": pj["best_f32"]["match_f32"],
                    "best_f32_iter": pj["best_f32"]["iter"],
                    "best_f32_its_f64": pj["best_f32"]["match_f64"],
                    "best_f32_cancel": pj["best_f32"]["cancel_ratio"],
                    "final_cancel": rj["cancel_ratio"],
                    "final_cond": rj["cond_V"],
                    "final_min_sep_over_width": pj["final"]["min_sep_over_width"],
                    "n_iter": rj["n_iter"],
                    "stop": rj["message"],
                }
            rr = rnd.get((Mc, p), [])
            if rr:
                f32s = [r["path"]["best_f32"]["match_f32"] for r in rr]
                ib = int(np.argmax(f32s))
                row["random_restarts"] = {
                    "n_seeds": len(rr),
                    "best_f32_on_path": f32s[ib],
                    "best_tag": rr[ib]["run"]["tag"],
                    "best_cancel": rr[ib]["path"]["best_f32"]["cancel_ratio"],
                    "median_best_f32": float(np.median(f32s)),
                    "best_final_f64_pure_ls": max(r["run"]["match_final_lstsq"] for r in rr),
                }
            if p == 8 and (Mc, "chirp", 8) in runs:
                row["chirp_placement_no_fit_f64"] = vp["chirp"]["start_f64"]
                row["chirp_placement_no_fit_f32"] = runs[(Mc, "chirp", 8)]["path"]["first"]["match_f32"]
            if vp:
                row["varpro"] = vp
                row["varpro_best_f32_valid"] = max(v["best_f32_on_path"] for v in vp.values())
                row["varpro_best_f64_pure_ls"] = max(v["final_f64_pure_ls"] for v in vp.values())
            cands = [row.get("varpro_best_f32_valid"), row.get("chirp_placement_no_fit_f32"),
                     row.get("fourier_bestN_undamped"),
                     row["random_restarts"]["best_f32_on_path"] if "random_restarts" in row else None]
            cands = [c for c in cands if c is not None]
            row["best_float32_valid_decaying"] = max(cands) if cands else None
            rows[p] = row
        dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
        thr = {}
        ps = sorted(rows)
        for t in THRESH:
            hit = [p for p in ps if rows[p]["best_float32_valid_decaying"] is not None
                   and rows[p]["best_float32_valid_decaying"] >= t]
            if hit:
                i = ps.index(hit[0])
                thr[str(t)] = {"smallest_tested_p": hit[0], "previous_tested_p": ps[i - 1] if i else None}
            else:
                thr[str(t)] = None
        out[m] = {"rows": rows, "thresholds_best_f32_valid": thr,
                  "int_tau_df_dominant": tj[m][dom]["int_tau_df"],
                  "int_tau_df_both_signs": tj[m]["pos"]["int_tau_df"] + tj[m]["neg"]["int_tau_df"],
                  "dominant_sign": dom}
    return out


def pole_table(alpha, beta, K, tr, Mc):
    """Per-pole frequency, ring time, tau at that frequency, energy share."""
    v = C.lstsq_full(-alpha + 1j * beta, K)
    f = beta / (2 * np.pi * C.DT)
    ring = np.where(alpha > UNDAMPED_ALPHA, C.DT / np.maximum(alpha, 1e-300), np.inf)
    tau = np.empty_like(f)
    for i, fi in enumerate(f):
        key = "pos" if fi >= 0 else "neg"
        af, tt = tr[f"af_{Mc}_{key}"], tr[f"tau_{Mc}_{key}"]
        tau[i] = np.interp(abs(fi), af, tt)
    e = (np.abs(v["c"]) * v["col_norm"]) ** 2
    share = e / e.sum()
    return {"f_hz": f, "ring_s": ring, "tau_s": tau, "share": share, "match": v["match"],
            "cancel": v["cancel_ratio"]}


def pole_summary(pt, band):
    f, ring, tau, sh = np.abs(pt["f_hz"]), pt["ring_s"], pt["tau_s"], pt["share"]
    inb = (f >= band[0]) & (f <= band[1])
    ratio = ring[inb] / tau[inb]
    fin = np.isfinite(ratio)
    out = {
        "n_poles": int(f.size),
        "n_in_band": int(inb.sum()),
        "n_undamped": int(np.sum(~np.isfinite(ring))),
        "n_ring_longer_than_window": int(np.sum(ring > T_WIN)),
        "median_ring_over_tau_in_band": float(np.median(ratio[fin])) if fin.any() else None,
        "q25_q75_ring_over_tau_in_band": [float(np.percentile(ratio[fin], 25)), float(np.percentile(ratio[fin], 75))] if fin.any() else None,
        "energy_weighted_median_freq_hz": float(f[np.argsort(f)][np.searchsorted(np.cumsum(sh[np.argsort(f)]), 0.5)]),
        "loglog_slope_ring_vs_tau_damped_in_band": None,
        "loglog_r2_ring_vs_tau_damped_in_band": None,
        "freq_range_hz": [float(f.min()), float(f.max())],
        "match": pt["match"], "cancel": pt["cancel"],
    }
    dm = inb & np.isfinite(ring) & (ring > 1.5 * C.DT)
    if dm.sum() >= 3:
        x, y = np.log(tau[dm]), np.log(ring[dm])
        A = np.vstack([x, np.ones_like(x)]).T
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        yhat = A @ coef
        r2 = 1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
        out["loglog_slope_ring_vs_tau_damped_in_band"] = float(coef[0])
        out["loglog_r2_ring_vs_tau_damped_in_band"] = float(r2)
        out["n_damped_in_band_used"] = int(dm.sum())
    return out


def style(ax):
    ax.set_facecolor("white")
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_match(T, tj):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), sharex=True)
    fig.patch.set_facecolor("white")
    xt = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    for j, Mc in enumerate(C.MCS):
        m = str(Mc)
        rows = T[m]["rows"]
        P = sorted(rows)
        Pn = [p for p in P if p in C.NS]
        Pv = [p for p in P if "varpro" in rows[p]]
        series = [
            ("Matrix pencil, growing poles reflected (tone_check)", Pn, [rows[p]["pencil_reflected_tone_check"] for p in Pn],
             dict(color=MUTED, ls="--", marker="s", mfc="white", lw=1.6)),
            ("Matrix pencil, growing poles clipped to |z|=1, no fit", Pn, [rows[p]["pencil_clipped_no_fit"] for p in Pn],
             dict(color=AQUA, ls="--", marker="^", lw=1.6)),
            ("Fourier tones, best N, undamped (tone_check)", Pn, [rows[p]["fourier_bestN_undamped"] for p in Pn],
             dict(color=YELLOW, ls=":", marker="D", lw=2)),
            ("Chirp placement (ring time = tau(f)), no fit", [p for p in P if "chirp_placement_no_fit_f32" in rows[p]],
             [rows[p]["chirp_placement_no_fit_f32"] for p in P if "chirp_placement_no_fit_f32" in rows[p]],
             dict(color=ORANGE, ls="-", marker="o", ms=3.5, lw=1.6)),
            ("VarPro, best float32-valid iterate, ordered starts (pencil x2, chirp)", Pv, [rows[p]["varpro_best_f32_valid"] for p in Pv],
             dict(color=BLUE, ls="-", marker="o", lw=2.2)),
            ("VarPro, random starts (best of 8-24 seeds), float32-valid", [p for p in P if "random_restarts" in rows[p]],
             [rows[p]["random_restarts"]["best_f32_on_path"] for p in P if "random_restarts" in rows[p]],
             dict(color=BLUE, ls="none", marker="*", ms=12)),
            ("VarPro, pure least squares in float64 (max over 3 starts)", Pv, [rows[p]["varpro_best_f64_pure_ls"] for p in Pv],
             dict(color=BLUE, ls="none", marker="o", mfc="white", ms=8, mew=1.5)),
            ("Matrix pencil, growing poles allowed (not S4D)", Pn, [rows[p]["pencil_growing_allowed_not_s4d"] for p in Pn],
             dict(color=MAGENTA, ls="-.", marker="v", lw=1.6)),
        ]
        ntb = T[m]["int_tau_df_dominant"]
        for i, ax in enumerate(axes[:, j]):
            style(ax)
            for lab, xs, ys, kw in series:
                ys = np.asarray(ys, float)
                yy = ys if i == 0 else np.clip(1.0 - ys, 1e-5, None)
                kw = dict(kw)
                kw.setdefault("ms", 5.5)
                ax.plot(xs, yy, label=lab if (i == 0 and j == 0) else None, **kw)
            ax.axvline(32, color=INK2, lw=0.8)
            ax.axvline(ntb, color=INK2, lw=0.8)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xt)
            ax.set_xticklabels([str(x) for x in xt])
            ax.minorticks_off()
            if i == 0:
                ax.set_ylim(0, 1.02)
                ax.set_title(f"chirp mass {Mc} (detector frame)", color=INK, fontsize=11)
                ax.text(32 * 0.94, 0.985, "S4D channel: 32", color=INK2, fontsize=8, rotation=90, va="top", ha="right")
                ax.text(ntb * 1.06, 0.02, f"integral of tau df\n(dominant sign) = {ntb:.0f}", color=INK2, fontsize=8,
                        rotation=90, va="bottom", ha="left")
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 1.2)
                ax.set_xlabel("number of poles p (complex modes in one channel)", color=INK, fontsize=10)
        axes[0, 0].set_ylabel("match with whitened chirp kernel", color=INK, fontsize=10)
        axes[1, 0].set_ylabel("1 - match (log scale)", color=INK, fontsize=10)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("How well p decaying poles (|z| <= 1) match a whitened BNS chirp kernel (8,064 samples, 2048 Hz)",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    fig.savefig(D / "match_vs_p.png", dpi=120, facecolor="white")
    plt.close(fig)


def fig_poles(sel, tj, tr):
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    fig.patch.set_facecolor("white")
    for j, Mc in enumerate(C.MCS):
        ax = axes[j]
        style(ax)
        m = str(Mc)
        dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
        af, tau = tr[f"af_{Mc}_{dom}"], tr[f"tau_{Mc}_{dom}"]
        band = tj[m][dom]["band_hz"]
        k = (af >= 20) & (af <= 1024)
        ax.plot(af[k], tau[k], color=INK, lw=1.6, label="tau(f): kernel group delay, monotone fit")
        ax.axhline(T_WIN, color=INK2, lw=0.8)
        ax.text(22, T_WIN * 1.12, "window 3.94 s", color=INK2, fontsize=8)
        ax.axvspan(band[0], band[1], color=GRID, alpha=0.35, lw=0)
        for (lab, color, mk, key) in (("VarPro p=32, best float32-valid", BLUE, "o", "p32"),
                                       ("VarPro p=128, best float32-valid", ORANGE, "s", "p128"),
                                       ("VarPro p=256 from chirp start, pure LS float64 (degenerate)", AQUA, "x", "p256deg")):
            pt = sel[m].get(key)
            if pt is None:
                continue
            f = np.abs(pt["f_hz"])
            ring = np.where(np.isfinite(pt["ring_s"]), pt["ring_s"], 60.0)
            size = 12 + 160 * np.sqrt(np.clip(pt["share"], 0, 1)) if key != "p256deg" else 14
            ax.scatter(f, ring, s=size, color=color, marker=mk, alpha=0.75 if key != "p256deg" else 0.6,
                       linewidths=0.8 if key == "p256deg" else 0.4, edgecolors="white" if key != "p256deg" else None,
                       label=lab if j == 0 else None, zorder=3)
        ax.axhline(60.0, color=GRID, lw=0.6)
        ax.text(22, 70, "undamped (|z| = 1)", color=INK2, fontsize=8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(20, 1100)
        ax.axhline(C.DT, color=GRID, lw=0.6)
        ax.text(1000, C.DT * 1.25, "1 sample (decay bound)", color=INK2, fontsize=8, ha="right")
        ax.set_ylim(3e-4, 150)
        ax.set_xlabel("pole frequency |f| (Hz)", color=INK, fontsize=10)
        ax.set_title(f"chirp mass {Mc}", color=INK, fontsize=11)
    axes[0].set_ylabel("ring time 1/a (s)", color=INK, fontsize=10)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("Optimized poles against the hold time tau(f) (shaded: 0.1-99.9% energy band; marker area ~ amplitude share)",
                 color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(D / "poles_vs_tau.png", dpi=120, facecolor="white")
    plt.close(fig)


def fig_scan(scan, tj):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    fig.patch.set_facecolor("white")
    cols = {1.0: MUTED, 1.2: BLUE, 1.5: ORANGE, 2.0: AQUA}
    for Mc in C.MCS:
        m = str(Mc)
        rows = scan[m]
        both = tj[m]["pos"]["int_tau_df"] + tj[m]["neg"]["int_tau_df"]
        dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
        denom = both if Mc == 1.0 else tj[m][dom]["int_tau_df"]
        x = np.array([r["p"] for r in rows]) / denom
        y = np.array([r["match_float32"] for r in rows])
        lab = f"Mc {Mc}" + (" (two signs: p / sum over both signs)" if Mc == 1.0 else "")
        for i, ax in enumerate(axes):
            style(ax)
            ax.plot(x, y if i == 0 else 1 - y, color=cols[Mc], marker="o", ms=4.5, lw=1.8,
                    ls="--" if Mc == 1.0 else "-", label=lab)
    for i, ax in enumerate(axes):
        ax.axvline(1.0, color=INK2, lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("p / integral of tau(f) df  (poles per hold-time cell)", color=INK, fontsize=10)
        if i == 0:
            ax.set_ylabel("match (chirp placement, no fit, float32)", color=INK, fontsize=10)
            ax.set_ylim(0, 1.02)
            ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
        else:
            ax.set_yscale("log")
            ax.set_ylabel("1 - match", color=INK, fontsize=10)
    fig.suptitle("One pole per band of equal integral of tau df, ring time tau(f): match jumps when p reaches the integral",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(D / "placement_scan.png", dpi=120, facecolor="white")
    plt.close(fig)


def main():
    rep, gro, scan, tj, runs, rnd = load()
    T = tables(rep, gro, scan, tj, runs, rnd)
    ker = np.load(D / "kernels.npz")
    tr = np.load(D / "tracks.npz")
    sel, psum = {}, {}
    for Mc in C.MCS:
        m = str(Mc)
        K = ker[f"K_{Mc}"]
        dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
        band = tj[m][dom]["band_hz"]
        sel[m], psum[m] = {}, {}
        for p in (32, 128, 256, 1024):
            cand = [(runs[(Mc, st, p)]["path"]["best_f32"]["match_f32"], st, runs[(Mc, st, p)]["npz"])
                    for st in STARTS if (Mc, st, p) in runs]
            cand += [(r["path"]["best_f32"]["match_f32"], r["run"]["start"], r["npz"]) for r in rnd.get((Mc, p), [])]
            if not cand:
                continue
            best, st, npz = max(cand)
            d = np.load(npz.replace(".npz", ".bestf32.npz"))
            pt = pole_table(d["alpha"], d["beta"], K, tr, Mc)
            sel[m][f"p{p}"] = pt
            psum[m][f"p{p}_best_f32_valid"] = dict(pole_summary(pt, band), start=st)
        if (Mc, "chirp", 256) in runs:
            d = np.load(runs[(Mc, "chirp", 256)]["npz"])
            pt = pole_table(d["alpha"], d["beta"], K, tr, Mc)
            sel[m]["p256deg"] = pt
            psum[m]["p256_chirp_start_pure_ls_final"] = pole_summary(pt, band)
        # p=32 pole list (best float32-valid)
        if "p32" in sel[m]:
            pt = sel[m]["p32"]
            o = np.argsort(np.abs(pt["f_hz"]))
            psum[m]["p32_pole_list"] = [
                {"f_hz": round(float(pt["f_hz"][i]), 2),
                 "ring_s": (round(float(pt["ring_s"][i]), 4) if np.isfinite(pt["ring_s"][i]) else "undamped"),
                 "tau_s": round(float(pt["tau_s"][i]), 4), "share": round(float(pt["share"][i]), 4)} for i in o]
    fig_match(T, tj)
    fig_poles(sel, tj, tr)
    fig_scan(scan, tj)
    json.dump({"tables": T, "poles": psum}, open(D / "summary.json", "w"), indent=1, default=str)
    for Mc in C.MCS:
        m = str(Mc)
        print(f"\n== Mc {Mc}  (int tau df dominant {T[m]['int_tau_df_dominant']:.1f}, both {T[m]['int_tau_df_both_signs']:.1f})")
        print("   p   refl   clip  fourier chirpPl  VP_f32  VP_f64  grow   bestValid | per start best_f32/final_f64/final_f32")
        for p in sorted(T[m]["rows"]):
            r = T[m]["rows"][p]
            g = lambda k: f"{r[k]:.4f}" if r.get(k) is not None else "   -  "  # noqa: E731
            vp = r.get("varpro", {})
            det = " ".join(f"{st[:6]}:{v['best_f32_on_path']:.4f}/{v['final_f64_pure_ls']:.4f}/{v['final_f32']:.4f}" for st, v in vp.items())
            if "random_restarts" in r:
                rr = r["random_restarts"]
                det += f" random(n={rr['n_seeds']}):{rr['best_f32_on_path']:.4f}/{rr['best_final_f64_pure_ls']:.4f}"
            print(f"{p:5d} {g('pencil_reflected_tone_check')} {g('pencil_clipped_no_fit')} {g('fourier_bestN_undamped')} "
                  f"{g('chirp_placement_no_fit_f32')} {g('varpro_best_f32_valid')} {g('varpro_best_f64_pure_ls')} "
                  f"{g('pencil_growing_allowed_not_s4d')} {g('best_float32_valid_decaying')} | {det}")
        print("   thresholds (best float32-valid):", T[m]["thresholds_best_f32_valid"])
        for k, v in psum[m].items():
            if k != "p32_pole_list":
                print("   poles", k, {kk: vv for kk, vv in v.items()})


if __name__ == "__main__":
    main()
