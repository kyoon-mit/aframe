"""Replot the three 2026-10-16 BNS site figures from saved results, in the site's colors.

Usage: python -B replot.py {count,fits,ring,all} [--out DIR]
"""

import argparse
import ast
import io
import json
import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "2"
sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
FITS = HERE.parent / "fits"
KERN = HERE.parent / "kernels"
SITE = Path("/home/kyoon/SSM-BNS/WEB/kyoon-mit.github.io/docs/assets/img/bns/2026-10-16")

# site tokens (docs/assets/css/main.css)
PAPER, INK, INK2, INK3 = "#ffffff", "#0f1e33", "#34435a", "#5d6b80"
RULE2, ACCENT, WARN = "#c5cdd9", "#13386b", "#b42318"

WIDTH_PX = 1400
MAX_BYTES = 250_000

plt.rcParams.update({
    "font.family": "DejaVu Sans",  # IBM Plex Sans is not installed here
    "font.size": 12,
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": True,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.edgecolor": RULE2, "axes.linewidth": 0.9,
    "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titleweight": "normal",
    "text.color": INK,
    "xtick.color": RULE2, "ytick.color": RULE2,
    "xtick.labelcolor": INK3, "ytick.labelcolor": INK3,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": RULE2, "grid.alpha": 0.5, "grid.linewidth": 0.6,
    "legend.frameon": False, "legend.labelcolor": INK2,
})
HALO = [pe.withStroke(linewidth=4.2, foreground=PAPER)]


def pick(path, names, ns):
    """Execute only the named top-level functions and assignments of a source file."""
    body = []
    for n in ast.parse(Path(path).read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            body.append(n)
        elif isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets):
            body.append(n)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    missing = [k for k in names if k not in ns]
    assert not missing, missing
    return ns


MCS = pick(FITS / "common.py", ["MCS"], {})["MCS"]


def save(fig, path, colors=256):
    """PNG at WIDTH_PX wide, quantized to a palette."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=WIDTH_PX / fig.get_figwidth())
    plt.close(fig)
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    size = os.path.getsize(path)
    print(f"{path}: {im.size[0]}x{im.size[1]} px, {size / 1000:.1f} kB")
    assert im.size[0] == WIDTH_PX and size < MAX_BYTES, (im.size, size)


def style_axes(ax, size=11):
    ax.tick_params(labelsize=size, length=3.5, width=0.8)


# ------------------------------------------------------------------ figure 1
def fig_count(out):
    """Resonators placed along the chirp, no fit: match against count / integral of tau df."""
    scan = json.load(open(FITS / "scan_chirp_placement.json"))
    tj = json.load(open(FITS / "tracks.json"))
    sty = {
        1.0: dict(color=INK3, ls=(0, (5, 3)), marker="o", mfc=PAPER, mew=1.5),
        1.2: dict(color=ACCENT, ls="-", marker="o"),
        1.5: dict(color=INK2, ls="-", marker="s", mfc=PAPER, mew=1.5),
        2.0: dict(color=INK, ls="-", marker="^"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.9), layout="constrained")
    for Mc in MCS:
        m = str(Mc)
        rows = scan[m]
        both = tj[m]["pos"]["int_tau_df"] + tj[m]["neg"]["int_tau_df"]
        dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
        denom = both if Mc == 1.0 else tj[m][dom]["int_tau_df"]
        x = np.array([r["p"] for r in rows]) / denom
        y = np.array([r["match_float32"] for r in rows])
        lab = f"chirp mass {Mc} M☉"
        if Mc == 1.0:
            lab += "\n(∫τ(f) df summed over\nboth signs of frequency)"
        for i, ax in enumerate(axes):
            ax.plot(x, y if i == 0 else 1 - y, lw=2.0, ms=6.5, label=lab, **sty[Mc])
    for i, ax in enumerate(axes):
        style_axes(ax, 12)
        ax.axvline(1.0, color=INK2, lw=1.0, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("number of resonators / ∫τ(f) df", fontsize=13)
        if i == 0:
            ax.set_ylabel("match", fontsize=13)
            ax.set_ylim(0, 1.02)
            ax.legend(fontsize=12, loc="upper left", handlelength=2.6)
        else:
            ax.set_yscale("log")
            ax.set_ylabel("1 − match", fontsize=13)
    fig.suptitle("One resonator per band of equal ∫τ(f) df, ring time τ(f), no fit, match in float32:\n"
                 "the match jumps when the number of resonators reaches ∫τ(f) df", fontsize=14, color=INK)
    save(fig, out / "match_vs_resonator_count.png")


# ------------------------------------------------------------------ figure 2
def fig_fits(out):
    """Every fit: match and 1 - match against the number of resonators, four chirp masses."""
    T = json.load(open(FITS / "summary.json"))["tables"]
    xt = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    fig, axes = plt.subplots(2, 4, figsize=(14, 9.6), sharex=True, layout="constrained")
    for j, Mc in enumerate(MCS):
        m = str(Mc)
        rows = {int(k): v for k, v in T[m]["rows"].items()}
        P = sorted(rows)
        Pn = [p for p in P if "pencil_reflected_tone_check" in rows[p]]
        Pv = [p for p in P if "varpro" in rows[p]]
        Pc = [p for p in P if "chirp_placement_no_fit_f32" in rows[p]]
        Pr = [p for p in P if "random_restarts" in rows[p]]
        # same series and drawing order as fits/report.py fig_match
        series = [
            ("refl", Pn, [rows[p]["pencil_reflected_tone_check"] for p in Pn],
             dict(color=INK3, ls=(0, (5, 3)), marker="s", mfc=PAPER, mew=1.4, lw=1.5)),
            ("clip", Pn, [rows[p]["pencil_clipped_no_fit"] for p in Pn],
             dict(color=INK3, ls=(0, (5, 3)), marker="^", lw=1.5)),
            ("four", Pn, [rows[p]["fourier_bestN_undamped"] for p in Pn],
             dict(color=INK2, ls=(0, (1, 2)), marker="D", ms=5.0, lw=2.0)),
            ("chirp", Pc, [rows[p]["chirp_placement_no_fit_f32"] for p in Pc],
             dict(color=INK, ls="-", marker="o", ms=3.4, lw=1.3)),
            ("vp", Pv, [rows[p]["varpro_best_f32_valid"] for p in Pv],
             dict(color=ACCENT, ls="-", marker="o", lw=2.6)),
            ("rand", Pr, [rows[p]["random_restarts"]["best_f32_on_path"] for p in Pr],
             dict(color=ACCENT, ls="none", marker="*", ms=12.5)),
            ("f64", Pv, [rows[p]["varpro_best_f64_pure_ls"] for p in Pv],
             dict(color=ACCENT, ls="none", marker="o", mfc=PAPER, ms=8.5, mew=1.6)),
            ("grow", Pn, [rows[p]["pencil_growing_allowed_not_s4d"] for p in Pn],
             dict(color=INK3, ls=(0, (6, 2, 1.5, 2)), marker="v", mfc=PAPER, mew=1.3, lw=1.5)),
        ]
        ntb = T[m]["int_tau_df_dominant"]
        for i, ax in enumerate(axes[:, j]):
            style_axes(ax, 10.5)
            for key, xs, ys, kw in series:
                ys = np.asarray(ys, float)
                yy = ys if i == 0 else np.clip(1.0 - ys, 1e-5, None)
                kw = dict(kw)
                kw.setdefault("ms", 5.5)
                ax.plot(xs, yy, label=key if (i == 0 and j == 0) else None, **kw)
            ax.axvline(32, color=INK2, lw=1.0, zorder=1)
            ax.axvline(ntb, color=INK2, lw=1.0, zorder=1)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xt)
            ax.set_xticklabels([str(x) if k % 2 == 0 else "" for k, x in enumerate(xt)])
            ax.minorticks_off()
            if i == 0:
                ax.set_ylim(0, 1.02)
                ax.set_title(f"chirp mass {Mc} M☉ (detector frame)", fontsize=11)
                ax.text(32 * 0.93, 0.985, "S4D channel: 32 resonators", color=INK2, fontsize=9.5, rotation=90,
                        va="top", ha="right")
            else:
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 1.2)
                ax.set_xlabel("number of resonators", fontsize=12)
                ax.text(ntb * 0.9, 1.4e-4, f"∫τ(f) df = {ntb:.0f}\n(dominant sign of frequency)", color=INK2,
                        fontsize=9, rotation=90, va="bottom", ha="right")
        axes[0, 0].set_ylabel("match", fontsize=12)
        axes[1, 0].set_ylabel("1 − match", fontsize=12)
    names = {
        "refl": "earlier fit (matrix pencil), growing resonators reflected",
        "clip": "earlier fit (matrix pencil), growing resonators clipped to |z| = 1, not refitted",
        "grow": "earlier fit (matrix pencil), growing resonators allowed (not usable in S4D)",
        "four": "Fourier components, strongest N, undamped (earlier fit)",
        "chirp": "resonators along the chirp, ring time = τ(f), no fit",
        "vp": "optimized decaying resonators, 3 fixed starts (2 matrix pencil, 1 chirp),\nbest step valid in float32",
        "rand": "optimized decaying resonators, random starts (best of 8 to 24), valid in float32",
        "f64": "optimized decaying resonators, match in float64 only (best of 3 starts)",
    }
    h, lab = axes[0, 0].get_legend_handles_labels()
    by = dict(zip(lab, h))
    order = ["refl", "clip", "grow", "four", "chirp", "vp", "rand", "f64"]
    fig.legend([by[k] for k in order], [names[k] for k in order], loc="outside lower center", ncol=2,
               fontsize=10.5, handlelength=3.2, columnspacing=2.0)
    fig.suptitle("How well decaying resonators (|z| ≤ 1) in one channel match a whitened BNS chirp kernel "
                 "(8,064 samples, 2048 Hz)", fontsize=13.5, color=INK)
    save(fig, out / "match_vs_resonators_all_fits.png")


# ------------------------------------------------------------------ figure 3
def fig_ring(out, ref_color=WARN):
    """Ring time of every trained resonator against tau(f) at its frequency, four S4D networks."""
    kc = pick(KERN / "kcommon.py", ["TSUN", "MERGE", "tau_of_f", "fdot", "t_step"], {"np": np})
    fg = pick(KERN / "figures.py", ["modes_inband", "init_inband"], {"np": np, "os": os, "H": str(KERN)})
    summary = json.load(open(KERN / "summary.json"))
    nets = [
        ("noisy_s4d_d64", "Noisy, S4D, 32 resonators per channel (64 states per channel)"),
        ("noisy_s4d_d512", "Noisy, S4D, 256 resonators per channel (512 states per channel)"),
        ("noisy_s4d_d512_band", "Noisy, S4D, 256 resonators per channel, band start"),
        ("clean_s4d_layer1", "Clean, S4D 64 × 32 × 4, first layer"),
    ]
    cmap = LinearSegmentedColormap.from_list("site_navy", [INK3, ACCENT, INK])
    fig, axs = plt.subplots(2, 2, figsize=(14, 11.6), layout="constrained")
    xx = np.geomspace(1e-3, 1e3, 200)
    t_start = kc["MERGE"] / 2048
    for ax, (n, title) in zip(axs.ravel(), nets):
        style_axes(ax, 11.5)
        f, ring, w = fg["modes_inband"](n)
        o = np.argsort(w)
        tau = kc["tau_of_f"](f, 1.2)
        fi, ri = fg["init_inband"](n)
        ax.scatter(kc["tau_of_f"](fi, 1.2), ri, s=2.4, color=RULE2, linewidths=0, rasterized=True,
                   label="same network at its start (fresh draw)")
        sc = ax.scatter(tau[o], ring[o], c=w[o], s=3.2, cmap=cmap, norm=LogNorm(1e-5, 1), linewidths=0,
                        rasterized=True)
        for k, mc in enumerate((2.75, 0.85)):
            ax.plot(xx, xx * (1.2 / mc) ** (5 / 3), color=ref_color, lw=1.3, ls=(0, (5, 3)),
                    label="ring time = τ(f), chirp mass 0.85 and 2.75 M☉" if k == 0 else None)
        ax.plot(xx, xx, color=ref_color, lw=2.2, path_effects=HALO, label="ring time = τ(f), 1.2 M☉")
        fl = np.geomspace(20, 1024, 200)
        ax.plot(kc["tau_of_f"](fl, 1.2), kc["t_step"](fl, 1.2), color=INK, lw=1.8, ls=(0, (7, 2, 1.5, 2)),
                path_effects=HALO, label="in-step time ḟ^(−1/2), 1.2 M☉")
        ax.axvline(t_start, color=INK3, lw=1.0)
        ax.text(t_start * 1.15, 1.6e-4, "chirp starts in the window\n(τ = 3.75 s)", color=INK2, fontsize=10,
                bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.85, pad=1.5))
        ax.axhline(4.0, color=INK3, lw=1.0)
        ax.text(1.3e-3, 5.0, "kernel length 4 s", color=INK2, fontsize=10,
                bbox=dict(facecolor=PAPER, edgecolor="none", alpha=0.85, pad=1.5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 400)
        ax.set_ylim(1e-4, 40)
        ax.set_xlabel("time before merger τ(f) (s)", fontsize=12.5)
        ax.set_ylabel("ring time (s)", fontsize=12.5)
        nm = summary[n]["modes"]
        ax.set_title(f"{title}\n{nm['inband_20Hz_to_nyquist']:,} of {nm['total']:,} resonators between 20 Hz and Nyquist",
                     fontsize=12.5)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2, fontsize=11.5, markerscale=5, handlelength=3.2,
               columnspacing=2.5)
    cb = fig.colorbar(sc, ax=axs, shrink=0.6)
    cb.set_label("trained resonator's share of its channel's energy", fontsize=12, color=INK2)
    cb.outline.set_edgecolor(RULE2)
    cb.ax.tick_params(labelsize=11, colors=RULE2, labelcolor=INK3)
    fig.suptitle("Ring time of each resonator against the time before merger τ(f) at its frequency, chirp mass 1.2 M☉",
                 fontsize=14, color=INK)
    save(fig, out / "ring_time_vs_hold_time.png", colors=128)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["count", "fits", "ring", "all"])
    ap.add_argument("--out", type=Path, default=SITE)
    ap.add_argument("--ring-ref", choices=["warn", "ink"], default="warn")
    a = ap.parse_args()
    if a.which in ("count", "all"):
        fig_count(a.out)
    if a.which in ("fits", "all"):
        fig_fits(a.out)
    if a.which in ("ring", "all"):
        fig_ring(a.out, WARN if a.ring_ref == "warn" else INK)
