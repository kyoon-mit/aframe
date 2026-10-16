"""Read the saved fit results behind figures 1 and 2 and print the values the page text quotes."""
import json
import sys
from pathlib import Path

import numpy as np

FITS = Path(sys.argv[1])
scan = json.load(open(FITS / "scan_chirp_placement.json"))
tj = json.load(open(FITS / "tracks.json"))
T = json.load(open(FITS / "summary.json"))["tables"]
MCS = [1.0, 1.2, 1.5, 2.0]

print("== figure 1: x = p / int_tau_df (denominator as in replot.py), match_float32")
for Mc in MCS:
    m = str(Mc)
    both = tj[m]["pos"]["int_tau_df"] + tj[m]["neg"]["int_tau_df"]
    dom = "pos" if tj[m]["energy_frac"]["pos"] >= 0.5 else "neg"
    denom = both if Mc == 1.0 else tj[m][dom]["int_tau_df"]
    rows = scan[m]
    x = np.array([r["p"] for r in rows]) / denom
    y = np.array([r["match_float32"] for r in rows])
    p = np.array([r["p"] for r in rows])
    print(f"Mc {Mc}: dom {dom}, energy_frac {tj[m]['energy_frac']}, int_tau_df pos {tj[m]['pos']['int_tau_df']:.2f} "
          f"neg {tj[m]['neg']['int_tau_df']:.2f} both {both:.2f}, denom {denom:.2f}")
    sel = (x > 0.6) & (x < 1.6)
    for pi, xi, yi in zip(p[sel], x[sel], y[sel]):
        print(f"   p {pi:5d}  x {xi:.3f}  match {yi:.4f}")
    print(f"   x range {x.min():.4f} to {x.max():.3f}; max match {y.max():.4f}; min 1-match {1 - y.max():.2e}")

print("== figure 2: rows at max N, and chirp placement reaching 0.99")
for Mc in MCS:
    m = str(Mc)
    rows = {int(k): v for k, v in T[m]["rows"].items()}
    P = sorted(rows)
    Pn = [q for q in P if "pencil_reflected_tone_check" in rows[q]]
    Pc = [q for q in P if "chirp_placement_no_fit_f32" in rows[q]]
    Pv = [q for q in P if "varpro" in rows[q]]
    Pr = [q for q in P if "random_restarts" in rows[q]]
    top = max(Pn)
    r = rows[top]
    print(f"Mc {Mc}: int_tau_df_dominant {T[m]['int_tau_df_dominant']}; Pn {Pn}")
    print(f"   at N={top}: reflected {r['pencil_reflected_tone_check']:.4f} clipped {r['pencil_clipped_no_fit']:.4f} "
          f"fourier {r['fourier_bestN_undamped']:.4f} growing {r['pencil_growing_allowed_not_s4d']:.6f}")
    c99 = [q for q in Pc if rows[q]["chirp_placement_no_fit_f32"] >= 0.99]
    print(f"   chirp placement Pc {Pc}")
    print(f"   chirp placement first N with match >= 0.99: {c99[:1]} "
          f"(value {rows[c99[0]]['chirp_placement_no_fit_f32']:.4f})" if c99 else "   never 0.99")
    print(f"   varpro Pv {Pv}: f32 valid " + ", ".join(f"{q}:{rows[q]['varpro_best_f32_valid']:.4f}" for q in Pv))
    print(f"   f64 pure ls " + ", ".join(f"{q}:{rows[q]['varpro_best_f64_pure_ls']:.5f}" for q in Pv))
    print(f"   random restarts Pr {Pr}: " + ", ".join(
        f"{q}:{rows[q]['random_restarts']['best_f32_on_path']:.4f} (keys {sorted(rows[q]['random_restarts'].keys())})" for q in Pr))
