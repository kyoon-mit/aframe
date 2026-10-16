"""Rebuild the numeric tables of theory.md from the saved results and report
any table row that is not found verbatim in theory.md."""
import json
import math
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
doc = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "theory.md")).read()
rows = []
c3 = json.load(open(f"{R}/c3_count.json"))
for m in ("0.85", "1.2", "1.8", "2.75"):
    a = c3["A_integrals"][m]
    rows.append(f"| {m} M☉ | {a['f1_Hz']:.2f} Hz | {a['to_merger']['cycles_closed']:.1f} | "
                f"{a['to_merger']['int_tau_df_closed']:.1f} | {a['to_900Hz']['int_tau_df_closed']:.1f} |")
p1 = json.load(open(f"{R}/c5_fits_1.json"))["part1"]
for c, lab in ((0.8, "0.8"), (1.0, "1.0"), (1.1, "1.1"), (1.5, "1.5"), (2.0, "2.0"), (math.e, "$e$")):
    vals = []
    for m in ("0.85", "1.2", "1.8", "2.75"):
        rr = [r for r in p1[m]["rows"] if abs(r["c"] - c) < 1e-9]
        vals.append(max(r["match_2"] for r in rr))
    rows.append(f"| {lab} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
p3 = json.load(open(f"{R}/c5_fits_3.json"))["part3"]["1.2"]
for r in p3:
    I = r["int_tau_df"]
    fu = r["fourier_per_int_tau"]
    fus = f"{fu:.2f}" if fu < 10 else (f"{fu:.1f}" if fu < 100 else f"{fu:.0f}")
    Is = f"{I:.1f}" if I >= 10 else f"{I:.2f}"
    rows.append(f"| {r['band'][0]} to {r['band'][1]} Hz | {Is} | {r['resonators_0.9_cap2']['N']} | "
                f"{r['resonators_per_int_tau']:.2f} | {r['fourier_count_0.9']} | {fus} |")
p2 = json.load(open(f"{R}/c5_fits_2.json"))["part2"]
for m in ("0.85", "1.2", "1.8", "2.75"):
    d = p2[m]
    rows.append(f"| {m} M☉ | {d['best_cap_2']['N']} | {d['best_cap_inf']['N']} | "
                f"{d['phase_budget_N_0.9']:.0f} | {d['fourier_count_0.9']} |")
c6 = json.load(open(f"{R}/c6_snr.json"))["C_templates"]
for m in ("0.85", "1.2", "1.8", "2.75"):
    e = c6[m]
    rows.append(f"| {m} M☉ | {e['fourier_count_0.9_analytic_hplus']} | {e['model_fourier_N_0.9']:.0f} | "
                f"{e['model_res_N_0.9_c1.000']:.0f} | {e['model_res_N_0.9_c2.718']:.0f} | {e['fit_N_0.9_cap2']['N']} |")
for n in ("16", "32", "64", "128", "256"):
    rows.append(f"| {n} | " + " | ".join(f"{c6[m]['fit_match_at_N_cap2'][n]['match']:.3f}"
                                         for m in ("0.85", "1.0", "1.2", "1.5", "1.8", "2.0", "2.75")) + " |")
bad = 0
for r in rows:
    ok = r in doc
    bad += not ok
    print("OK  " if ok else "DIFF", r)
print("rows not found verbatim:", bad)
