"""Markdown tables for fisher.md, printed from the JSON outputs."""
import json

import numpy as np

D = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/"
F = json.load(open(D + "fisher_results.json"))
T = F["table"]
MCS = ["0.85", "1.2", "2.75"]
RS = ["2.0", "1.0", "0.44"]


def pc(x, d=4):
    return f"{100 * x:.{d}f}%"


print("### shared t_c (ordered)")
print("| chirp mass | coherent, 4 s | R = 2 s | R = 1 s | R = 0.44 s |")
print("|---|---|---|---|---|")
for mc in MCS:
    row = T[mc]
    cells = [pc(row["coherent"])]
    for R in RS:
        r = row[R]
        cells.append(f"{pc(r['shared_tc'])} (x{r['shared_tc'] / row['coherent']:.2f}; N = {r['N']})")
    print(f"| {mc} M☉ | " + " | ".join(cells) + " |")

print("\n### shared t_c, range over grid offsets 0 to 0.95 R")
print("| chirp mass | R = 2 s | R = 1 s | R = 0.44 s | R = 4 s |")
print("|---|---|---|---|---|")
for mc in MCS:
    row = T[mc]
    cells = [f"{pc(row[R]['offset_range'][0])} to {pc(row[R]['offset_range'][1])}" for R in RS + ["4.0"]]
    print(f"| {mc} M☉ | " + " | ".join(cells) + " |")

print("\n### free time per segment (not ordered)")
print("| chirp mass | R = 2 s | R = 1 s | R = 0.44 s | R = 0.44 s, offsets |")
print("|---|---|---|---|---|")
for mc in MCS:
    row = T[mc]
    cells = [f"{pc(row[R]['per_seg_time'])} (x{row[R]['per_seg_time'] / row['coherent']:.2f})" for R in RS]
    o = row["0.44"]["per_seg_time_offset_range"]
    cells.append(f"{pc(o[0])} to {pc(o[1])}")
    print(f"| {mc} M☉ | " + " | ".join(cells) + " |")

print("\n### closed form vs block inversion, max relative difference")
mx = 0.0
for mc in MCS:
    row = T[mc]
    mx = max(mx, abs(row["coherent_closed"] / row["coherent"] - 1))
    for R in ["0.44", "1.0", "2.0", "4.0"]:
        r = row[R]
        mx = max(mx, abs(r["shared_tc_closed"] / r["shared_tc"] - 1), abs(r["per_seg_time_closed"] / r["per_seg_time"] - 1))
print(f"max |closed/inverse - 1| = {mx:.2e}")

print("\n### ratio to coherent vs R (shared t_c), and local exponents")
for mc in MCS:
    c = F["convergence_ratio"][mc]
    Rv = np.array([float(k) for k in c])
    rv = np.array(list(c.values()))
    s = " ".join(f"{k}:{v:.4f}" for k, v in c.items())
    i = [list(Rv).index(x) for x in (0.25, 0.44, 1.0, 2.0)]
    ex = [np.log(rv[i[n]] / rv[i[n + 1]]) / np.log(Rv[i[n + 1]] / Rv[i[n]]) for n in range(3)]
    print(f"Mc {mc}: {s}; exponents 0.25-0.44 {ex[0]:.3f}, 0.44-1 {ex[1]:.3f}, 1-2 {ex[2]:.3f}")

print("\n### sensitivity")
S = F["sensitivity"]
print("| PSD | " + " | ".join(f"{mc}: coherent / R 0.44 / R 0.44 time" for mc in MCS) + " |")
print("|---|---|---|---|")
for kind in ("o3H1", "o3L1", "ajith"):
    cells = []
    for mc in MCS:
        v = S[kind][mc]
        cells.append(f"{pc(v['coherent'])} / {pc(v['R0.44'], 3)} / {pc(v['R0.44_t'], 3)}")
    print(f"| {kind} | " + " | ".join(cells) + " |")
cells = []
for mc in MCS:
    cells.append(f"{pc(T[mc]['coherent'])} / {pc(T[mc]['0.44']['shared_tc'], 3)} / {pc(T[mc]['0.44']['per_seg_time'], 3)}")
print("| O3 network (main) | " + " | ".join(cells) + " |")
cells = []
for mc in MCS:
    v = S["o3_cap512"][mc]
    cells.append(f"{pc(v['coherent'])} / {pc(v['R0.44'], 3)} / n.c.")
print("| O3 network, band cut at 512 Hz | " + " | ".join(cells) + " |")

print("\n### shares of SNR^2 per segment, R = 0.44 s (early to late), and rho_k at SNR 8")
for mc in MCS:
    fr = np.array(T[mc]["0.44"]["frac"])
    print(f"Mc {mc}: shares " + " ".join(f"{x:.3f}" for x in fr) + "; rho_k " + " ".join(f"{8 * np.sqrt(x):.2f}" for x in fr))

print("\n### checks")
print("tau check", F["check_tau_maxrel"], "white", F["check_white_maxrel"])
print("numeric derivs", F["check_numeric_derivs"])
for c in F["check_cf94"]:
    print(c)
print("sep-fisher compare", F["sep_fisher_compare"])

Dd = json.load(open(D + "detection_results.json"))
print("\n### detection")
print("| N | threshold on sum of N powers | SNR for 50% | ratio to N = 1 | volume ratio |")
print("|---|---|---|---|---|")
for N, v in Dd["plain_equal"].items():
    print(f"| {N} | {v['eta']:.2f} | {v['rho50']:.3f} | {v['ratio']:.3f} | {v['volume']:.2f} |")
for k, v in Dd["unequal"].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "frac"})
