"""Summarize out/b_cnn_mc.json as a Markdown table (out/b_cnn_table.md):
first-order prediction and odd part at A = 8, even part at A = 1, 4, 8, 16,
the log-log slope of |even| over A = 1, 2, 4, 8, and the ratio even(16)/even(8)."""
import json
import numpy as np

J = json.load(open("out/b_cnn_mc.json"))
rows = []
hdr = ("| network | signal | 1st-order prediction, A = 8 | odd part, A = 8 | even part, A = 1 | A = 4 | A = 8 | A = 16 "
       "| slope of log abs(even), A = 1 to 8 |")
rows.append(hdr)
rows.append("|" + "---|" * 9)
notes = []
for key, r in J.items():
    net, sig = [x.strip() for x in key.split("|")]
    pred = r.get("first_order_exact_per_unit_A", r["first_order_snr_per_unit_A"])
    src = "a" if "first_order_exact_per_unit_A" in r else "autograd"
    od, ods = r["odd"]["8.0"]
    ev = {a: r["even"][a] for a in ("1.0", "2.0", "4.0", "8.0", "16.0")}
    A = np.array([1, 2, 4, 8.0]); E = np.array([abs(ev[f"{a:.1f}"][0]) for a in A])
    slope = np.polyfit(np.log(A), np.log(E), 1)[0] if np.all(E > 0) else float("nan")
    f = lambda v: f"{v[0]:+.4f} ± {v[1]:.4f}"
    pred_s = f"{8*pred:+.4f}" + ("" if src == "a" else f" ± {8*r['first_order_snr_per_unit_A_se']:.4f}")
    rows.append(f"| {net} | {sig} | {pred_s} | {od:+.4f} ± {ods:.4f} | {f(ev['1.0'])} | {f(ev['4.0'])} | {f(ev['8.0'])} | {f(ev['16.0'])} | {slope:.3f} |")
    if "sHs" in r:
        h, hs = r["sHs"]
        notes.append(f"{key}: second-order prediction (A^2/2) E[s^T Hess s] / sigma_o at A = 1, 4, 8, 16: " +
                     ", ".join(f"{0.5*a*a*h/r['sigma_o']:+.4f} ± {0.5*a*a*hs/r['sigma_o']:.4f}" for a in (1, 4, 8, 16)))
    ws = r["w_structure"]
    notes.append(f"{key}: sigma_o = {r['sigma_o']:.4g}; w interior mean {ws['interior_mean']:.3e}, rms about its S-periodic pattern {ws['interior_resid_rms']:.2e}"
                 + (f"; rms(w_autograd - w_from_a)/rms(w_from_a) = {r['w_autograd_vs_exact_rms']/r['w_exact_rms']:.3f}" if 'w_exact_rms' in r else "")
                 + (f"; interior part of the first-order prediction per unit A {r['first_order_interior_part']:+.6f}" if 'first_order_interior_part' in r else "")
                 + (f"; S-periodic pattern {['%.3e' % p for p in ws['interior_pattern']]}" if r['S'] > 1 else ""))
out = "\n".join(rows) + "\n\n" + "\n".join("- " + n for n in notes) + "\n"
print(out)
open("out/b_cnn_table.md", "w").write(out)
