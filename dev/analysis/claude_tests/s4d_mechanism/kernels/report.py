"""Print the numbers kernels.md quotes, from summary.json and the per-network tables."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

S = json.load(open(os.path.join(kc.HERE, "summary.json")))
NETS = ["noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "noisy_free", "clean_s4d_layer1", "clean_free"]
for n in NETS:
    s = S[n]
    print(f"\n### {n}")
    print(json.dumps(s["info"]))
    for k in ("best_inband_match", "best_match_unit_norm", "best_match_network_energy", "best_inband_match_with_D",
              "one_resonance_bound", "gain_over_one_resonance", "gd_slope_vs_chirp", "energy_share_20Hz_to_nyquist",
              "energy_share_negative_freq", "channels_inband_side_negative", "D2_over_kernel_norm2", "lag90_s", "best_chirp_mass",
              "class_counts", "class_counts_if_G_cut_1.5", "class_counts_if_G_cut_3.0", "class_counts_top_quarter_by_readout_weight"):
        if k in s:
            print(f"  {k}: {s[k]}")
    for k, v in s.get("by_class", {}).items():
        print(f"  [{k}] {v}")
    if "modes" in s:
        for k, v in s["modes"].items():
            print(f"  modes.{k}: {v}")
    T = json.load(open(os.path.join(kc.HERE, f"{n}_table.json")))["channels"]
    mi = np.array([r["best_inband_match"] if r["best_inband_match"] is not None else np.nan for r in T])
    top = np.argsort(np.nan_to_num(mi))[::-1][:5]
    for c in top:
        r = T[c]
        extra = {k: r.get(k) for k in ("n_inband", "n_modes_side_in_chirp_window_band", "ring_wmedian_s", "ring_over_tau_wmedian",
                                        "ring_over_tstep_wmedian", "energy_share_20Hz_to_nyquist", "energy_share_negative_freq",
                                        "D2_over_kernel_norm2", "readout_weight_norm")}
        print(f"  top ch {c}: inband {r['best_inband_match']:.3f} (unit-norm {r.get('best_match_real', r.get('best_match_pos')):.3f}), "
              f"Mc {r['best_chirp_mass']:.3f}, band {r['f10']:.0f}-{r['f90']:.0f} Hz, G {r['gain_over_one_resonance']:.2f}, "
              f"one-res {r['one_resonance_bound']:.3f} at {r['one_resonance_bound_f0']:.0f} Hz ring {r['one_resonance_bound_ring_s']:.3f} s, "
              f"slope {r['gd_slope_vs_chirp']}, lag90 {r['lag90_s']:.3f} s, class {r['class']}, {extra}")
