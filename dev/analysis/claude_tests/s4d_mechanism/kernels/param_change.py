"""How far training moved each resonance from its known start values.

S4D-Lin start: log_A_real = log 0.5 and A_imag = pi n exactly (log_dt random, log-uniform in [1e-4, 10]).
Band start: log_dt = log 0.01, log_A_real = log(1 / (4 * 1024) / 0.01), A_imag = 2 pi f / 1024 / 0.01 with
f = 20 + (512 - 20) / 256 * (n + u_h), u_h one random offset per channel.
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kcommon as kc  # noqa: E402

out = {}
for name in ("noisy_s4d_d64", "noisy_s4d_d512", "noisy_s4d_d512_band", "clean_s4d_layer1"):
    sd, ck = kc.load_state(name)
    pre = "net.s4_layers.0.kernel." if name.startswith("clean") else "kernel."
    la = sd[pre + "log_A_real"].double().numpy()
    ai = sd[pre + "A_imag"].double().numpy()
    ldt = sd[pre + "log_dt"].double().numpy()
    H, N = ai.shape
    q = lambda v: {str(p): round(float(np.percentile(np.abs(v), p)), 4) for p in (50, 90, 99)} | {"max": round(float(np.abs(v).max()), 3)}
    r = {}
    if "band" not in name:
        d_la = la - math.log(0.5)
        d_ai = (ai - math.pi * np.arange(N)[None]) / math.pi
        r["abs_change_log_A_real"] = q(d_la)
        r["abs_change_A_imag_in_start_spacings"] = q(d_ai)
        r["share_modes_ring_changed_over_10pct"] = round(float((np.abs(d_la) > math.log(1.1)).mean()), 4)
        r["share_modes_A_imag_moved_over_half_spacing"] = round(float((np.abs(d_ai) > 0.5).mean()), 4)
        u = (ldt - math.log(1e-4)) / (math.log(10) - math.log(1e-4))
        r["log_dt_position_in_start_range_min_q1_median_q3_max"] = [round(float(x), 3) for x in np.percentile(u, [0, 25, 50, 75, 100])]
    else:
        d_ldt = ldt - math.log(0.01)
        d_la = la - math.log(1 / (4.0 * 1024) / 0.01)
        sp = 2 * math.pi * ((512 - 20) / 256) / 1024 / 0.01
        off = (ai - 2 * math.pi * 20 / 1024 / 0.01) / sp - np.arange(N)[None]
        r["abs_change_log_dt"] = q(d_ldt)
        r["abs_change_log_A_real"] = q(d_la)
        r["abs_change_A_imag_in_start_spacings_vs_channel_median_offset"] = q(off - np.median(off, 1, keepdims=True))
        r["share_modes_ring_changed_over_10pct"] = round(float((np.abs(d_la + d_ldt[:, None]) > math.log(1.1)).mean()), 4)
    out[name] = r
    print(name, json.dumps(r))
with open(os.path.join(kc.HERE, "param_change_from_start.json"), "w") as fh:
    json.dump(out, fh, indent=1)
