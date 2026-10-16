"""Theoretical and gradient receptive field of ResNetGridNet (aframe ResNet1D)."""

import json
import time

import numpy as np
import torch

from rf_common import (FS, N, RESNET_DIR, back_interval, block_union_check, build, centered_span, features,
                       grad_maps, load_state, main_path, nonzero_interval, shortest_span, whitened_noise)

torch.set_num_threads(4)
T_IDX = 128          # central index of the 256 last-stage time steps
SEEDS = 20
N_FEAT = 8
QS = (0.5, 0.9, 0.99)

# ---------------------------------------------------------------- theory
net = build(0)
layers = main_path(net)
r, j = 1, 1
rows = []
for name, k, s, p, d in layers:
    r += (k - 1) * d * j
    j *= s
    rows.append((name, k, s, p, r, j))
print("layer-by-layer (name, k, s, pad, r_out, j_out):")
for row in rows:
    print("  ", row)
R_TH = r
lo_th, hi_th = back_interval(layers, T_IDX, T_IDX)
lo_u, hi_u = block_union_check(net, T_IDX)
assert (lo_th, hi_th) == (lo_u, hi_u)
print(f"theory: r = {R_TH} samples = {R_TH / FS:.4f} s = {R_TH / N:.4f} of window; jump {j} samples")
print(f"central index {T_IDX}: input interval [{lo_th}, {hi_th}] (main path == union with skips), "
      f"center {(lo_th + hi_th) / 2}")
with torch.no_grad():
    n_out = features(net, torch.zeros(1, 2, N, dtype=torch.float64)).shape[-1]
for i in (0, n_out - 1):
    a, b = back_interval(layers, i, i)
    print(f"last-stage index {i} of {n_out}: interval [{a}, {b}], inside the window [{max(a, 0)}, {min(b, N - 1)}] = "
          f"{min(b, N - 1) - max(a, 0) + 1} samples")
C = (lo_th + hi_th) // 2


def summarize(tag, g2s):
    g2s = np.asarray(g2s)
    P = g2s.sum(1, keepdims=True)
    prof = (g2s / P).mean(0)
    nzi = [nonzero_interval(g) for g in g2s]
    los = np.array([a for a, _, _ in nzi])
    his = np.array([b for _, b, _ in nzi])
    spans = np.array([c for _, _, c in nzi])
    inside = (g2s[:, lo_th:hi_th + 1].sum(1) / P[:, 0])
    out = {
        "maps": len(g2s),
        "nonzero_lo_min_max": [int(los.min()), int(los.max())],
        "nonzero_hi_min_max": [int(his.min()), int(his.max())],
        "nonzero_span_min_max": [int(spans.min()), int(spans.max())],
        "maps_with_exact_theory_interval": int(((los == lo_th) & (his == hi_th)).sum()),
        "maps_with_whole_window": int(((los == 0) & (his == N - 1)).sum()),
        "mass_inside_theory_mean_min": [float(inside.mean()), float(inside.min())],
        "avg_profile_centered": {q: centered_span(prof, C, q) for q in QS},
        "avg_profile_shortest": {q: shortest_span(prof, q) for q in QS},
        "per_map_centered_mean_std": {q: [float(np.mean(v)), float(np.std(v))] for q in QS
                                      for v in [[centered_span(g, C, q) for g in g2s]]},
        "per_map_shortest_mean_std": {q: [float(np.mean(v)), float(np.std(v))] for q in QS
                                      for v in [[shortest_span(g, q) for g in g2s]]},
    }
    print(f"\n== {tag}: {out['maps']} gradient maps")
    print(f"  nonzero gradient: lo in {out['nonzero_lo_min_max']}, hi in {out['nonzero_hi_min_max']}, "
          f"span in {out['nonzero_span_min_max']}; exact theory interval {out['maps_with_exact_theory_interval']}"
          f"/{out['maps']}; whole window {out['maps_with_whole_window']}/{out['maps']}")
    print(f"  squared-gradient mass inside [{lo_th},{hi_th}]: mean {inside.mean():.6f}, min {inside.min():.6f}")
    for q in QS:
        a = out["avg_profile_centered"][q]
        b = out["avg_profile_shortest"][q]
        pm = out["per_map_centered_mean_std"][q]
        ps = out["per_map_shortest_mean_std"][q]
        print(f"  {int(q * 100):>2}%: averaged map centered {a} samples = {a / FS:.4f} s = {a / N:.4f} window; "
              f"shortest {b}; per map centered {pm[0]:.1f} +- {pm[1]:.1f}, shortest {ps[0]:.1f} +- {ps[1]:.1f}")
    return out, prof


def run(tag, state=None):
    res = {}
    profs = {}
    for frozen in (False, True):
        g2s, nact = [], []
        t0 = time.time()
        for seed in range(SEEDS):
            net = build(seed, state=state)
            x0 = whitened_noise(torch.Generator().manual_seed(1000 + seed))
            g2, pick, act, na = grad_maps(net, x0, T_IDX, N_FEAT, torch.Generator().manual_seed(2000 + seed), frozen)
            g2s.extend(g2)
            nact.append(na)
            if seed == 0 and not frozen:
                # one feature by itself must give the same map as in the batch
                x1 = x0.clone().requires_grad_(True)
                features(net, x1)[0, int(pick[0]), T_IDX].backward()
                ref = x1.grad.pow(2).sum(1)[0].numpy()
                assert np.allclose(ref, g2[0], rtol=1e-9, atol=1e-30), "batched gradient differs"
        key = f"{tag} / norm stats {'frozen' if frozen else 'live'}"
        print(f"\n[{key}] active channels at t={T_IDX}: mean {np.mean(nact):.0f} of 512, "
              f"min {np.min(nact)}; {time.time() - t0:.0f} s")
        res[key], profs[key] = summarize(key, g2s)
    return res, profs


results, profiles = {}, {}
r0, p0 = run("init (20 seeds)")
results.update(r0)
profiles.update(p0)
for ck in ("ck_244.ckpt", "last.ckpt"):
    state, epoch = load_state(RESNET_DIR + ck)
    r1, p1 = run(f"trained {ck} (epoch {epoch})", state)
    results.update(r1)
    profiles.update(p1)

for k, p in profiles.items():
    out = 1 - p[lo_th:hi_th + 1].sum()
    near = p[max(lo_th - FS, 0):lo_th].sum() + p[hi_th + 1:hi_th + 1 + FS].sum()
    print(f"{k}: outside mass {out:.4f}; share of it within 1 s of the span {near / max(out, 1e-30):.3f} "
          f"(uniform {2 * FS / (N - R_TH):.3f})")

meta = {"r_theory": R_TH, "jump": j, "interval": [lo_th, hi_th], "center": C, "t_idx": T_IDX,
        "seeds": SEEDS, "features_per_seed": N_FEAT}
with open("rf_resnet.json", "w") as f:
    json.dump({"meta": meta, "results": {k: {kk: ({str(a): b for a, b in vv.items()} if isinstance(vv, dict) else vv)
                                                 for kk, vv in v.items()} for k, v in results.items()}}, f, indent=1)
np.savez("rf_resnet_profiles.npz", **{k.replace(" ", "_").replace("/", "-"): v for k, v in profiles.items()})
print("\nsaved rf_resnet.json, rf_resnet_profiles.npz")
