"""Independent re-test of the post hoc window-start finding and the edge-free T3, own code and seeds.
Clean S4D ck_69, ck_81 (analyzed) and ck_135 (current best, copied to verify/ckpt), free kernel ck_46.
Windows are processed in chunks of 400 so resident memory stays well under 4 GB."""
import json, os, sys, resource
import numpy as np, torch
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
sys.path.insert(0, f"{S}/harness")
import harness
torch.set_num_threads(8)
XP = f"{S}/harness/data/clean_X_f32.npy"
mc = np.load(f"{S}/harness/data/clean_chirp_mass.npy").astype(np.float64)
n, I, L = 4000, 2, 8192
R = 64
ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(R) + 0.5) / R))
CH = 400

def analytic(x):
    F = np.fft.fft(x, n=2 * L, axis=-1)
    h = np.zeros(2 * L); h[0] = 1; h[1:L] = 2; h[L] = 1
    return np.fft.ifft(F * h, axis=-1)[..., :L]

def internal_taper(seg):
    t = np.ones(L)
    for b in range(seg, L, seg):
        t[b - R:b] *= ramp[::-1]; t[b:b + R] *= ramp
    return t

rng = np.random.default_rng(777)
PH = {seg: rng.uniform(0, 2 * np.pi, size=(n, L // seg)) for seg in (1024, 512)}
TAP = {seg: internal_taper(seg) for seg in (1024, 512)}

def variant(k, X, sl):
    v = X.copy()
    if k == "clean": return v
    if k == "ramp_start": v[..., :R] *= ramp; return v
    if k == "zero_first_64": v[..., :R] = 0; return v
    if k == "ramp_end": v[..., -R:] *= ramp[::-1]; return v
    if k == "notch_mid":
        c0 = 4096; v[..., c0 - R:c0] *= ramp[::-1]; v[..., c0:c0 + R] *= ramp; return v
    if k == "shift_left_64": return np.concatenate([X[..., R:], np.zeros(X.shape[:-1] + (R,), np.float32)], -1)
    seg = int(k.split("seg")[1])
    if k.startswith("internal_ramps"): return (X * TAP[seg]).astype(np.float32)
    if k.startswith("T3_edgefree"):
        xa = analytic(X.astype(np.float64))
        rot = np.repeat(np.exp(1j * PH[seg][sl]), seg, axis=1)[:, None, :]
        return (TAP[seg] * np.real(xa * rot)).astype(np.float32)
    raise KeyError(k)

KEYS = ["clean", "ramp_start", "zero_first_64", "ramp_end", "notch_mid", "shift_left_64",
        "internal_ramps_seg1024", "T3_edgefree_seg1024", "internal_ramps_seg512", "T3_edgefree_seg512"]
X8 = harness.read_rows(XP, 0, 8)
chk = float(np.abs(TAP[1024] * np.real(analytic(X8.astype(np.float64))) - TAP[1024] * X8).max())

nets = [("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/harness/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_69.ckpt", "ck_69"),
        ("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/harness/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_81.ckpt", "ck_81"),
        ("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/verify/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_135.ckpt", "ck_135"),
        ("sig_only_reg_free_kernel", f"{S}/harness/ckpt/sig_only_reg_free_kernel__ck_46.ckpt", "ck_46")]
out = {"analytic_zero_phase_check_max_abs": chk, "n": n}
for name, path, tag in nets:
    net = harness.load_net(name, path, device="cuda:0", gpu_gb=3.0)
    est = {k: np.empty(n) for k in KEYS}
    for s0 in range(0, n, CH):
        sl = slice(s0, s0 + CH)
        X = harness.read_rows(XP, s0, s0 + CH)
        for k in KEYS:
            est[k][sl] = net(variant(k, X, sl))
    np.savez(f"{S}/verify/v_onset_est_{tag}.npz", **est)
    r = {}
    for k in KEYS:
        err = (est[k] - mc) / mc
        r[k] = {"w1": float((np.abs(err) < 0.01).mean()), "w5": float((np.abs(err) < 0.05).mean()),
                "med_err_mc_lt_1.5": float(np.median(err[mc < 1.5])), "med_err_mc_gt_2.2": float(np.median(err[mc > 2.2]))}
    saved = f"{S}/harness/data/est_clean__{name}__{tag}.npy"
    if os.path.exists(saved):
        r["flips_vs_saved_f16"] = int(((np.abs(est['clean'] - mc) / mc < 0.01) != (np.abs(np.load(saved) - mc) / mc < 0.01)).sum())
    out[f"{name}:{tag}"] = r
    print(tag, {k: (v["w1"], v["w5"]) if isinstance(v, dict) else v for k, v in r.items()}, flush=True)
    print("   rss GB %.2f" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)
    del net; torch.cuda.empty_cache()
out["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(out, open(f"{S}/verify/v_onset.json", "w"), indent=1)
print("rss GB", out["maxrss_GB"], "zero-phase check", chk)
