"""Post hoc check of the T2 interpretation: does the clean S4D need the global segment order, or only the first
segment at the window start? Edge-free shuffles (internal 64-sample ramps only), own seeds, 4000 clean windows."""
import json, sys, resource
import numpy as np, torch
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
sys.path.insert(0, f"{S}/harness")
import harness
torch.set_num_threads(8)
XP = f"{S}/harness/data/clean_X_f32.npy"
mc = np.load(f"{S}/harness/data/clean_chirp_mass.npy").astype(np.float64)
n, L, SEG, R = 4000, 8192, 1024, 64
ramp = 0.5 * (1 - np.cos(np.pi * (np.arange(R) + 0.5) / R))
tap = np.ones(L)
for b in range(SEG, L, SEG):
    tap[b - R:b] *= ramp[::-1]; tap[b:b + R] *= ramp
rng = np.random.default_rng(4242)
def perm(fixed):
    free = [i for i in range(8) if i not in fixed]
    while True:
        p = list(range(8)); q = list(rng.permutation(free))
        for i, j in zip(free, q): p[i] = j
        if p != list(range(8)): return p
PERMS = {"all8": [perm([]) for _ in range(n)], "keep_first": [perm([0]) for _ in range(n)],
         "keep_first_last": [perm([0, 7]) for _ in range(n)], "keep_last": [perm([7]) for _ in range(n)]}
def shuffled(X, ps):
    Y = np.empty_like(X)
    for i, p in enumerate(ps):
        segs = X[i].reshape(2, 8, SEG)
        Y[i] = segs[:, p, :].reshape(2, L)
    return (Y * tap).astype(np.float32)
nets = [("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/harness/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_69.ckpt", "ck_69"),
        ("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/verify/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_135.ckpt", "ck_135"),
        ("sig_only_reg_free_kernel", f"{S}/harness/ckpt/sig_only_reg_free_kernel__ck_46.ckpt", "ck_46")]
out = {}
for name, path, tag in nets:
    net = harness.load_net(name, path, device="cuda:0", gpu_gb=3.0)
    est = {k: np.empty(n) for k in ["internal_ramps"] + list(PERMS)}
    for s0 in range(0, n, 400):
        X = harness.read_rows(XP, s0, s0 + 400)
        est["internal_ramps"][s0:s0 + 400] = net((X * tap).astype(np.float32))
        for k, ps in PERMS.items():
            est[k][s0:s0 + 400] = net(shuffled(X, ps[s0:s0 + 400]))
    r = {k: {"w1": float((np.abs(e - mc) / mc < 0.01).mean()), "w5": float((np.abs(e - mc) / mc < 0.05).mean())} for k, e in est.items()}
    out[tag] = r
    print(tag, {k: (v["w1"], v["w5"]) for k, v in r.items()}, "rss %.2f GB" % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6), flush=True)
    del net; torch.cuda.empty_cache()
json.dump(out, open(f"{S}/verify/v_shuffle.json", "w"), indent=1)
