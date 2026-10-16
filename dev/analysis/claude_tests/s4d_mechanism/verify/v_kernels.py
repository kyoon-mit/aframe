"""Independent kernel checks: (1) clean S4D ck_69 layer-1 kernels from raw parameters, modes and units;
(2) per-channel match against real harness clean windows; (3) d512 pieces' group delay vs Newtonian tau(f)."""
import json, math, sys, resource
import numpy as np, torch
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
sys.path.insert(0, f"{S}/harness")
import harness
torch.set_num_threads(8)
TS = 4.925490947e-6
def tau(f, mc): return 5 / 256 * (TS * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)
out = {}
# ---- (1) clean S4D layer 1
ck = torch.load(f"{S}/verify/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_69.ckpt", map_location="cpu", weights_only=False)
sd = ck["state_dict"]
p = lambda k: sd[f"model.net.s4_layers.0.{k}"].double().numpy()
dt = np.exp(p("kernel.log_dt"))[:, None]; a = np.exp(p("kernel.log_A_real")); b = p("kernel.A_imag")
C = p("kernel.C"); C = C[..., 0] + 1j * C[..., 1]
A = -a + 1j * b; dtA = A * dt
fs, L = 2048.0, 8192
coef = 2 * C * (np.exp(dtA) - 1) / A
l = np.arange(L)
K = np.real(np.einsum("hn,hnl->hl", coef, np.exp(dtA[..., None] * l)))          # ml4gw S4DKernel: real part
net = harness.load_net("sig_only_reg_s4d_max_dropout0_64x32x4", f"{S}/verify/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__ck_69.ckpt", device="cpu")
lay = net.arch.net.s4_layers[0]
with torch.no_grad():
    Kmod = lay.kernel(L).double().numpy()
freq = np.angle(np.exp(dtA)) * fs / (2 * np.pi)          # wraps aliased modes into (-fs/2, fs/2]
ring = 1 / (a * dt * fs)
inb = (np.abs(freq) >= 20) & (np.abs(freq) < fs / 2)
out["clean_s4d"] = dict(modes_shape=list(a.shape), kernel_vs_module_max_abs=float(np.abs(K - Kmod).max()), kernel_max_abs=float(np.abs(Kmod).max()),
                        n_inband=int(inb.sum()), ring_inband_median_ms=float(1e3 * np.median(ring[inb])),
                        ring_inband_p95_s=float(np.percentile(ring[inb], 95)), n_aliased=int((np.abs(b * dt[:, 0:1]) > np.pi).sum()),
                        dt_range=[float(dt.min()), float(dt.max())])
# ---- (2) match of each real kernel against real clean windows (phase-maximized envelope peak / (||k|| ||x||))
X = harness.read_rows(f"{S}/harness/data/clean_X_f32.npy", 0, 1000).astype(np.float64)       # (1000, 2, 8192)
mc = np.load(f"{S}/harness/data/clean_chirp_mass.npy")[:1000]
n2 = 2 * L
h = np.zeros(n2); h[0] = 1; h[1:L] = 2; h[L] = 1
Kf = torch.tensor(np.fft.fft(K / np.linalg.norm(K, axis=1, keepdims=True), n=n2))
KDn = K.copy(); KDn[:, 0] += p("D"); KDn /= np.linalg.norm(KDn, axis=1, keepdims=True)
KDf = torch.tensor(np.fft.fft(KDn, n=n2))
best = np.zeros(64); bestD = np.zeros(64); best_mc = np.zeros(64)
for i0 in range(0, 1000, 25):
    x = X[i0:i0 + 25].reshape(-1, L)
    xn = np.linalg.norm(x, axis=1)
    Xa = torch.tensor(np.fft.fft(x, n=n2) * h)
    for KK, arr in ((Kf, best), (KDf, bestD)):
        y = torch.fft.ifft(Xa[:, None, :] * KK[None], n=n2)[..., :L]        # (50, 64, L)
        m = (y.abs().amax(-1).numpy() / xn[:, None])                         # match per (window-detector, channel)
        j = m.argmax(0)
        upd = m.max(0) > arr
        if arr is best:
            best_mc[upd] = np.repeat(mc[i0:i0 + 25], 2)[j[upd]]
        arr[:] = np.maximum(arr, m.max(0))
out["clean_s4d_match_vs_real_windows"] = dict(n_windows=1000, detectors=2, median=float(np.median(best)), p90=float(np.percentile(best, 90)),
                                              max=float(best.max()), with_D_median=float(np.median(bestD)), with_D_max=float(bestD.max()))
# self-check: matched real kernel of one window gives 1
x0 = X[0, 0]; k0 = x0[::-1] / np.linalg.norm(x0)
y0 = np.fft.ifft(np.fft.fft(x0, n2) * h * np.fft.fft(k0, n2))[:L]
out["self_match_check"] = float(np.abs(y0).max() / np.linalg.norm(x0))
del X
# ---- (3) d512 top pieces: group delay against Newtonian tau(f) at the table's best chirp mass
tab = json.load(open(f"{S}/kernels/noisy_s4d_d512_table.json"))["channels"]
pieces = sorted([c for c in tab if c["class"] == "delay-line piece"], key=lambda c: -c["best_inband_match"])[:8]
ck2 = torch.load(f"{S}/verify/ckpt/s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96__ck_288.ckpt", map_location="cpu", weights_only=False)
sd2 = ck2["state_dict"]
q = lambda k: sd2[f"model.kernel.{k}"].double().numpy()
dt2 = np.exp(q("log_dt")); a2 = np.exp(q("log_A_real")); b2 = q("A_imag"); C2 = q("C"); C2 = C2[..., 0] + 1j * C2[..., 1]
fs2, L2 = 1024.0, 4096
l2 = np.arange(L2)
nf = 4 * L2
fr = np.fft.fftfreq(nf, 1 / fs2)
res = []
for c in pieces:
    hh = c["channel"]
    A2 = -a2[hh] + 1j * b2[hh]; dA = A2 * dt2[hh]
    k = 2 * np.einsum("n,nl->l", C2[hh] * (np.exp(dA) - 1) / A2, np.exp(dA[:, None] * l2))     # ComplexS4DKernel
    Hk = np.fft.fft(k, nf); Hl = np.fft.fft(k * l2, nf)
    gd = np.real(Hl / Hk) / fs2
    P = np.abs(Hk) ** 2
    sel = (fr >= c["f10"]) & (fr <= c["f90"]) & (P > 1e-3 * P[(fr >= 20) & (fr < 512)].max())
    tt = tau(fr[sel], c["best_chirp_mass"])
    W = P[sel]
    xm = (W * tt).sum() / W.sum(); ym = (W * gd[sel]).sum() / W.sum()
    slope = float((W * (tt - xm) * (gd[sel] - ym)).sum() / (W * (tt - xm) ** 2).sum())
    ring_ = 1 / (a2[hh] * dt2[hh] * fs2); f_ = np.angle(np.exp(dA)) * fs2 / (2 * np.pi)
    inm = (f_ >= c["f10"]) & (f_ <= c["f90"])
    res.append(dict(channel=hh, table_slope=c["gd_slope_vs_chirp"], my_slope_vs_newtonian=slope, offset_s=float(ym - xm),
                    best_mc=c["best_chirp_mass"], band=[c["f10"], c["f90"]], dt=float(dt2[hh]),
                    modes_in_band=int(inm.sum()), median_ring_over_tau_in_band=float(np.median(ring_[inm] / tau(np.abs(f_[inm]), c["best_chirp_mass"]))),
                    modes_in_band_over_int_tau_df=float(inm.sum() / (np.trapz(tau(np.linspace(c['f10'], c['f90'], 2000), c['best_chirp_mass']), np.linspace(c['f10'], c['f90'], 2000))))))
out["d512_pieces"] = res
out["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(out, open(f"{S}/verify/v_kernels.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
