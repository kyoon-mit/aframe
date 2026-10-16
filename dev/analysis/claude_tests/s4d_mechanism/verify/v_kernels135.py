"""Clean S4D layer-1 check on the current best checkpoint ck_135 (copied): modes, ring times, match vs real windows."""
import json, sys, resource
import numpy as np, torch
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
sys.path.insert(0, f"{S}/harness")
import harness
torch.set_num_threads(8)
out = {}
for tag in ("ck_69", "ck_135"):
    sd = torch.load(f"{S}/verify/ckpt/sig_only_reg_s4d_max_dropout0_64x32x4__{tag}.ckpt", map_location="cpu", weights_only=False)["state_dict"]
    p = lambda k: sd[f"model.net.s4_layers.0.{k}"].double().numpy()
    dt = np.exp(p("kernel.log_dt"))[:, None]; a = np.exp(p("kernel.log_A_real")); b = p("kernel.A_imag")
    C = p("kernel.C"); C = C[..., 0] + 1j * C[..., 1]
    A = -a + 1j * b; dtA = A * dt; fs, L = 2048.0, 8192
    K = np.real(np.einsum("hn,hnl->hl", 2 * C * (np.exp(dtA) - 1) / A, np.exp(dtA[..., None] * np.arange(L))))
    D = p("D")
    freq = np.angle(np.exp(dtA)) * fs / (2 * np.pi); ring = 1 / (a * dt * fs)
    inb = (np.abs(freq) >= 20) & (np.abs(freq) < fs / 2)
    X = harness.read_rows(f"{S}/harness/data/clean_X_f32.npy", 0, 1000).astype(np.float64)
    n2 = 2 * L; h = np.zeros(n2); h[0] = 1; h[1:L] = 2; h[L] = 1
    Kf = torch.tensor(np.fft.fft(K / np.linalg.norm(K, axis=1, keepdims=True), n=n2))
    best = np.zeros(64)
    for i0 in range(0, 1000, 25):
        x = X[i0:i0 + 25].reshape(-1, L); xn = np.linalg.norm(x, axis=1)
        Xa = torch.tensor(np.fft.fft(x, n=n2) * h)
        y = torch.fft.ifft(Xa[:, None, :] * Kf[None], n=n2)[..., :L]
        best = np.maximum(best, (y.abs().amax(-1).numpy() / xn[:, None]).max(0))
    P = np.abs(np.fft.rfft(K, 4 * L)) ** 2; f = np.fft.rfftfreq(4 * L, 1 / fs)
    c = np.cumsum(P, 1) / P.sum(1, keepdims=True)
    f10 = f[np.argmax(c >= 0.1, 1)]; f90 = f[np.argmax(c >= 0.9, 1)]
    out[tag] = dict(n_inband=int(inb.sum()), ring_inband_median_ms=float(1e3 * np.median(ring[inb])), ring_inband_p95_s=float(np.percentile(ring[inb], 95)),
                    match_real_windows_median=float(np.median(best)), match_real_windows_max=float(best.max()),
                    D2_over_K2_median=float(np.median(D ** 2 / (K ** 2).sum(1))), band10_90_median_Hz=[float(np.median(f10)), float(np.median(f90))])
    print(tag, out[tag], flush=True)
out["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
json.dump(out, open(f"{S}/verify/v_kernels135.json", "w"), indent=1)
print(out["maxrss_GB"])
