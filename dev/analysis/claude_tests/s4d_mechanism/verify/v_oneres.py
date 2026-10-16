"""Best single real resonance e^{-t/T} cos(2 pi f0 t) (phase-maximized, max over time) against the same 1000 real
clean windows x 2 detectors used for the clean S4D layer-1 channel matches; apples-to-apples one-resonance bound."""
import json, sys, numpy as np, torch
S = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath"
sys.path.insert(0, f"{S}/harness")
import harness
torch.set_num_threads(8)
torch.cuda.set_per_process_memory_fraction(3.0 * 2**30 / torch.cuda.get_device_properties(0).total_memory, 0)
dev = torch.device("cuda:0")
fs, L = 2048.0, 8192; n2 = 2 * L
t = np.arange(L) / fs
F0 = np.array([40, 60, 80, 100, 130, 170, 220, 300, 400, 550, 750])
T = np.geomspace(0.002, 2.0, 16)
ks = []
for f0 in F0:
    for TT in T:
        k = np.exp(-t / TT) * np.cos(2 * np.pi * f0 * t)
        ks.append(k / np.linalg.norm(k))
ks = np.array(ks)
Kf = torch.tensor(np.fft.fft(ks, n=n2), dtype=torch.complex64, device=dev)
h = torch.zeros(n2, device=dev); h[0] = 1; h[1:L] = 2; h[L] = 1
X = harness.read_rows(f"{S}/harness/data/clean_X_f32.npy", 0, 1000).astype(np.float64)
best = np.zeros(len(ks))
for i0 in range(0, 1000, 10):
    x = X[i0:i0 + 10].reshape(-1, L)
    xn = torch.tensor(np.linalg.norm(x, axis=1), device=dev)
    Xa = torch.fft.fft(torch.tensor(x, device=dev), n=n2) * h
    Xa = Xa.to(torch.complex64)
    for j0 in range(0, len(ks), 44):
        y = torch.fft.ifft(Xa[:, None, :] * Kf[None, j0:j0 + 44], n=n2)[..., :L]
        m = (y.abs().amax(-1) / xn[:, None]).max(0).values.cpu().numpy()
        best[j0:j0 + 44] = np.maximum(best[j0:j0 + 44], m)
B = best.reshape(len(F0), len(T))
out = {"f0": F0.tolist(), "ring_s": T.tolist(), "best_over_ring_per_f0": B.max(1).tolist(),
       "best_ring_per_f0": T[B.argmax(1)].tolist(), "overall_max": float(B.max())}
json.dump(out, open(f"{S}/verify/v_oneres.json", "w"), indent=1)
print(json.dumps(out, indent=1))
