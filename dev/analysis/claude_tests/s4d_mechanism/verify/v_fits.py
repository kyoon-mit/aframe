"""Independent recheck of FITS: (a) clipping growing pencil poles (no fit of poles) and tone_check's reflection,
amplitudes by own least squares on column-normalized modes; (b) the best float32-safe p=32 fits re-evaluated in complex64
with own code; (c) amplitude-cancellation ratio of those fits."""
import json, glob, numpy as np, torch
F = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/fits"
Kz = np.load(f"{F}/kernels.npz"); Pz = np.load(f"{F}/pencil_poles.npz")
L = 8064; lag = np.arange(L, dtype=np.float64)
def fit(logz, K):
    V = np.exp(lag[:, None] * logz[None, :]); s = np.linalg.norm(V, axis=0); Vn = V / s
    c, *_ = np.linalg.lstsq(Vn, K, rcond=None); Kh = Vn @ c
    return float(np.abs(np.vdot(Kh, K)) / (np.linalg.norm(Kh) * np.linalg.norm(K))), c / s
def logz(z): return np.log(np.abs(z)) + 1j * np.angle(z)
out = {"clip": {}, "reflect": {}, "p32": {}}
for mc in ("1.0", "1.2", "1.5", "2.0"):
    K = Kz[f"K_{mc}"]
    for N in (256, 512, 2048):
        z = Pz[f"z_{mc}_{N}"]
        zc = np.where(np.abs(z) > 1, z / np.abs(z), z)
        zr = np.where(np.abs(z) > 1, 1 / np.conj(z), z)
        out["clip"][f"{mc}_{N}"] = fit(logz(zc), K)[0]
        out["reflect"][f"{mc}_{N}"] = fit(logz(zr), K)[0]
    best = None
    for f in glob.glob(f"{F}/restarts/*_Mc{mc}_p32.bestf32.npz"):
        d = np.load(f); a, b = d["alpha"], d["beta"]
        m64, c = fit(-a + 1j * b, K)
        s = torch.tensor(-a + 1j * b, dtype=torch.complex64)
        V = torch.exp(torch.arange(L, dtype=torch.float32)[:, None] * s[None, :])
        Kh = (V @ torch.tensor(c, dtype=torch.complex64)).numpy().astype(np.complex128)
        m32 = float(np.abs(np.vdot(Kh, K)) / (np.linalg.norm(Kh) * np.linalg.norm(K)))
        Vd = np.exp(lag[:, None] * (-a + 1j * b)[None, :])
        cancel = float(np.sum((np.abs(c) * np.linalg.norm(Vd, axis=0)) ** 2) / np.linalg.norm(Vd @ c) ** 2)
        if best is None or (m32 > best["m32"] and cancel < 100):
            best = dict(file=f.split("/")[-1], m64=m64, m32=m32, cancel=cancel, n_decay_zero=int((a <= 1e-12).sum()))
    out["p32"][mc] = best
json.dump(out, open(f"{F}/../verify/v_fits.json", "w"), indent=1)
print(json.dumps(out, indent=1))
