"""Bank and free-kernel accuracy on spinning signals at SNR >= 16 (test A + test B)."""
import json
import numpy as np
P = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/predict"
out = {}
lab = {s: np.load(f"{P}/data/test{s}/labels.npz") for s in "AB"}
def cat(key):
    return np.concatenate([lab[s][key] for s in "AB"])
m1, m2 = cat("file_mass_1"), cat("file_mass_2")
chi = (m1 * cat("file_a_1") * np.cos(cat("file_tilt_1")) + m2 * cat("file_a_2") * np.cos(cat("file_tilt_2"))) / (m1 + m2)
q = np.minimum(m1, m2) / np.maximum(m1, m2)
snr, mc = cat("snr").astype(float), cat("chirp_mass").astype(float)
for tag, name in (("bank_mlp64k", "bank__mlp1x1024_d0.1"), ("free_mlp64k", "free__mlp1x1024_d0.1"), ("free_own", "free__own")):
    est = np.concatenate([np.load(f"{P}/results/est_test{s}__{name}.npy") for s in "AB"])
    ok = np.abs(est - mc) / mc < 0.01
    hi = snr >= 16
    a = hi & (chi < -0.05)
    b = hi & (np.abs(q - 0.8) < 0.1) & (np.abs(chi) < 0.03)
    out[tag] = {"n_spin_neg": int(a.sum()), "w1_spin_neg": float(ok[a].mean()),
                "median_signed_err_spin_neg": float(np.median(((est - mc) / mc)[a])),
                "n_template_like": int(b.sum()), "w1_template_like": float(ok[b].mean())}
print(json.dumps(out, indent=1))
json.dump(out, open("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/writeup_checks/spin_subset.json", "w"), indent=1)
