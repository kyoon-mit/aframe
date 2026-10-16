"""All-pass check (fixed): N resonators (conjugate pole pairs) with mirrored RHP zeros turn the phase by 2 pi N over positive freqs;
the same filter written as D + partial fractions (an S4D-form channel) is identical."""
import json, numpy as np
N = 6; a = np.full(N, 5.0); f = np.linspace(50, 300, N)
p = np.concatenate([-a + 2j * np.pi * f, -a - 2j * np.pi * f])
w = np.linspace(0, 2 * np.pi * 2e5, 4000001)
s = 1j * w
Hs = np.ones_like(s)
for pp in p:
    Hs = Hs * (s + np.conj(pp)) / (s - pp)          # zero at -conj(p): mirror image in the right half plane
th = np.unwrap(np.angle(Hs))
# partial fractions: H = 1 + sum r_k/(s - p_k), r_k = prod_j (p_k + conj p_j) / prod_{j!=k} (p_k - p_j)
r = []
for k, pk in enumerate(p):
    num = np.prod([pk + np.conj(pj) for pj in p]); den = np.prod([pk - pj for j, pj in enumerate(p) if j != k])
    r.append(num / den)
Hpf = 1 + sum(rk / (s - pk) for rk, pk in zip(r, p))
out = dict(turn_over_2piN=float((th[0] - th[-1]) / (2 * np.pi * N)), absH_dev=float(np.abs(np.abs(Hs) - 1).max()),
           partial_fraction_vs_product=float(np.abs(Hpf - Hs).max()), residues_conjugate_pairs=float(np.abs(np.array(r[:N]) - np.conj(r[N:])).max()))
print(out); json.dump(out, open("v_allpass.json", "w"), indent=1)
