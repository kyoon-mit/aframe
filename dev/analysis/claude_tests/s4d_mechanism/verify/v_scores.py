"""Recompute within-1/2/5/10% from the harness's saved estimates and labels (no harness import)."""
import glob, json, os, numpy as np
D = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/harness/data"
out = {}
for s in ("clean", "noisy"):
    mc = np.load(f"{D}/{s}_chirp_mass.npy").astype(np.float64)
    lab = np.load(f"{D}/{s}_labels.npz")
    assert np.array_equal(lab["chirp_mass"], np.load(f"{D}/{s}_chirp_mass.npy"))
    snr = np.load(f"{D}/{s}_snr.npy")
    for f in sorted(glob.glob(f"{D}/est_{s}__*.npy")):
        e = np.load(f).astype(np.float64)
        err = np.abs(e - mc) / mc
        r = {t: round(100 * float((err < t / 100).mean()), 3) for t in (1, 2, 5, 10)}
        r4000 = round(100 * float((err[:4000] < 0.01).mean()), 3)
        key = os.path.basename(f)[4:-4]
        out[key] = dict(all=r, first4000_w1=r4000, n=len(e))
        print(key, r, "first4000:", r4000)
    # SNR population check
    p20 = float((snr > 20).mean()); p8 = float((snr < 8).mean()); p16 = float((snr >= 16).mean())
    th20 = (20.0**-2 - 100.0**-2) / (4.0**-2 - 100.0**-2)
    th8 = (4.0**-2 - 8.0**-2) / (4.0**-2 - 100.0**-2)
    th16 = (16.0**-2 - 100.0**-2) / (4.0**-2 - 100.0**-2)
    n = len(snr)
    z = lambda p, th: (p - th) / np.sqrt(th * (1 - th) / n)
    out[f"{s}_snr"] = dict(n=n, gt20=p20, th20=th20, z20=z(p20, th20), lt8=p8, th8=th8, z8=z(p8, th8), ge16=p16, th16=th16, z16=z(p16, th16),
                          min=float(snr.min()), max=float(snr.max()), median=float(np.median(snr)))
    print(s, out[f"{s}_snr"])
json.dump(out, open("v_scores.json", "w"), indent=1)
