"""Tables of the 23 October update from test1.npz, test2.npz and the
test2_spread.py output.

usage: summarize.py <test1.npz> <test2.npz> <spread.npz>
"""

import sys

import numpy as np

N = 8192
LOUD = 8.0
IFOS = ("H1", "L1")


def test1(path):
    d = np.load(path)
    orders = list(d["orders"])
    print("== Test 1: mean NLL per sample (nats)")
    for split in ("O3a_val", "O3b"):
        mx = d[f"{split}_maxabs"]
        for i, ifo in enumerate(IFOS):
            for label, m in (("clean", mx[:, i] <= LOUD), ("loud", mx[:, i] > LOUD)):
                net = d[f"{split}_network"][m, i] / N
                row = f"{split} {ifo} {label:5s} n={m.sum():6d} network {net.mean():.4f}"
                for name in (f"typical_{ifo}", f"mean_{ifo}", "typical_pooled"):
                    g = d[f"{split}_{name}_p{N - 1}"][m, i] / N
                    row += f" | {name} {g.mean():.4f} diff {np.mean(net - g):+.4f} better {100 * np.mean(net < g):.1f}%"
                print(row)
    print("\n== Test 1: order sweep, clean windows, own-detector typical spectrum")
    for split in ("O3a_val", "O3b"):
        mx = d[f"{split}_maxabs"]
        for i, ifo in enumerate(IFOS):
            m = mx[:, i] <= LOUD
            print(f"{split} {ifo}", " ".join(
                f"p{p} {d[f'{split}_typical_{ifo}_p{p}'][m, i].mean() / N:.4f}" for p in orders))
    print("\n== Test 1: O3b per file, clean, network minus typical (own detector)")
    fi, mx = d["O3b_file"], d["O3b_maxabs"]
    for k, f in enumerate(d["O3b_files"]):
        row = f.split("/")[-1]
        for i, ifo in enumerate(IFOS):
            m = (fi == k) & (mx[:, i] <= LOUD)
            net = d["O3b_network"][m, i] / N
            g = d[f"O3b_typical_{ifo}_p{N - 1}"][m, i] / N
            row += f" | {ifo} n={m.sum()} {net.mean():.4f} vs {g.mean():.4f} ({np.mean(net - g):+.4f})"
        print(row)
    print("\n== Test 1: network minus typical by position, clean windows")
    segs = ((0, 1), (1, 16), (16, 128), (128, 1024), (1024, 4096), (4096, 8192))
    for split in ("O3a_val", "O3b"):
        for i, ifo in enumerate(IFOS):
            diff = d[f"{split}_profile_network"][i] - d[f"{split}_profile_typical_{ifo}"][i]
            print(f"{split} {ifo}", " ".join(f"[{a},{b}) {diff[a:b].mean():+.4f}" for a, b in segs))
    print("\nlog det checks (Cholesky, Levinson):", {
        k.removeprefix("logdet_"): tuple(np.round(d[k], 6)) for k in d if k.startswith("logdet_")})


def test2(path):
    d = np.load(path)
    snr = d["snr"]
    loud = (d["maxabs_noise"] > LOUD).any(1)
    stats = {
        "matched filter": (d["mf_noise"], d["mf_signal"]),
        "network": (d["net_noise"].sum(1), d["net_signal"].sum(1)),
        "Gaussian": (d["gauss_noise"].sum(1), d["gauss_signal"].sum(1)),
    }
    print(f"\n== Test 2: {len(snr)} windows, loud {loud.sum()}; SNR quantiles",
          np.quantile(snr, [0, 0.25, 0.5, 0.75, 1]).round(2),
          "; whitened |h| / SNR median", np.median(d["h_norm"] / snr).round(4))
    bins = ((4, 5), (5, 6), (6, 8), (8, 100))
    for label, keep in (("all windows", np.ones(len(snr), bool)), ("clean windows", ~loud)):
        print(f"-- {label}: fraction detected")
        for fap in (1e-1, 1e-2, 1e-3):
            thr = {k: np.quantile(nz[keep], 1 - fap) for k, (nz, _) in stats.items()}
            print(f"FAP {fap:g}: " + " ".join(
                f"{k} {(sg[keep] > thr[k]).mean():.4f} (threshold {thr[k]:.2f})" for k, (_, sg) in stats.items()))
            for a, b in bins:
                m = keep & (snr >= a) & (snr < b)
                print(f"   SNR [{a},{b}) n={m.sum()}: " + " ".join(
                    f"{k} {(sg[m] > thr[k]).mean():.4f}" for k, (_, sg) in stats.items()))
    for k, (nz, _) in stats.items():
        thr = np.quantile(nz[~loud], 1 - 1e-3)
        print(f"{k}: loud noise windows above the clean 1e-3 threshold {(nz[loud] > thr).sum()} of {loud.sum()}")


def spread(path):
    d = np.load(path)
    lr, hn = d["lr"], d["hnorm"]
    print("\n== Test 2 diagnosis: (LR + |h|^2/2) / |h| on noise windows, by detector SNR |h|")
    for i, ifo in enumerate(IFOS):
        z = (lr[:, i] + hn[:, i] ** 2 / 2) / hn[:, i]
        for a, b in ((2, 4), (4, 8), (8, 16), (16, 100)):
            m = (hn[:, i] >= a) & (hn[:, i] < b)
            print(f"{ifo} |h| [{a},{b}) n={m.sum()}: mean {z[m].mean():.2f} std {z[m].std():.2f}")


if __name__ == "__main__":
    test1(sys.argv[1])
    test2(sys.argv[2])
    spread(sys.argv[3])
