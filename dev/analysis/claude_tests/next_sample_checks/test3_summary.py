"""Summary of test3.npz, and the ideal detection on test2.npz's own windows.

usage: test3_summary.py <test3.npz> <test2.npz>
"""

import sys

import numpy as np
from scipy.stats import norm

FAP = 1e-3
BINS = [(3.6, 4), (4, 4.5), (4.5, 5), (5, 6), (6, 8), (8, 12), (12, 100)]


def main():
    z, t = (np.load(p) for p in sys.argv[1:3])
    s, size = z["snr"], int(z["samples_per_window"])
    zc = norm.isf(FAP)
    print(f"windows {len(s)}, samples per window {size}, noise NLL per sample "
          f"{float(z['noise_nll_per_sample']):.5f} (exact {0.5 * np.log(2 * np.pi * np.e):.5f})")
    print("SNR in window / aframe label: min {:.4f} median {:.4f} max {:.4f}".format(
        *np.quantile(s / z["snr_label"], [0, 0.5, 1])))
    thr = np.quantile(z["mf_noise"], 1 - FAP)
    thr_d = np.quantile(z["draw_noise"], 1 - FAP)
    print(f"thresholds at {FAP}: matched filter {thr:.3f} (exact {zc:.3f}), "
          f"random draw {thr_d:.3f} (exact {np.sqrt(2) * zc:.3f})")
    print("SNR | windows | peak sample | signal share of power | separation "
          "| found (exact) | amplitude error (1/SNR) | found, random draw "
          "| energy recovered without template (exact)")
    for lo, hi in BINS:
        k = (s >= lo) & (s < hi)
        mfn, mfs = z["mf_noise"][k], z["mf_signal"][k]
        dn, ds = z["draw_noise"][k], z["draw_signal"][k]
        print(f"{lo} to {hi} | {k.sum()} | {np.median(z['peak'][k]):.3f} "
              f"| {np.median(z['sig_frac'][k]):.2e} "
              f"| {(mfs.mean() - mfn.mean()) / mfn.std():.3f} "
              f"| {(mfs > thr).mean():.4f} ({norm.sf(zc - s[k]).mean():.4f}) "
              f"| {(z['amp'][k] - 1).std():.4f} ({np.mean(1 / s[k]):.4f}) "
              f"| {(ds > thr_d).mean():.4f} "
              f"| {np.median(z['shrink'][k]):.2e} "
              f"({np.median(s[k] ** 2 / (s[k] ** 2 + size)):.2e})")
    print(f"all | found {(z['mf_signal'] > thr).mean():.4f} "
          f"(exact {norm.sf(zc - s).mean():.4f}), random draw "
          f"{(z['draw_signal'] > thr_d).mean():.4f}")

    clean = t["maxabs_noise"].max(1) <= float(t["loud"])
    h = t["h_norm"]

    def found(sig, noi, k):
        return (sig[k] > np.quantile(noi[clean], 1 - FAP)).mean()

    gs, gn = t["gauss_signal"].sum(1), t["gauss_noise"].sum(1)
    print(f"test2 clean windows {clean.sum()}, found at {FAP}:")
    for name, k in [("all", clean),
                    ("label 4 to 5", clean & (t["snr"] >= 4) & (t["snr"] < 5))]:
        print(f"  {name}: ideal {norm.sf(zc - h[k]).mean():.4f}, "
              f"Gaussian {found(gs, gn, k):.4f}, "
              f"matched filter {found(t['mf_signal'], t['mf_noise'], k):.4f}")


if __name__ == "__main__":
    main()
