import numpy as np, pandas as pd, sys
for path in sys.argv[1:]:
    print("=====", path)
    d = pd.read_csv(path + "/param_est_results.csv")
    b = pd.read_csv(path + "/param_est_results_bkg.csv")
    print(d.columns.tolist(), len(d), len(b))
    snr = d.snr.values
    print("snr min max median", snr.min(), snr.max(), np.median(snr), "frac<8", (snr<8).mean(), (snr<8).sum())
    for v in ["chirp_mass", "mass_ratio"]:
        if f"{v}_true" not in d: continue
        t, p, s = d[f"{v}_true"].values, d[f"{v}_pred"].values, d[f"sigma_{v}_pred"].values
        z = (p - t)/(s+1e-12)
        print(v, "z mean %.3f std %.4f" % (z.mean(), z.std()), "median|z| %.3f (0.674 Gauss)" % np.median(abs(z)),
              "frac|z|<1 %.3f (0.683)" % (abs(z)<1).mean(), "frac|z|<2 %.3f (0.954)" % (abs(z)<2).mean())
        print("  mean|e|/mean s %.3f (0.798)" % (abs(p-t).mean()/s.mean()))
        print("  true: mean %.4f std %.4f min %.4f max %.4f" % (t.mean(), t.std(), t.min(), t.max()))
        print("  pred: mean %.4f" % p.mean(), "bkg sigma median %.4f" % b[f"sigma_{v}_pred"].median(), "bkg pred mean %.4f" % b[f"{v}_pred"].mean())
        rel = abs(p - t)/t
        bins = [(4,6),(6,8),(8,12),(12,16),(16,25),(25,50)]
        for lo, hi in bins:
            m = (snr>=lo)&(snr<hi)
            print("  %d-%d n=%d w1 %.3f w2 %.3f w5 %.3f w10 %.3f medsig %.3f zstd %.3f medz %.3f" % (lo,hi,m.sum(),(rel[m]<.01).mean(),(rel[m]<.02).mean(),(rel[m]<.05).mean(),(rel[m]<.10).mean(), np.median(s[m]), z[m].std(), np.median(abs(z[m]))))
        print("  all: w1 %.3f w2 %.3f w5 %.3f w10 %.3f" % ((rel<.01).mean(),(rel<.02).mean(),(rel<.05).mean(),(rel<.10).mean()))
        m = snr<8; print("  snr<8 w10 %.3f" % (rel[m]<.1).mean(), " snr>=16 w10 %.3f" % (rel[snr>=16]<.1).mean())
        # constant prediction baseline
        for c in [np.median(t), t.mean()]:
            r0 = abs(c - t)/t
            print("  const pred %.4f: w2 %.3f w5 %.3f w10 %.3f" % (c, (r0<.02).mean(), (r0<.05).mean(), (r0<.1).mean()))
        # AUC
        ss, sb = s, b[f"sigma_{v}_pred"].values
        from scipy.stats import mannwhitneyu
        U = mannwhitneyu(-ss, -sb).statistic
        print("  AUC(-sigma) %.4f" % (U/(len(ss)*len(sb))))
