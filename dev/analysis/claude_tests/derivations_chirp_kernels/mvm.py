"""Match against kernel chirp-mass offset, as in match_vs_mc.png."""
import numpy as np
exec(open("tuning.py").read().split("for mk in")[0])
off = np.linspace(-0.03, 0.03, 241)
mcs = 1.2 * np.exp(off)
s = chirp(1.2)
m = np.array([peak(s, x) for x in mcs])
m = m / m.max()
pct = (mcs / 1.2 - 1) * 100
for p in (1.0, 2.0, 3.0):
    i = np.argmin(np.abs(pct - p)); print(f"+{p}%: {m[i]:.3f} (at {pct[i]:.3f}%)")
neg = pct < -0.3
print("max for offsets below -0.3%:", m[neg].max().round(3), "at", pct[neg][np.argmax(m[neg])].round(3))
print("first offset below 0 where match<0.25:", pct[(pct<0)&(m<0.25)].max().round(3))
print("max below -1%:", m[pct < -1].max().round(3))
