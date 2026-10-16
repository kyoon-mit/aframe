import json, sys, numpy as np
def load(fn):
    d = json.load(open(fn))
    ep = sorted(int(e) for e in d)
    w1 = {e: d[str(e)].get('val/within_1pct_chirp_mass') for e in ep}
    w2 = {e: d[str(e)].get('val/within_2pct_chirp_mass') for e in ep}
    return ep, w1, w2
def rng(w, a, b):
    xs = [v for e, v in w.items() if a <= e <= b and v is not None]
    return (100 * np.mean(xs), len(xs)) if xs else (float('nan'), 0)
for fn, name in [(a.split('=')[0], a.split('=')[1]) for a in sys.argv[1:]]:
    ep, w1, w2 = load(fn)
    s = f'{name}: epochs {ep[0]}-{ep[-1]} (n={len(ep)})'
    for a, b in [(60, 99), (100, 139), (140, 199), (200, 219), (200, 299), (300, 300), (598, 598), (580, 619), (620, 799), (780, 799)]:
        m1, n1 = rng(w1, a, b); m2, n2 = rng(w2, a, b)
        if n1: s += f' | {a}-{b}: {m1:.1f}/{m2:.1f} (n{n1})'
    best = max((v, e) for e, v in w1.items() if v is not None)
    s += f' | best1% {100*best[0]:.1f}@{best[1]} | last {ep[-1]}: {100*w1[ep[-1]]:.1f}/{100*w2[ep[-1]]:.1f}'
    print(s)
