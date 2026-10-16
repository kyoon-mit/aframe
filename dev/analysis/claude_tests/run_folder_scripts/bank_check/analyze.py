"""Chirp-mass accuracy of bank variants from bank_check.py outputs.

usage: analyze.py FILE.npz [FILE.npz ...]
"""

import sys

import numpy as np

BANDS = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 25), (25, 50)]


def parabola_peak(x, y):
    """Vertex x of the parabola through three points; None if not a maximum."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / d
    xv = -b / (2 * a)
    ok = (a < 0) & (xv > x0) & (xv < x2)
    return np.where(ok, xv, x1)


def estimate(peak, mc, groups, interp):
    """Best template's log chirp mass; groups: list of index arrays, each a
    1D chirp-mass grid (one mass ratio). Interpolation within the winning grid."""
    best_val = np.full(len(peak), -np.inf)
    best_lmc = np.zeros(len(peak))
    for g in groups:
        g = g[np.argsort(mc[g])]
        lm = np.log(mc[g])
        p = peak[:, g]
        j = p.argmax(1)
        v = p[np.arange(len(p)), j]
        lmc = lm[j]
        if interp:
            jj = np.clip(j, 1, len(g) - 2)
            rows = np.arange(len(p))
            x = np.stack([lm[jj - 1], lm[jj], lm[jj + 1]])
            y = np.stack([p[rows, jj - 1], p[rows, jj], p[rows, jj + 1]])
            inner = (j > 0) & (j < len(g) - 1)
            lmc = np.where(inner, parabola_peak(x, y), lmc)
        better = v > best_val
        best_val = np.where(better, v, best_val)
        best_lmc = np.where(better, lmc, best_lmc)
    return np.exp(best_lmc), best_val


def table(name, est, true, snr):
    err = np.abs(est - true) / true
    cells = []
    for a, b in BANDS:
        s = (snr >= a) & (snr < b)
        cells.append("  --  " if s.sum() < 30 else f"{100*(err[s]<.01).mean():3.0f}/{100*(err[s]<.02).mean():3.0f}")
    cells.append(f"{100*(err<.01).mean():3.0f}/{100*(err<.02).mean():3.0f}")
    print(f"{name:34s} " + " ".join(f"{c:>7s}" for c in cells))


if __name__ == "__main__":
    for fn in sys.argv[1:]:
        d = np.load(fn)
        sig = np.isfinite(d["true_mc"])
        peak, mc, q, tag = d["peak"][sig], d["bank_mc"], d["bank_q"], d["tag"]
        true, snr = d["true_mc"][sig], d["snr"][sig]
        print(f"\n{fn}: {sig.sum()} signal windows; within 1/2% by SNR band "
              + " ".join(f"{a}-{b}" for a, b in BANDS) + " all")
        fine = np.where(tag == 0)[0]
        fine = fine[np.argsort(mc[fine])]
        q3 = np.where(tag == 1)[0]
        qgroups = [q3[np.abs(q[q3] - qq) < 0.06] for qq in (0.5, 0.7, 0.9)]
        for interp in (False, True):
            t = "interp" if interp else "argmax"
            est, _ = estimate(peak, mc, qgroups, interp)
            table(f"q 0.5/0.7/0.9, 0.5% grid, {t}", est, true, snr)
            for k, step in ((1, 0.25), (2, 0.5), (4, 1), (8, 2), (16, 4)):
                est, _ = estimate(peak, mc, [fine[::k]], interp)
                table(f"q 0.8, {step}% grid ({len(fine[::k])}), {t}", est, true, snr)
            est, _ = estimate(peak, mc, [g[np.argsort(mc[g])][::2] for g in qgroups], interp)
            table(f"q 0.5/0.7/0.9, 1% grid, {t}", est, true, snr)
