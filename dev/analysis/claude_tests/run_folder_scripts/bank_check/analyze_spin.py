"""Chirp mass from the Mc x chi_eff bank: chi=0 slice, joint argmax, and
marginalized over chi (logsumexp of beta * SNR^2)."""
import sys
import numpy as np
from analyze import estimate, table, parabola_peak, BANDS

for fn in sys.argv[1:]:
    d = np.load(fn)
    sig = np.isfinite(d["true_mc"])
    peak, mc, chi = d["peak"][sig], d["bank_mc"], d["bank_chi"]
    true, snr = d["true_mc"][sig], d["snr"][sig]
    print(f"\n{fn}: {sig.sum()} signal windows; within 1/2% by SNR band " + " ".join(f"{a}-{b}" for a, b in BANDS) + " all")
    gi = np.round((np.log(mc) - np.log(0.85)) / 0.005).astype(int)
    G = gi.max() + 1
    groups = {c: np.where(np.isclose(chi, c))[0] for c in (-0.2, -0.1, 0.0, 0.1, 0.2)}
    for interp in (False, True):
        t = "interp" if interp else "argmax"
        est, _ = estimate(peak, mc, [groups[0.0]], interp); table(f"chi 0 only (235), {t}", est, true, snr)
        est, _ = estimate(peak, mc, [groups[c] for c in (-0.1, 0.0, 0.1)], interp); table(f"chi -.1/0/.1 joint max, {t}", est, true, snr)
        est, _ = estimate(peak, mc, list(groups.values()), interp); table(f"chi 5 joint max (1159), {t}", est, true, snr)
    lgrid = np.log(0.85) + 0.005 * np.arange(G)
    for chis in ((-0.1, 0.0, 0.1), (-0.2, -0.1, 0.0, 0.1, 0.2)):
        for beta in (0.25, 0.5, 1.0):
            M = np.full((len(peak), G), -np.inf)
            for c in chis:
                g = groups[c]
                M[:, gi[g]] = np.logaddexp(M[:, gi[g]], beta * peak[:, g])
            valid = np.isfinite(M).all(0)
            Mv, lv = M[:, valid], lgrid[valid]
            j = Mv.argmax(1); rows = np.arange(len(Mv)); jj = np.clip(j, 1, len(lv) - 2)
            x = np.stack([lv[jj - 1], lv[jj], lv[jj + 1]]); y = np.stack([Mv[rows, jj - 1], Mv[rows, jj], Mv[rows, jj + 1]])
            l = np.where((j > 0) & (j < len(lv) - 1), parabola_peak(x, y), lv[j])
            table(f"marg chi{len(chis)} beta {beta}, interp", np.exp(l), true, snr)
