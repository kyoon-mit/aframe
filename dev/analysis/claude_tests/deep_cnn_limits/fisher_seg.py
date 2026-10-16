"""Chirp-mass Cramer-Rao bound when phase is kept only inside time segments
of length R (one free phase per segment) against the coherent bound.

Writes fisher_results.json and prints the tables used in fisher.md.
"""
import json
import sys

import numpy as np

from common import (Noise, band, dpsi_dlnmc, dpsi_dtc, f_isco, f_of_tau, moments, nodes,
                    psi, segment_edges_f, tau_of_f)

SNR = 8.0
MCS = (0.85, 1.2, 2.75)
RS = (0.44, 1.0, 2.0, 4.0)
OUT = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/fisher_results.json"


def inv_equil(F):
    """Inverse of a symmetric positive matrix after unit-diagonal scaling."""
    d = 1 / np.sqrt(np.diag(F))
    return d[:, None] * np.linalg.inv(d[:, None] * F * d[None, :]) * d[None, :]


def coherent(mc, noise, snr=SNR, T=4.0, fcap=1024.0, flo=None, fhi=None):
    """sigma(lnMc), sigma(t_c), sigma(phi_c) from the 3x3 Fisher matrix
    (lnMc, t_c, phi_c), and sigma(lnMc) from the closed form
    1 / (snr^2 * partial variance) of the centered moments."""
    lo, hi = band(mc, T, fcap)
    flo = lo if flo is None else flo
    fhi = hi if fhi is None else fhi
    m = moments(flo, fhi, mc, noise)
    s = snr ** 2 / m["m0"]
    F = s * np.array([[m["maa"], m["mab"], -m["ma"]],
                      [m["mab"], m["mbb"], -m["mb"]],
                      [-m["ma"], -m["mb"], m["m0"]]])
    C = inv_equil(F)
    G = snr ** 2 * m["cov"]
    closed = np.sqrt(np.linalg.inv(G)[0, 0])
    return dict(sig_lnmc=np.sqrt(C[0, 0]), sig_tc=np.sqrt(C[1, 1]), sig_phic=np.sqrt(C[2, 2]),
                sig_lnmc_closed=closed, flo=flo, fhi=fhi, F=F)


def segmented(mc, R, noise, snr=SNR, T=4.0, fcap=1024.0, offset=0.0, per_segment_time=False):
    """Block Fisher matrix with one phase per segment (and, if
    per_segment_time, one arrival time per segment); returns sigma(lnMc)
    after inverting the full matrix, the Schur-complement closed form,
    the frequency edges and each segment's share of SNR^2."""
    fe = segment_edges_f(mc, R, T, fcap, offset)
    segs = [moments(fe[k], fe[k + 1], mc, noise) for k in range(len(fe) - 1)]
    m0tot = sum(s["m0"] for s in segs)
    sc = snr ** 2 / m0tot
    N = len(segs)
    if not per_segment_time:
        F = np.zeros((2 + N, 2 + N))
        for k, s in enumerate(segs):
            F[0, 0] += s["maa"]
            F[0, 1] += s["mab"]
            F[1, 1] += s["mbb"]
            F[0, 2 + k] = -s["ma"]
            F[1, 2 + k] = -s["mb"]
            F[2 + k, 2 + k] = s["m0"]
        F[1, 0] = F[0, 1]
        F[2:, 0], F[2:, 1] = F[0, 2:], F[1, 2:]
        F *= sc
        sig = np.sqrt(inv_equil(F)[0, 0])
        G = sum(sc * s["m0"] * s["cov"] for s in segs)
        closed = np.sqrt(np.linalg.inv(G)[0, 0])
    else:
        P = 1 + 2 * N
        F = np.zeros((P, P))
        for k, s in enumerate(segs):
            it, ip = 1 + k, 1 + N + k
            F[0, 0] += s["maa"]
            F[0, it] = s["mab"]
            F[0, ip] = -s["ma"]
            F[it, it] = s["mbb"]
            F[it, ip] = -s["mb"]
            F[ip, ip] = s["m0"]
        F = np.triu(F) + np.triu(F, 1).T
        F *= sc
        sig = np.sqrt(inv_equil(F)[0, 0])
        info = sum(sc * s["m0"] * (s["cov"][0, 0] - s["cov"][0, 1] ** 2 / s["cov"][1, 1]) for s in segs)
        closed = 1 / np.sqrt(info)
    frac = [s["m0"] / m0tot for s in segs]
    return dict(sig_lnmc=sig, sig_lnmc_closed=closed, edges=fe.tolist(), frac=frac, N=N)


# ------------------------------------------------------------------ checks
def check_tau(mc=1.2):
    """tau(f) against -(1/2pi) d(Psi - 2 pi f t_c)/df by central differences."""
    out = []
    for f in (50.0, 80.0, 150.0, 400.0):
        h = 1e-4 * f
        d = (psi(f + h, np.log(mc), 0, 0) - psi(f - h, np.log(mc), 0, 0)) / (2 * h)
        out.append(abs(-d / (2 * np.pi) / tau_of_f(f, mc) - 1))
    return max(out)


def check_numeric_derivs(mc, noise, snr=SNR):
    """Coherent Fisher from finite differences of the complex waveform
    against the analytic one; returns max relative difference of entries
    and of sigma(lnMc)."""
    flo, fhi = band(mc)
    f, q = nodes(flo, fhi, noise.knot)
    S = noise(f)
    amp = f ** (-7 / 6)
    A = snr / np.sqrt(4 * np.sum(q * amp ** 2 / S))
    th0 = np.array([np.log(mc), 0.0, 0.0])
    steps = np.array([1e-7, 1e-8, 1e-6])

    def h(th):
        return A * amp * np.exp(1j * psi(f, *th))

    D = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = steps[i]
        D.append((h(th0 + e) - h(th0 - e)) / (2 * steps[i]))
    Fn = np.array([[4 * np.real(np.sum(q * D[i] * np.conj(D[j]) / S)) for j in range(3)] for i in range(3)])
    Fa = coherent(mc, noise, snr)["F"]
    rel = np.max(np.abs(Fn - Fa) / np.sqrt(np.outer(np.diag(Fa), np.diag(Fa))))
    sn, sa = np.sqrt(inv_equil(Fn)[0, 0]), np.sqrt(inv_equil(Fa)[0, 0])
    return rel, abs(sn / sa - 1)


def check_white_analytic(mc=1.2, f1=80.0, f2=1024.0):
    """Moments for S = 1 against closed-form power-law integrals."""
    from common import TSUN
    K = -5 / 128 * (np.pi * TSUN * mc) ** (-5 / 3)

    def I(p):
        return (f2 ** (p + 1) - f1 ** (p + 1)) / (p + 1)

    exact = dict(m0=I(-7 / 3), ma=K * I(-4), mb=2 * np.pi * I(-4 / 3), maa=K * K * I(-17 / 3),
                 mab=2 * np.pi * K * I(-3), mbb=4 * np.pi ** 2 * I(-1 / 3))
    m = moments(f1, f2, mc, Noise("white"))
    return max(abs(m[k] / exact[k] - 1) for k in exact)


def check_cf94():
    """Cutler and Flanagan 1994 eq. (28): Newtonian waveform, parameters
    (ln A, t_c, phi_c, ln Mc), noise eq. (4) above 10 Hz, cut at f_ISCO,
    S/N = 10. Paper: Dlnmc = 1.2e-5 (Mc/Msun)^(5/3), Dtc = 0.40 ms,
    Dphic = 0.25 rad, DlnA = 0.10."""
    noise = Noise("cf94")
    res = []
    for m1, m2 in ((1.4, 1.4), (1.0, 1.0), (1.4, 1.0)):
        mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        r = coherent(mc, noise, snr=10.0, flo=10.0, fhi=f_isco(m1 + m2))
        # ln A decouples from the phase parameters: Re(i ...) = 0, so the
        # 4x4 inverse is the 3x3 inverse plus 1/snr^2 for ln A.
        res.append(dict(m1=m1, m2=m2, mc=mc, dlnmc=r["sig_lnmc"], paper_dlnmc=1.2e-5 * mc ** (5 / 3),
                        dtc_ms=1e3 * r["sig_tc"], dphic=r["sig_phic"], dlnA=1 / 10.0))
    return res


def main():
    noise = Noise("o3")
    R = {}
    print("band edges (O3 average PSD, harmonic mean of H1 and L1):")
    for mc in MCS:
        flo, fhi = band(mc)
        print(f"  Mc {mc}: f_4s = {flo:.2f} Hz, f_hi = {fhi:.1f} Hz (f_ISCO {f_isco(2 ** 1.2 * mc):.1f})")

    # checks
    R["check_tau_maxrel"] = check_tau()
    R["check_white_maxrel"] = check_white_analytic()
    R["check_numeric_derivs"] = {str(mc): check_numeric_derivs(mc, noise) for mc in MCS}
    R["check_cf94"] = check_cf94()
    print("check tau(f) vs dPsi/df: max rel diff", R["check_tau_maxrel"])
    print("check white-noise moments vs closed form: max rel diff", R["check_white_maxrel"])
    print("check numerical vs analytic Fisher (max rel entry diff, rel sigma diff):", R["check_numeric_derivs"])
    for c in R["check_cf94"]:
        print("CF94 check {m1}+{m2}: Dlnmc {dlnmc:.3e} (paper {paper_dlnmc:.3e}), Dtc {dtc_ms:.3f} ms (0.40), "
              "Dphic {dphic:.3f} rad (0.25)".format(**c))

    # main table
    tab = {}
    for mc in MCS:
        coh = coherent(mc, noise)
        row = dict(coherent=coh["sig_lnmc"], coherent_closed=coh["sig_lnmc_closed"],
                   sig_tc_ms=1e3 * coh["sig_tc"], sig_phic=coh["sig_phic"])
        for Rv in RS:
            sg = segmented(mc, Rv, noise)
            st = segmented(mc, Rv, noise, per_segment_time=True)
            offs = [segmented(mc, Rv, noise, offset=o)["sig_lnmc"] for o in np.arange(0, 1, 0.05)]
            offs_t = [segmented(mc, Rv, noise, offset=o, per_segment_time=True)["sig_lnmc"]
                      for o in np.arange(0, 1, 0.05)]
            row[str(Rv)] = dict(shared_tc=sg["sig_lnmc"], shared_tc_closed=sg["sig_lnmc_closed"],
                                N=sg["N"], frac=sg["frac"], edges=sg["edges"],
                                offset_range=[min(offs), max(offs)],
                                per_seg_time=st["sig_lnmc"], per_seg_time_closed=st["sig_lnmc_closed"],
                                per_seg_time_offset_range=[min(offs_t), max(offs_t)])
        tab[str(mc)] = row
    R["table"] = tab

    print(f"\nsigma(ln Mc) at SNR {SNR}, in % (shared t_c, one phase per segment), ratio to coherent:")
    for mc in MCS:
        row = tab[str(mc)]
        s = f"  Mc {mc}: coherent {100 * row['coherent']:.4f}%  (closed {100 * row['coherent_closed']:.4f}%)"
        print(s)
        for Rv in RS:
            r = row[str(Rv)]
            print(f"     R {Rv}: N {r['N']}, {100 * r['shared_tc']:.4f}% (closed {100 * r['shared_tc_closed']:.4f}%), "
                  f"x{r['shared_tc'] / row['coherent']:.3f}; offsets {100 * r['offset_range'][0]:.4f}-"
                  f"{100 * r['offset_range'][1]:.4f}%; per-seg time {100 * r['per_seg_time']:.4f}% "
                  f"(closed {100 * r['per_seg_time_closed']:.4f}%), x{r['per_seg_time'] / row['coherent']:.2f}, "
                  f"offsets {100 * r['per_seg_time_offset_range'][0]:.4f}-{100 * r['per_seg_time_offset_range'][1]:.4f}%")
            print("        SNR^2 shares (early to late):", " ".join(f"{x:.3f}" for x in r["frac"]))

    # convergence R -> 4 s
    conv = {}
    for mc in MCS:
        coh = tab[str(mc)]["coherent"]
        conv[str(mc)] = {str(Rv): segmented(mc, Rv, noise)["sig_lnmc"] / coh
                         for Rv in (0.25, 0.44, 1.0, 2.0, 3.0, 3.5, 3.9, 3.99, 4.0)}
    R["convergence_ratio"] = conv
    print("\nratio segmented/coherent vs R (shared t_c):")
    for mc in MCS:
        print("  Mc", mc, " ".join(f"R{k}:{v:.6f}" for k, v in conv[str(mc)].items()))

    # sensitivity: PSD choice and the 512 Hz cap of the free-kernel network
    sens = {}
    for kind in ("o3H1", "o3L1", "ajith"):
        nz = Noise(kind)
        sens[kind] = {str(mc): dict(coherent=coherent(mc, nz)["sig_lnmc"],
                                    **{f"R{Rv}": segmented(mc, Rv, nz)["sig_lnmc"] for Rv in RS},
                                    **{f"R{Rv}_t": segmented(mc, Rv, nz, per_segment_time=True)["sig_lnmc"] for Rv in RS})
                      for mc in MCS}
    sens["o3_cap512"] = {str(mc): dict(coherent=coherent(mc, noise, fcap=512.0)["sig_lnmc"],
                                       **{f"R{Rv}": segmented(mc, Rv, noise, fcap=512.0)["sig_lnmc"] for Rv in RS})
                         for mc in MCS}
    R["sensitivity"] = sens
    print("\nsensitivity (sigma in %):")
    for kind, d in sens.items():
        for mc, v in d.items():
            print(f"  {kind:10s} Mc {mc}: " + " ".join(f"{k}:{100 * x:.4f}" for k, x in v.items()))

    # comparison with derivations #sep-fisher (3.5 s window, 4 params, 3.5PN)
    sep = {}
    for mc in MCS:
        sep[str(mc)] = {f"snr{s}": coherent(mc, noise, snr=s, T=3.5)["sig_lnmc"] for s in (8.0, 12.0)}
    R["sep_fisher_compare"] = sep
    print("\n3.5 s window, Newtonian 3 parameters (compare #sep-fisher 0.198% median, SNR 8-12, 4 params 3.5PN):")
    for mc, v in sep.items():
        print(f"  Mc {mc}: " + " ".join(f"{k}:{100 * x:.4f}%" for k, x in v.items()))

    json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("\nwrote", OUT)


if __name__ == "__main__":
    sys.exit(main())
