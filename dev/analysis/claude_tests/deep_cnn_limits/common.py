"""Newtonian stationary-phase chirp, noise spectra and band integrals.

Phase (Cutler and Flanagan 1994, eq. 24, G = c = 1):
    Psi(f) = 2 pi f t_c - phi_c - pi/4 + (3/128) (pi Mc f)^(-5/3)
Amplitude |h(f)| = A f^(-7/6).
"""
import numpy as np

TSUN = 4.925490947e-6          # G Msun / c^3, s
O3_BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"


# ---------------------------------------------------------------- chirp
def tau_of_f(f, mc):
    """Newtonian time before merger at GW frequency f (s)."""
    return 5 / 256 * (TSUN * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(tau, mc):
    """Newtonian GW frequency a time tau before merger (Hz)."""
    return (256 * tau / 5) ** (-3 / 8) * (TSUN * mc) ** (-5 / 8) / np.pi


def mtot_equal(mc):
    """Total mass for equal masses (Msun)."""
    return 2 ** (6 / 5) * mc


def f_isco(mtot):
    """c^3 / (6^(3/2) pi G M) in Hz."""
    return 1 / (6 ** 1.5 * np.pi * TSUN * mtot)


def psi(f, lnmc, tc, phic):
    """SPA phase."""
    mc = np.exp(lnmc)
    return 2 * np.pi * f * tc - phic - np.pi / 4 + 3 / 128 * (np.pi * TSUN * mc * f) ** (-5 / 3)


def dpsi_dlnmc(f, mc):
    return -5 / 128 * (np.pi * TSUN * mc * f) ** (-5 / 3)


def dpsi_dtc(f):
    return 2 * np.pi * f


def band(mc, T=4.0, fmax_cap=1024.0):
    """(f_lo, f_hi): frequency T before merger, up to min(f_ISCO, cap)."""
    return f_of_tau(T, mc), min(f_isco(mtot_equal(mc)), fmax_cap)


# ---------------------------------------------------------------- noise
def o3_avg_psd(which="harmonic"):
    """O3 average PSD per detector (mean of 4000 training-window PSDs),
    0.5 Hz grid 0..1024 Hz. which: 'harmonic' (network), 'H1', 'L1'."""
    z = np.load(O3_BANK)
    p = z["psd"]
    fgrid = np.arange(p.shape[1]) * 0.5
    if which == "H1":
        s = p[0]
    elif which == "L1":
        s = p[1]
    else:
        s = 2 / (1 / p[0] + 1 / p[1])
    return fgrid, s


def ajith2011_zdhp(f):
    """aLIGO zero-detuned high-power fit (Ajith 2011, arXiv:1107.1267)."""
    x = f / 245.4
    return 1e-48 * (0.0152 * x ** -4 + 0.2935 * x ** 2.25 + 2.7951 * x ** 1.5
                    - 6.5080 * x ** 0.75 + 17.7622)


def cf94_eq4(f):
    """Cutler and Flanagan 1994 eq. (4), f > 10 Hz."""
    f0, s0 = 70.0, 3e-48
    return s0 * ((f0 / f) ** 4 + 2 * (1 + (f / f0) ** 2))


class Noise:
    """PSD as a callable; tabulated PSDs are linear between knots."""

    def __init__(self, kind="o3"):
        self.kind = kind
        if kind.startswith("o3"):
            which = {"o3": "harmonic", "o3H1": "H1", "o3L1": "L1"}[kind]
            self.fg, self.sg = o3_avg_psd(which)
        self.knot = 0.5          # Hz, integration cell size

    def __call__(self, f):
        if self.kind.startswith("o3"):
            return np.interp(f, self.fg, self.sg)
        if self.kind == "ajith":
            return ajith2011_zdhp(f)
        if self.kind == "cf94":
            return cf94_eq4(f)
        if self.kind == "white":
            return np.ones_like(f)
        raise ValueError(self.kind)


# ---------------------------------------------------------------- quadrature
_GX, _GW = np.polynomial.legendre.leggauss(8)


def nodes(fa, fb, cell=0.5):
    """Gauss-Legendre nodes and weights on [fa, fb], cells cut at multiples
    of `cell` so a piecewise-linear PSD is smooth inside every cell."""
    if fb <= fa:
        return np.zeros(0), np.zeros(0)
    k0, k1 = np.floor(fa / cell), np.ceil(fb / cell)
    edges = np.arange(k0, k1 + 1) * cell
    edges = np.unique(np.clip(edges, fa, fb))
    lo, hi = edges[:-1], edges[1:]
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    f = (mid[:, None] + half[:, None] * _GX[None, :]).ravel()
    w = (half[:, None] * _GW[None, :]).ravel()
    return f, w


def moments(fa, fb, mc, noise):
    """Integrals of w(f) = f^(-7/3)/S(f) times {1, a, b, a a, a b, b b} over
    [fa, fb], with a = dPsi/dlnMc and b = dPsi/dt_c; also the centered
    (within-interval) covariance computed in two passes."""
    f, q = nodes(fa, fb, noise.knot)
    w = q * f ** (-7 / 3) / noise(f)
    a, b = dpsi_dlnmc(f, mc), dpsi_dtc(f)
    m0 = w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    raw = dict(m0=m0, ma=ma, mb=mb, maa=(w * a * a).sum(), mab=(w * a * b).sum(), mbb=(w * b * b).sum())
    ac, bc = a - ma / m0, b - mb / m0
    cov = np.array([[(w * ac * ac).sum(), (w * ac * bc).sum()],
                    [(w * ac * bc).sum(), (w * bc * bc).sum()]]) / m0
    raw["cov"] = cov
    return raw


def segment_edges_f(mc, R, T=4.0, fmax_cap=1024.0, offset=0.0):
    """Ascending frequency edges of time segments of length R counted back
    from merger (boundaries at tau = (j + offset) R, j = 0, 1, ...), plus the
    window start tau = T, mapped by tau(f) and clipped to the band."""
    flo, fhi = band(mc, T, fmax_cap)
    taus = [0.0, T]
    j = 0
    while (j + offset) * R < T:
        t = (j + offset) * R
        if t > 0:
            taus.append(t)
        j += 1
    taus = np.array(taus)
    fe = np.full(taus.shape, np.inf)
    pos = taus > 0
    fe[pos] = f_of_tau(taus[pos], mc)
    fe = np.unique(np.clip(fe, flo, fhi))      # sorted ascending
    return fe
