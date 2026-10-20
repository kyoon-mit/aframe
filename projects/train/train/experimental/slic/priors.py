"""Injection distribution of the 11 SLIC parameters: torch density, sampler.

Masses follow ``priors.priors.end_o3_ratesandpops_bns_uniform_chirp``:
detector-frame chirp mass uniform, redshift uniform in the source frame, and
q = m2/m1 uniform on the part of [q_min, 1] whose source-frame component
masses fall inside the mass box. The joint density of (chirp mass, q) is that
rule integrated over redshift. snr follows ``ml4gw.distributions.PowerLaw``;
tc is uniform over the merger positions of the data module.
"""

import math

import numpy as np
import torch
from numpy.polynomial import chebyshev, legendre

from utils.cosmology import DEFAULT_COSMOLOGY

PARAMETERS = (
    "chirp_mass",
    "mass_ratio",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "phase",
    "snr",
    "tc",
)

NEWTON_STEPS = 40
REDSHIFT_DEGREE = 24
REDSHIFT_GRID = 8193


def _cubic_root(f, df, shape_like):
    """Root of an increasing convex cubic below 1, by Newton from q = 1."""
    with torch.no_grad():
        q = torch.ones_like(shape_like)
        for _ in range(NEWTON_STEPS):
            q = q - f(q) / df(q)
    return q - f(q) / df(q)


def _clamp(x, lo, hi):
    return torch.minimum(torch.maximum(x, lo), hi)


class InjectionPrior:
    """Log density, gradient and sampler of the injection distribution.

    Args:
        mc_det_min, mc_det_max: range of the detector-frame chirp mass.
        q_min: lower edge of the mass-ratio draw.
        z_max: largest redshift.
        spin_max: largest spin magnitude.
        snr_min, snr_max, snr_index: ``ml4gw.distributions.PowerLaw``.
        sample_rate: of the whitened window, in Hz.
        merger_index_min, merger_index_max: first and last window index of
            the geocenter merger; tc covers them with half a sample each side.
        n_nodes: Gauss-Legendre nodes per redshift interval.
    """

    def __init__(
        self,
        mc_det_min: float = 0.85,
        mc_det_max: float = 2.75,
        q_min: float = 0.4,
        z_max: float = 0.15,
        spin_max: float = 0.4,
        snr_min: float = 4.0,
        snr_max: float = 100.0,
        snr_index: float = -3.0,
        sample_rate: float = 2048.0,
        merger_index_min: int = 7169,
        merger_index_max: int = 8192,
        n_nodes: int = 32,
        cosmology=DEFAULT_COSMOLOGY,
    ):
        self.mc_det_min, self.mc_det_max = mc_det_min, mc_det_max
        self.q_min, self.z_max = q_min, z_max
        self.spin_max = spin_max
        self.snr_min, self.snr_max = snr_min, snr_max
        self.snr_index = snr_index
        self.tc_min = (merger_index_min - 0.5) / sample_rate
        self.tc_max = (merger_index_max + 0.5) / sample_rate

        # source-frame mass box, as in DetectorFrameFlatChirpPriorDict
        self.m_min = mc_det_min / (1 + z_max) * 2**0.2
        self.m_max = mc_det_max * 2**0.2
        # source-frame chirp masses where q_min stops being the binding edge
        f1 = (1 + q_min) ** 0.2 * q_min**-0.6
        f2 = (1 + q_min) ** 0.2 * q_min**0.4
        self.mc_kink_low = self.m_min / f2
        self.mc_kink_high = self.m_max / f1

        u, w = legendre.leggauss(n_nodes)
        self._u = torch.tensor((u + 1) / 2, dtype=torch.float64)
        self._w = torch.tensor(w / 2, dtype=torch.float64)

        self._fit_redshift(cosmology)

        a = snr_index + 1
        self._snr_norm = math.log(abs(a)) - math.log(
            abs(snr_max**a - snr_min**a)
        )

        two_pi = 2 * math.pi
        self.lower = torch.tensor(
            [mc_det_min, q_min, 0, 0, 0, 0, 0, 0, 0, snr_min, self.tc_min],
            dtype=torch.float64,
        )
        self.upper = torch.tensor(
            [mc_det_max, 1, spin_max, spin_max, math.pi, math.pi]
            + [two_pi, two_pi, two_pi, snr_max, self.tc_max],
            dtype=torch.float64,
        )
        tc_mid = 0.5 * (self.tc_min + self.tc_max)
        self._inside_point = torch.tensor(
            [1.5, 0.8, 0.2, 0.2, 1.5, 1.5, 3.0, 3.0, 3.0, 10.0, tc_mid],
            dtype=torch.float64,
        )
        self.std = self._std()

    def _fit_redshift(self, cosmology):
        """Chebyshev fit of p(z) / z^2 with p(z) ~ dVc/dz / (1 + z)."""
        n = 4 * REDSHIFT_DEGREE
        x = np.cos(np.pi * (np.arange(n) + 0.5) / n)
        z = (x + 1) / 2 * self.z_max
        p = cosmology.differential_comoving_volume(z).value / (1 + z)
        coef = chebyshev.chebfit(x, p / z**2, REDSHIFT_DEGREE)
        u, w = legendre.leggauss(200)
        zz = (u + 1) / 2 * self.z_max
        fit = chebyshev.chebval(2 * zz / self.z_max - 1, coef)
        coef = coef / np.sum(w * self.z_max / 2 * zz**2 * fit)
        self._z_coef = torch.tensor(coef, dtype=torch.float64)

        grid = np.linspace(0, self.z_max, REDSHIFT_GRID)
        dens = grid**2 * chebyshev.chebval(2 * grid / self.z_max - 1, coef)
        steps = (dens[1:] + dens[:-1]) / 2 * np.diff(grid)
        cdf = np.concatenate([[0], np.cumsum(steps)])
        self._z_grid = torch.tensor(grid, dtype=torch.float64)
        self._z_cdf = torch.tensor(cdf / cdf[-1], dtype=torch.float64)

    def redshift_density(self, z: torch.Tensor) -> torch.Tensor:
        x = 2 * z / self.z_max - 1
        coef = self._z_coef.to(z.device)
        t_prev, t = torch.ones_like(x), x
        out = coef[0] * t_prev + coef[1] * t
        for c in coef[2:]:
            t_prev, t = t, 2 * x * t - t_prev
            out = out + c * t
        return z**2 * out

    def ratio_lower_edge(self, mc_source: torch.Tensor) -> torch.Tensor:
        """Smallest q whose source-frame masses lie in the box."""
        a = (self.m_max / mc_source) ** 5
        b = (self.m_min / mc_source) ** 5
        # m1 <= m_max  <=>  a q^3 - q - 1 >= 0
        q1 = _cubic_root(
            lambda q: a * q**3 - q - 1,
            lambda q: 3 * a * q**2 - 1,
            mc_source,
        )
        # m2 >= m_min  <=>  q^3 + q^2 - b >= 0
        q2 = _cubic_root(
            lambda q: q**3 + q**2 - b,
            lambda q: 3 * q**2 + 2 * q,
            mc_source,
        )
        return torch.clamp(torch.maximum(q1, q2), min=self.q_min)

    def _redshift_range(self, mc, q):
        """Redshifts at which (mc, q) is accepted by the mass box."""
        f1 = (1 + q) ** 0.2 * q**-0.6
        f2 = (1 + q) ** 0.2 * q**0.4
        z_lo = torch.clamp(mc * f1 / self.m_max - 1, min=0)
        z_hi = torch.clamp(mc * f2 / self.m_min - 1, max=self.z_max)
        return z_lo, z_hi

    def _edge_segment(self, mc, z_a, z_b, m_edge, power):
        """Integral of p(z) / (1 - q_lo) over [z_a, z_b] where q_lo is set by
        one mass edge, in s = -log(1 - q_lo), with z = mc f(q_lo) / m_edge - 1
        and f(q) = (1 + q)^0.2 q^power."""
        s_a = -torch.log1p(-self.ratio_lower_edge(mc / (1 + z_a)))
        s_b = -torch.log1p(-self.ratio_lower_edge(mc / (1 + z_b)))
        u, w = self._u.to(mc.device), self._w.to(mc.device)
        s = s_a[..., None] + (s_b - s_a)[..., None] * u
        r = -torch.expm1(-s)
        f = (1 + r) ** 0.2 * r**power
        z = mc[..., None] * f / m_edge - 1
        dz_dr = mc[..., None] * f * (0.2 / (1 + r) + power / r) / m_edge
        integrand = w * self.redshift_density(z) * dz_dr.abs()
        return (s_b - s_a).abs() * integrand.sum(-1)

    def mass_density(self, mc: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """Joint density of (detector-frame chirp mass, q) in the support."""
        z_lo, z_hi = self._redshift_range(mc, q)
        z_hi = torch.maximum(z_hi, z_lo)
        # q_lo is set by m1 <= m_max below k_high, by q_min between,
        # by m2 >= m_min above k_low
        k_high = _clamp(mc / self.mc_kink_high - 1, z_lo, z_hi)
        k_low = _clamp(mc / self.mc_kink_low - 1, z_lo, z_hi)
        u, w = self._u.to(mc.device), self._w.to(mc.device)
        z = k_high[..., None] + (k_low - k_high)[..., None] * u
        flat = (w * self.redshift_density(z)).sum(-1)
        flat = (k_low - k_high) * flat / (1 - self.q_min)
        heavy = self._edge_segment(mc, z_lo, k_high, self.m_max, -0.6)
        light = self._edge_segment(mc, k_low, z_hi, self.m_min, 0.4)
        width = self.mc_det_max - self.mc_det_min
        return (flat + heavy + light) / width

    def in_support(self, theta: torch.Tensor) -> torch.Tensor:
        lower = self.lower.to(theta.device)
        upper = self.upper.to(theta.device)
        inside = ((theta >= lower) & (theta <= upper)).all(-1)
        inside &= (theta[..., 4] > 0) & (theta[..., 4] < math.pi)
        inside &= (theta[..., 5] > 0) & (theta[..., 5] < math.pi)
        q = theta[..., 1].clamp(min=self.q_min)
        z_lo, z_hi = self._redshift_range(theta[..., 0], q)
        return inside & (z_hi > z_lo)

    def _log_prob_inside(self, theta):
        out = torch.log(self.mass_density(theta[..., 0], theta[..., 1]))
        out = out - 2 * math.log(self.spin_max)
        out = out + torch.log(torch.sin(theta[..., 4]) / 2)
        out = out + torch.log(torch.sin(theta[..., 5]) / 2)
        out = out - 3 * math.log(2 * math.pi)
        out = out + self._snr_norm + self.snr_index * torch.log(theta[..., 9])
        return out - math.log(self.tc_max - self.tc_min)

    def _safe(self, theta, inside):
        point = self._inside_point.to(theta.device)
        return torch.where(inside[..., None], theta, point)

    def log_prob(self, theta: torch.Tensor) -> torch.Tensor:
        """Log density, -inf outside the support."""
        inside = self.in_support(theta)
        out = self._log_prob_inside(self._safe(theta, inside))
        return torch.where(inside, out, torch.full_like(out, -math.inf))

    def log_prob_and_grad(self, theta: torch.Tensor):
        """Log density and its gradient; gradient NaN outside the support."""
        inside = self.in_support(theta)
        with torch.enable_grad():
            safe = self._safe(theta, inside).detach().requires_grad_(True)
            out = self._log_prob_inside(safe)
            (grad,) = torch.autograd.grad(out.sum(), safe)
        out = out.detach()
        out = torch.where(inside, out, torch.full_like(out, -math.inf))
        nan = torch.full_like(grad, math.nan)
        return out, torch.where(inside[..., None], grad, nan)

    def sample(self, n: int, generator: torch.Generator = None):
        """n draws, shape (n, 11)."""

        def uniform(lo, hi):
            u = torch.rand(n, generator=generator, dtype=torch.float64)
            return lo + (hi - lo) * u

        mc = uniform(self.mc_det_min, self.mc_det_max)
        u = uniform(0, 1)
        i = torch.searchsorted(self._z_cdf, u).clamp(1, REDSHIFT_GRID - 1)
        c0, c1 = self._z_cdf[i - 1], self._z_cdf[i]
        z0, z1 = self._z_grid[i - 1], self._z_grid[i]
        z = z0 + (u - c0) / (c1 - c0) * (z1 - z0)
        with torch.no_grad():
            q_lo = self.ratio_lower_edge(mc / (1 + z))
        q = uniform(0, 1) * (1 - q_lo) + q_lo
        a_1, a_2 = uniform(0, self.spin_max), uniform(0, self.spin_max)
        tilt_1 = torch.arccos(uniform(-1, 1))
        tilt_2 = torch.arccos(uniform(-1, 1))
        angles = [uniform(0, 2 * math.pi) for _ in range(3)]
        a = self.snr_index + 1
        lo, hi = self.snr_min**a, self.snr_max**a
        snr = (lo + uniform(0, 1) * (hi - lo)) ** (1 / a)
        tc = uniform(self.tc_min, self.tc_max)
        return torch.stack(
            [mc, q, a_1, a_2, tilt_1, tilt_2, *angles, snr, tc], -1
        )

    def _std(self) -> torch.Tensor:
        """Standard deviation of each parameter."""
        u, w = legendre.leggauss(400)
        u, w = torch.tensor((u + 1) / 2), torch.tensor(w / 2)
        mc = u * (self.mc_det_max - self.mc_det_min) + self.mc_det_min
        z = u * self.z_max
        weight = w[:, None] * (w * self.z_max * self.redshift_density(z))
        q_lo = self.ratio_lower_edge(mc[:, None] / (1 + z[None, :]))
        mean = (weight * (1 + q_lo) / 2).sum()
        second = (weight * (1 + q_lo + q_lo**2) / 3).sum()
        q_std = float(torch.sqrt(second - mean**2))

        a = self.snr_index

        def snr_moment(k):
            if a + k + 1 == 0:
                return math.log(self.snr_max / self.snr_min)
            e = a + k + 1
            return (self.snr_max**e - self.snr_min**e) / e

        m0, m1, m2 = snr_moment(0), snr_moment(1), snr_moment(2)
        snr_std = math.sqrt(m2 / m0 - (m1 / m0) ** 2)
        flat = 1 / math.sqrt(12)
        mc_std = (self.mc_det_max - self.mc_det_min) * flat
        tilt_std = math.sqrt(math.pi**2 / 4 - 2)
        return torch.tensor(
            [mc_std, q_std]
            + [self.spin_max * flat] * 2
            + [tilt_std] * 2
            + [2 * math.pi * flat] * 3
            + [snr_std, (self.tc_max - self.tc_min) * flat],
            dtype=torch.float64,
        )
