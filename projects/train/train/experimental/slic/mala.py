"""MALA sampler of Legin et al. (2024), batched over windows and walkers.

Port of slicgw ``src/slicgw/sampling/mala.py`` and ``sampling/utils.py``
(MIT, Ronan Legin). Proposal theta' = theta + eps M M^T g + sqrt(2 eps) M xi,
with g the gradient of the log posterior and M a lower Cholesky factor. The
change in log posterior in the acceptance ratio is Simpson's rule with 2
intervals along the straight line from theta to theta'. Warm-up adapts the
step size every 5 steps and re-estimates M from the covariance of block means
over a growing window; both are frozen afterwards. Proposals outside the
prior support, or with a NaN log alpha or a non-finite score, are rejected.
Shapes: theta (windows, walkers, parameters); step size (windows,); M
(windows, parameters, parameters).
"""

import math
from typing import Callable

import torch

TARGET_ACCEPTANCE = 0.60
GAIN = 0.5
FIRST_WINDOW = 100
MAX_WINDOW = 1000
WINDOW_GROWTH = 1.4
LAST_UPDATE_BEFORE_END = 200
START_SCALE = 0.001
START_REDRAWS = 100


def _is_update(step, last, window, n_warmup):
    final = step == n_warmup - LAST_UPDATE_BEFORE_END
    return step + 1 == last + window or final


def adaptation_steps(n_warmup: int) -> list[tuple[int, int]]:
    """(step, window length) of every mass-matrix update in the warm-up."""
    out = []
    window, last = FIRST_WINDOW, FIRST_WINDOW
    for step in range(n_warmup):
        if _is_update(step, last, window, n_warmup):
            out.append((step, window))
            if window <= MAX_WINDOW:
                window = int(window * WINDOW_GROWTH)
            last = step + 1
    return out


def simpson_delta(g_start, g_mid, g_end, v):
    """Line integral of g along v (Simpson's rule, 2 intervals)."""
    y0 = (g_start * v).sum(-1)
    y1 = (g_mid * v).sum(-1)
    y2 = (g_end * v).sum(-1)
    return 0.5 / 3 * (y0 + 4 * y1 + y2)


def _times(x, matrix):
    """x @ matrix^T over the last axis, one matrix per window."""
    return torch.einsum("wkd,wed->wke", x, matrix)


def _kernel_quadratic(x, mean, chol, step_size):
    """(x - mean)^T (2 eps M M^T)^{-1} (x - mean)."""
    diff = (x - mean).transpose(1, 2)
    r = torch.linalg.solve_triangular(chol, diff, upper=False)
    return (r**2).sum(1) / (2 * step_size[:, None])


def _normal(shape, generator, device):
    return torch.randn(
        shape, generator=generator, dtype=torch.float64, device=device
    )


def _uniform(shape, generator, device):
    return torch.rand(
        shape, generator=generator, dtype=torch.float64, device=device
    )


def fisher_mass_matrix(
    jacobian: Callable, theta_true: torch.Tensor, prior, variance: float
) -> torch.Tensor:
    """Cholesky of inv(J^T J / variance + diag(prior std)^-2), J whitened
    Jacobian; diag(prior std) where that fails."""
    # jacobian: (windows, parameters) -> (windows, ..., parameters)
    jac = jacobian(theta_true).to(torch.float64)
    jac = jac.reshape(jac.shape[0], -1, jac.shape[-1])
    fisher = jac.transpose(1, 2) @ jac / variance
    fisher = fisher + torch.diag(prior.std.to(fisher) ** -2)
    n_dim = fisher.shape[-1]
    finite = torch.isfinite(fisher).all(-1).all(-1)
    eye = torch.eye(n_dim, dtype=fisher.dtype, device=fisher.device)
    fisher = torch.where(finite[:, None, None], fisher, eye)
    chol_f, info_f = torch.linalg.cholesky_ex(fisher)
    ok = finite & (info_f == 0)
    cov = torch.cholesky_inverse(torch.where(ok[:, None, None], chol_f, eye))
    chol, info = torch.linalg.cholesky_ex(cov)
    ok &= (info == 0) & torch.isfinite(chol).all(-1).all(-1)
    fallback = torch.diag_embed(prior.std.to(chol)).expand_as(chol)
    return torch.where(ok[:, None, None], chol, fallback)


def initial_walkers(
    theta_true: torch.Tensor,
    mass_matrix: torch.Tensor,
    prior,
    n_walkers: int = 32,
    generator: torch.Generator = None,
) -> torch.Tensor:
    """Draws from N(theta_true, 0.001^2 diag(M M^T)) clipped to the prior
    bounds; draws still outside the support are redrawn."""
    device = theta_true.device
    metric = mass_matrix @ mass_matrix.transpose(1, 2)
    scale = START_SCALE * torch.diagonal(metric, dim1=1, dim2=2).sqrt()
    lower, upper = prior.lower.to(device), prior.upper.to(device)
    shape = (theta_true.shape[0], n_walkers, theta_true.shape[1])
    truth = theta_true[:, None].expand(shape)
    theta = truth.clone()
    redraw = torch.ones(shape[:2], dtype=torch.bool, device=device)
    for _ in range(START_REDRAWS):
        draw = truth + scale[:, None] * _normal(shape, generator, device)
        draw = torch.minimum(torch.maximum(draw, lower), upper)
        theta = torch.where(redraw[..., None], draw, theta)
        redraw = redraw & ~prior.in_support(theta)
        if not redraw.any():
            return theta
    return torch.where(redraw[..., None], truth, theta)


class MALA:
    """Args:
        likelihood_score: theta (windows, n, parameters) -> gradient of the
            log likelihood, same shape, float64. Called only at points inside
            the prior support.
        prior: ``priors.InjectionPrior`` or any object with
            ``log_prob_and_grad``, ``in_support``, ``lower``, ``upper``.
    """

    def __init__(self, likelihood_score: Callable, prior):
        self.likelihood_score = likelihood_score
        self.prior = prior

    def posterior_score(self, theta: torch.Tensor) -> torch.Tensor:
        _, grad = self.prior.log_prob_and_grad(theta)
        return self.likelihood_score(theta) + grad

    def step(self, theta, score, step_size, chol, noise, uniform):
        """One MALA step; returns theta, score, acceptance, log alpha."""
        eps = step_size[:, None, None]
        metric = chol @ chol.transpose(1, 2)
        drift = theta + eps * _times(score, metric)
        proposal = drift + torch.sqrt(2 * eps) * _times(noise, chol)
        v = proposal - theta
        mid = 0.5 * v + theta

        prior = self.prior
        inside = prior.in_support(proposal) & prior.in_support(mid)
        walkers = theta.shape[1]
        points = torch.cat([proposal, mid], 1)
        keep = torch.cat([inside, inside], 1)[..., None]
        points = torch.where(keep, points, torch.cat([theta, theta], 1))
        scores = self.posterior_score(points)
        score_new, score_mid = scores[:, :walkers], scores[:, walkers:]

        delta = simpson_delta(score, score_mid, score_new, v)
        delta = torch.where(inside, delta, torch.full_like(delta, -math.inf))
        back_mean = proposal + eps * _times(score_new, metric)
        forward = _kernel_quadratic(proposal, drift, chol, step_size)
        backward = _kernel_quadratic(theta, back_mean, chol, step_size)
        log_alpha = delta - 0.5 * backward + 0.5 * forward
        finite = torch.isfinite(scores).all(-1)
        valid = finite[:, :walkers] & finite[:, walkers:]
        valid &= ~torch.isnan(log_alpha)
        log_alpha = torch.where(
            valid, log_alpha, torch.full_like(log_alpha, -math.inf)
        )
        accepted = torch.log(uniform) < log_alpha
        theta = torch.where(accepted[..., None], proposal, theta)
        score = torch.where(accepted[..., None], score_new, score)
        return theta, score, accepted, log_alpha

    def sample(
        self,
        theta_start: torch.Tensor,
        mass_matrix: torch.Tensor,
        step_size: float = 0.002,
        n_warmup: int = 10000,
        n_steps: int = 20000,
        generator: torch.Generator = None,
    ) -> dict:
        """Warm-up, then kept steps, from theta_start (windows, walkers,
        parameters).

        Returns chain (n_steps, windows, walkers, parameters) of kept states,
        warmup (n_warmup + 1, ...) from the start, the final step size and
        mass matrix, mean acceptance over kept steps, the update steps and
        the count of failed updates (M kept) per window.
        """
        theta = theta_start.to(torch.float64)
        device = theta.device
        n_windows, n_walkers, n_dim = theta.shape
        eps = torch.full(
            (n_windows,), step_size, dtype=torch.float64, device=device
        )
        chol = mass_matrix.to(theta).clone()
        score = self.posterior_score(theta)

        shape = (n_windows, n_walkers, n_dim)
        warmup = torch.empty((n_warmup + 1, *shape), dtype=torch.float64)
        chain = torch.empty((n_steps, *shape), dtype=torch.float64)
        warmup[0] = theta.cpu()
        acc_window = torch.zeros_like(eps)
        acc_kept = torch.zeros_like(eps)
        failed = torch.zeros(n_windows, dtype=torch.long)
        updates = []
        window, last = FIRST_WINDOW, FIRST_WINDOW

        for step in range(n_warmup + n_steps):
            noise = _normal(theta.shape, generator, device)
            uniform = _uniform(theta.shape[:2], generator, device)
            theta, score, accepted, _ = self.step(
                theta, score, eps, chol, noise, uniform
            )
            acc = accepted.to(torch.float64).mean(1)
            if step >= n_warmup:
                chain[step - n_warmup] = theta.cpu()
                acc_kept += acc
                continue

            acc_window += acc
            if (step + 1) % 5 == 0:
                error = TARGET_ACCEPTANCE - acc_window / 5
                eps = eps / (1 + GAIN * error)
                acc_window.zero_()
            if _is_update(step, last, window, n_warmup):
                block = window // 5
                n_usable = window - window % block
                recent = warmup[max(step + 1 - n_usable, 0) : step + 1]
                means = recent.reshape(-1, block, *shape).mean(1)
                means = means.permute(1, 0, 2, 3).reshape(n_windows, -1, n_dim)
                centered = means.to(device)
                centered = centered - centered.mean(1, keepdim=True)
                cov = centered.transpose(1, 2) @ centered
                cov = cov / (centered.shape[1] - 1)
                new, info = torch.linalg.cholesky_ex(cov)
                chol = torch.where((info == 0)[:, None, None], new, chol)
                failed += (info != 0).cpu().long()
                updates.append(step)
                if window <= MAX_WINDOW:
                    window = int(window * WINDOW_GROWTH)
                last = step + 1
            warmup[step + 1] = theta.cpu()

        return {
            "chain": chain,
            "warmup": warmup,
            "step_size": eps,
            "mass_matrix": chol,
            "acceptance": acc_kept / max(n_steps, 1),
            "updates": updates,
            "failed_updates": failed,
        }
