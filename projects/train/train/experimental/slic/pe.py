"""Posterior samples of dumped windows with SLIC (Legin et al. 2024): MALA
over the 11 ``forward.PARAMETERS`` with the score network as noise model.

Per window, the data are its whitened ``X`` plus diffusion noise
sigma(t_min) N(0, 1). The gradient of the log likelihood is -J^T s(d - h),
with h the window's template (``forward.WindowTemplate``), J its Jacobian
and s the network's score at t_min, summed over H1 and L1. It takes one
backward pass through the template per evaluation; the network runs without
gradients. Walkers start near the true values (``mala.initial_walkers``)
with the start matrix of ``mala.fisher_mass_matrix``; walkers whose start
score is not finite are redrawn. Defaults are the paper's settings.

    python -m train.experimental.slic.pe --windows <dump.h5> \\
        --checkpoint <score.ckpt> --out <pe.h5> [--start 0 --count 16]

Seeds: diffusion noise of dump window i from ``seed + i``; walkers and
sampler from ``seed + SAMPLER_SEED_OFFSET + start``.

Output (N windows):
    samples             (N, steps // thin, walkers, 11) kept steps, every
                        ``thin``-th
    chain               (N, warmup + steps + 1, walkers, 11) start, warm-up
                        and kept states (with --full_chain)
    theta_true          (N, 11)
    window              (N,) index in the dump
    acceptance          (N,) mean over kept steps
    step_size           (N,) after warm-up
    mass_matrix         (N, 11, 11) after warm-up
    start_mass_matrix   (N, 11, 11)
    failed_updates      (N,) mass-matrix updates that kept the old matrix
    stuck_walkers       (N,) walkers with a non-finite start score
"""

import argparse
import math
import time
from typing import Callable

import h5py
import numpy as np
import torch

from train.experimental.slic import mala
from train.experimental.slic.dump_windows import read_window
from train.experimental.slic.forward import PARAMETERS, WindowTemplate
from train.experimental.slic.priors import InjectionPrior
from train.experimental.slic.sde import VESDE
from train.experimental.slic.task import load_score_net

T_MIN = 0.3
N_WALKERS = 32
STEP_SIZE = 0.002
N_WARMUP = 10000
N_STEPS = 20000
THIN = 20
SEED = 9872
SAMPLER_SEED_OFFSET = 1_000_000


def network_score(net: torch.nn.Module, t_min: float = T_MIN) -> Callable:
    """Noise score of (n, detectors, samples) float64 residuals: each
    detector through the single-channel network at t_min, in float32."""

    @torch.no_grad()
    def score(x):
        n, ifos, size = x.shape
        flat = x.reshape(n * ifos, 1, size).float()
        t = torch.full((len(flat),), t_min, device=x.device)
        return net.score(flat, t).reshape(n, ifos, size).to(x.dtype)

    return score


def gaussian_score(variance: float) -> Callable:
    """Score of white Gaussian noise with the given variance."""
    return lambda x: -x / variance


class LikelihoodScore:
    """Gradient of the log likelihood of each window's data.

    Args:
        templates: one ``WindowTemplate`` per window.
        data: (windows, 2, 8192) float64.
        noise_score: (n, 2, 8192) residuals -> score, same shape.

    ``__call__`` maps theta (windows, n, 11) to (windows, n, 11); rows with
    a non-finite score or template gradient come back NaN.
    """

    def __init__(self, templates, data, noise_score, log_every=1000):
        self.templates = templates
        self.data = data
        self.noise_score = noise_score
        self.log_every = log_every
        self.calls = 0
        self.start = time.time()

    def __call__(self, theta: torch.Tensor) -> torch.Tensor:
        out = []
        for template, data, rows in zip(self.templates, self.data, theta):
            rows = rows.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                h = template(rows)
            score = self.noise_score(data - h.detach())
            ok = torch.isfinite(score).flatten(1).all(1)
            score = torch.where(ok[:, None, None], score, 0.0)
            if h.requires_grad:
                (grad,) = torch.autograd.grad(h, rows, grad_outputs=-score)
            else:
                grad = torch.zeros_like(rows)
            out.append(torch.where(ok[:, None], grad, math.nan))
        self.calls += 1
        if self.log_every and self.calls % self.log_every == 0:
            elapsed = time.time() - self.start
            print(
                f"{self.calls} evaluations, {elapsed:.0f} s "
                f"({elapsed / self.calls:.3f} s each)",
                flush=True,
            )
        return torch.stack(out)


def template_jacobian(template: WindowTemplate, theta: torch.Tensor):
    """(2, 8192, 11) Jacobian at theta (11,), by double backward."""
    rows = theta[None].detach().clone().requires_grad_(True)
    with torch.enable_grad():
        h = template(rows)[0]
        u = torch.zeros_like(h, requires_grad=True)
        (vjp,) = torch.autograd.grad(
            h, rows, grad_outputs=u, create_graph=True
        )
        columns = [
            torch.autograd.grad(vjp[0, k], u, retain_graph=True)[0]
            for k in range(len(PARAMETERS))
        ]
    return torch.stack(columns, -1).detach()


def start_walkers(sampler, theta_true, chol, prior, n_walkers, generator):
    """``mala.initial_walkers`` with walkers of non-finite start score
    redrawn; returns walkers and the count still non-finite per window."""
    theta = mala.initial_walkers(theta_true, chol, prior, n_walkers, generator)
    bad = ~torch.isfinite(sampler.posterior_score(theta)).all(-1)
    for _ in range(mala.START_REDRAWS):
        if not bad.any():
            break
        draw = mala.initial_walkers(
            theta_true, chol, prior, n_walkers, generator
        )
        theta = torch.where(bad[..., None], draw, theta)
        bad = ~torch.isfinite(sampler.posterior_score(theta)).all(-1)
    return theta, bad.sum(1)


def run(
    templates,
    data,
    theta_true,
    noise_score,
    variance,
    n_walkers=N_WALKERS,
    step_size=STEP_SIZE,
    n_warmup=N_WARMUP,
    n_steps=N_STEPS,
    generator=None,
    prior=None,
    log_every=1000,
):
    """MALA for a batch of windows; returns the ``MALA.sample`` output plus
    ``start_mass_matrix``, ``stuck_walkers`` and ``seconds``. ``variance``
    is the whitened noise variance used for the start matrix."""
    prior = prior or InjectionPrior()
    likelihood = LikelihoodScore(templates, data, noise_score, log_every)
    sampler = mala.MALA(likelihood, prior)
    chol = mala.fisher_mass_matrix(
        lambda thetas: torch.stack(
            [template_jacobian(m, t) for m, t in zip(templates, thetas)]
        ),
        theta_true,
        prior,
        variance,
    )
    theta, stuck = start_walkers(
        sampler, theta_true, chol, prior, n_walkers, generator
    )
    start = time.time()
    likelihood.calls, likelihood.start = 0, start
    out = sampler.sample(
        theta, chol, step_size, n_warmup, n_steps, generator=generator
    )
    out.update(
        start_mass_matrix=chol,
        stuck_walkers=stuck,
        seconds=time.time() - start,
    )
    return out


def load_windows(path, indices, device):
    """Templates, whitened data (float64) and true values of dump windows."""
    templates, data, truth = [], [], []
    for i in indices:
        w = read_window(path, i)
        template = WindowTemplate(
            w["psds"], w["dec"], w["psi"], w["phi"], w["theta_jn"]
        )
        templates.append(template.to(device))
        data.append(torch.as_tensor(w["X"], dtype=torch.float64))
        truth.append(torch.as_tensor(w["theta"], dtype=torch.float64))
    data = torch.stack(data).to(device)
    return templates, data, torch.stack(truth).to(device)


def diffusion_noise(shape, sigma, index, seed, device):
    """sigma N(0, 1) of dump window ``index``."""
    generator = torch.Generator().manual_seed(seed + index)
    noise = torch.randn(shape, generator=generator, dtype=torch.float64)
    return (sigma * noise).to(device)


def save(path, out, theta_true, indices, thin, full_chain, attrs):
    chain = out["chain"].permute(1, 0, 2, 3)  # (windows, steps, walkers, 11)
    with h5py.File(path, "w") as f:
        f["samples"] = chain[:, ::thin].numpy()
        if full_chain:
            warmup = out["warmup"].permute(1, 0, 2, 3)
            f["chain"] = torch.cat([warmup, chain], 1).numpy()
        f["theta_true"] = theta_true.cpu().numpy()
        f["window"] = np.asarray(indices)
        for key in (
            "acceptance",
            "step_size",
            "mass_matrix",
            "start_mass_matrix",
            "failed_updates",
            "stuck_walkers",
        ):
            f[key] = out[key].cpu().numpy()
        f.attrs.update(attrs, parameters=list(PARAMETERS))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--windows", required=True, help="dump_windows h5")
    parser.add_argument("--checkpoint", required=True, help="ScoreTask ckpt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--walkers", type=int, default=N_WALKERS)
    parser.add_argument("--warmup", type=int, default=N_WARMUP)
    parser.add_argument("--steps", type=int, default=N_STEPS)
    parser.add_argument("--step_size", type=float, default=STEP_SIZE)
    parser.add_argument("--thin", type=int, default=THIN)
    parser.add_argument("--t_min", type=float, default=T_MIN)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--full_chain", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    with h5py.File(args.windows, "r") as f:
        total = len(f["X"])
    indices = list(range(args.start, min(args.start + args.count, total)))
    templates, data, theta_true = load_windows(args.windows, indices, device)
    net = load_score_net(args.checkpoint, map_location=device)
    sigma = float(VESDE().sigma(torch.tensor(args.t_min, dtype=torch.float64)))
    data = data + torch.stack(
        [
            diffusion_noise(data.shape[1:], sigma, i, args.seed, device)
            for i in indices
        ]
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + SAMPLER_SEED_OFFSET + args.start)

    print(
        f"{len(indices)} windows ({indices[0]}..{indices[-1]}), "
        f"{args.walkers} walkers, {args.warmup}+{args.steps} steps, "
        f"t_min {args.t_min} (sigma {sigma:.6f}), {device}",
        flush=True,
    )
    out = run(
        templates,
        data,
        theta_true,
        network_score(net, args.t_min),
        1 + sigma**2,
        n_walkers=args.walkers,
        step_size=args.step_size,
        n_warmup=args.warmup,
        n_steps=args.steps,
        generator=generator,
    )
    attrs = dict(
        windows_file=str(args.windows),
        checkpoint=str(args.checkpoint),
        t_min=args.t_min,
        sigma=sigma,
        seed=args.seed,
        walkers=args.walkers,
        warmup=args.warmup,
        steps=args.steps,
        initial_step_size=args.step_size,
        thin=args.thin,
        seconds=out["seconds"],
    )
    save(
        args.out,
        out,
        theta_true,
        indices,
        args.thin,
        args.full_chain,
        attrs,
    )
    acceptance = " ".join(f"{a:.3f}" for a in out["acceptance"].tolist())
    print(
        f"{out['seconds']:.0f} s; acceptance {acceptance}; stuck walkers "
        f"{out['stuck_walkers'].tolist()} -> {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
