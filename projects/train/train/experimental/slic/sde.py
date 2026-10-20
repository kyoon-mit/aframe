"""Variance exploding diffusion, its training loss and the weight average,
ported from scoregen_jax (sde.py, losses.py)."""

import torch


class VESDE:
    """Noise level sigma(t) = sigma_min * (sigma_max / sigma_min) ** t."""

    def __init__(self, sigma_min: float = 1e-3, sigma_max: float = 1e4):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sigma(self, t: torch.Tensor) -> torch.Tensor:
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t


def denoising_loss(
    net: torch.nn.Module, x: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """sum((net(x + sigma(t) z, t) + z) ** 2) / batch, t ~ U(eps, 1).

    ``net`` has an ``sde`` and maps (windows, t) to its output.
    """
    t = torch.rand(len(x), device=x.device, dtype=x.dtype) * (1 - eps) + eps
    z = torch.randn_like(x)
    sigma = net.sde.sigma(t)[:, None, None]
    output = net(x + sigma * z, t)
    return (output + z).square().sum() / len(x)


@torch.no_grad()
def ema_update(average: list, params: list, rate: float) -> None:
    """average = average * rate + params * (1 - rate), in place."""
    torch._foreach_mul_(average, rate)
    torch._foreach_add_(average, params, alpha=1 - rate)
