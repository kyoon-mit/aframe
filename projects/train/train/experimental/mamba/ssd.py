"""Mamba-2 layer in plain PyTorch, with the chunked scan of its state-space
dual (SSD) form.

T. Dao and A. Gu, "Transformers are SSMs: generalized models and efficient
algorithms through structured state space duality", ICML 2024,
arXiv:2405.21060; the scan is their ``ssd_minimal_discrete``. On GPU the
layer uses mamba_ssm's Triton scan (``mamba_chunk_scan_combined``) when the
package is installed (``MAMBA_SKIP_CUDA_BUILD=TRUE uv pip install --no-deps
--no-build-isolation mamba-ssm``; no CUDA build needed), and the plain
PyTorch scan otherwise. Per head h the layer runs the causal recurrence

    state_t = exp(dt_t A_h) state_{t-1} + dt_t B_t x_t,    y_t = C_t state_t + D_h x_t

where dt_t, B_t and C_t are computed from the input, so the filter changes
along the sequence (a "selective" state space model).
"""

import functools
import importlib.util
import math
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


@functools.cache
def triton_scan():
    """mamba_ssm's ``mamba_chunk_scan_combined``, or None without mamba_ssm.
    The package's own __init__ imports its language models and transformers,
    so only its directory is registered and the Triton module is imported
    from it."""
    spec = importlib.util.find_spec("mamba_ssm")
    if spec is None:
        return None
    if "mamba_ssm" not in sys.modules:
        package = types.ModuleType("mamba_ssm")
        package.__path__ = list(spec.submodule_search_locations)
        sys.modules["mamba_ssm"] = package
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    return mamba_chunk_scan_combined


def segsum(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (..., T, T): entry (i, j) is x[j+1] + ... + x[i] for
    j <= i, and -inf for j > i."""
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    below = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    sums = torch.cumsum(x.masked_fill(~below, 0), dim=-2)
    lower = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
    return sums.masked_fill(~lower, -torch.inf)


def ssd_scan(X: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor,
             chunk: int) -> torch.Tensor:
    """y_t = sum over s <= t of C_t . exp(A_{s+1} + ... + A_t) B_s x_s, per head.

    X (b, l, h, p), A (b, l, h), B and C (b, l, n) shared by all heads ->
    (b, l, h, p); the length must be a multiple of ``chunk``.
    """
    b, l, h, p = X.shape
    c, n = l // chunk, B.shape[-1]
    X = X.reshape(b, c, chunk, h, p)
    B, C = B.reshape(b, c, chunk, n), C.reshape(b, c, chunk, n)
    A_cumsum = torch.cumsum(A.reshape(b, c, chunk, h), dim=2)  # (b, c, chunk, h)
    # within each chunk: weight of x_s in y_t is (C_t . B_s) exp(cumsum_t - cumsum_s), s <= t
    gap = A_cumsum[:, :, :, None] - A_cumsum[:, :, None]  # (b, c, t, s, h)
    causal = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool, device=X.device))
    decay = torch.exp(gap.masked_fill(~causal[:, :, None], -torch.inf))
    weights = (C @ B.transpose(-1, -2))[..., None] * decay  # (b, c, t, s, h)
    Y_diag = torch.einsum("bctsh,bcshp->bcthp", weights, X)
    # state at the end of each chunk, then carried across chunks
    to_end = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)  # (b, c, chunk, h)
    states = B.transpose(-1, -2) @ (X * to_end[..., None]).reshape(b, c, chunk, h * p)  # (b, c, n, hp)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    totals = F.pad(A_cumsum[:, :, -1].transpose(1, 2), (1, 0))  # (b, h, c + 1)
    decay_chunk = torch.exp(segsum(totals))  # (b, h, c + 1, c + 1)
    states = torch.einsum("bhzc,bcnhp->bzhpn", decay_chunk, states.reshape(b, c + 1, n, h, p))[:, :-1]
    # carried states read out within each chunk
    Y_off = torch.einsum("bctn,bchpn->bcthp", C, states) * torch.exp(A_cumsum)[..., None]
    return (Y_diag + Y_off).reshape(b, l, h, p)


class GatedRMSNorm(nn.Module):
    """RMS norm of y * silu(z), with a learned scale (Mamba-2's output norm)."""

    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = y * F.silu(z)
        return y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps) * self.weight


class Mamba2(nn.Module):
    """One Mamba-2 mixer, (B, L, d_model) -> (B, L, d_model), one group of
    B and C shared by all heads. Defaults are those of the reference
    implementation except ``d_state``."""

    def __init__(self, d_model: int, d_state: int = 64, expand: int = 2, headdim: int = 64,
                 d_conv: int = 4, chunk: int = 64, dt_min: float = 1e-3, dt_max: float = 0.1,
                 dt_floor: float = 1e-4, A_init_range: tuple[float, float] = (1.0, 16.0),
                 triton: bool = True):
        super().__init__()
        self.triton = triton
        d_inner = expand * d_model
        if d_inner % headdim:
            raise ValueError(f"expand * d_model = {d_inner} is not a multiple of headdim {headdim}")
        self.d_inner, self.d_state, self.headdim = d_inner, d_state, headdim
        self.nheads = d_inner // headdim
        self.chunk = chunk
        self.in_proj = nn.Linear(d_model, 2 * d_inner + 2 * d_state + self.nheads, bias=False)
        conv_dim = d_inner + 2 * d_state
        self.conv1d = nn.Conv1d(conv_dim, conv_dim, d_conv, groups=conv_dim, padding=d_conv - 1)
        dt = torch.exp(torch.rand(self.nheads) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus inverse
        A = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = GatedRMSNorm(d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        b, l, _ = u.shape
        z, xBC, dt = self.in_proj(u).split(
            [self.d_inner, self.d_inner + 2 * self.d_state, self.nheads], dim=-1)
        xBC = F.silu(self.conv1d(xBC.transpose(1, 2))[..., :l].transpose(1, 2))
        x, B, C = xBC.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        x = x.reshape(b, l, self.nheads, self.headdim)
        A = -torch.exp(self.A_log)  # (h,)
        scan = triton_scan() if self.triton and u.is_cuda else None
        if scan is not None:
            y = scan(x, dt, A, B[:, :, None], C[:, :, None], self.chunk, D=self.D,
                     dt_bias=self.dt_bias, dt_softplus=True)
        else:
            dt = F.softplus(dt + self.dt_bias)  # (b, l, h)
            y = ssd_scan(x * dt[..., None], A * dt, B, C, self.chunk)
            y = y + x * self.D[:, None]
        y = self.norm(y.reshape(b, l, self.d_inner), z)
        return self.out_proj(y)
