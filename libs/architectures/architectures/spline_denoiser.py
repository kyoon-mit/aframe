"""S4D denoiser whose output is confined to a smooth B-spline basis.

The free-sample denoiser can emit anything, including a one-sample impulse at
the window edge, which is exactly the degenerate solution the magnitude
spectrum losses were measured to select. Expanding the output on a cubic
B-spline basis with knots every ``stride`` samples makes that solution
inexpressible: nothing shorter than the knot spacing can be represented.

It also reduces the output dimension from L to about L/stride coefficients per
channel, which is a large variance reduction on a target (a chirp) that is
genuinely smooth, so little modelling capacity is given up.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from architectures import Architecture
from architectures.networks.s4d_variants import S4ModelSeq2Seq


def cubic_bspline_basis(length: int, stride: int) -> torch.Tensor:
    """Uniform cubic B-spline basis, shape ``(n_basis, length)``.

    Knots are spaced ``stride`` samples apart, with two extra control points
    at each end so the span is covered out to the boundary. Each row is one
    basis function evaluated on the sample grid; a curve is a weighted sum of
    the rows, so the weights are the coefficients the network predicts.
    """
    n_basis = length // stride + 3
    t = torch.arange(length, dtype=torch.float32)
    # centre of control point i, in samples
    centres = (torch.arange(n_basis, dtype=torch.float32) - 1.0) * stride
    # distance in knot units
    u = (t.unsqueeze(0) - centres.unsqueeze(1)) / stride
    a = u.abs()
    # cubic B-spline: support |u| < 2
    basis = torch.zeros_like(a)
    m1 = a < 1.0
    m2 = (a >= 1.0) & (a < 2.0)
    basis[m1] = (4.0 - 6.0 * a[m1] ** 2 + 3.0 * a[m1] ** 3) / 6.0
    basis[m2] = ((2.0 - a[m2]) ** 3) / 6.0
    return basis


class SplineS4Denoiser(Architecture):
    """S4D sequence model that predicts smooth B-spline coefficients.

    The S4D backbone runs at full resolution, its output is pooled down to one
    vector per control point, a linear layer turns that into per-channel
    coefficients, and the fixed basis expands them back to a full-length
    waveform. Only the coefficients are learned; the basis is constant.

    Args:
        num_ifos: number of detectors, so the input and output channels.
        kernel_length: analysis window in seconds, used with sample_rate to
            size the basis.
        sample_rate: samples per second.
        stride: knot spacing in samples. Larger is smoother and cheaper, and
            sets the shortest feature the model can express.
        d_model, d_state, n_layers, dropout, prenorm, num_groups, dt_min,
            dt_max: passed through to the S4D backbone.
    """

    def __init__(
        self,
        num_ifos: int,
        kernel_length: float = 4.0,
        sample_rate: float = 2048.0,
        stride: int = 16,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        prenorm: bool = False,
        num_groups: Optional[int] = None,
        dt_min: float = 1e-3,
        dt_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_ifos = num_ifos
        self.stride = stride
        length = int(kernel_length * sample_rate)
        basis = cubic_bspline_basis(length, stride)
        # constant, but must follow the module to its device
        self.register_buffer("basis", basis, persistent=False)
        self.n_basis = basis.shape[0]

        self.model = S4ModelSeq2Seq(
            d_input=num_ifos,
            d_output=d_model,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            num_groups=num_groups,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.to_coeff = torch.nn.Linear(d_model, num_ifos)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # (B, C, L) -> S4D features (B, d_model, L)
        h = self.model(X)
        # one feature vector per control point
        h = F.adaptive_avg_pool1d(h, self.n_basis)
        # (B, n_basis, d_model) -> (B, n_basis, C) -> (B, C, n_basis)
        coeff = self.to_coeff(h.transpose(1, 2)).transpose(1, 2)
        # expand on the fixed basis: (B, C, n_basis) @ (n_basis, L)
        return coeff @ self.basis
