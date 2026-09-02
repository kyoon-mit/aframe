from typing import Optional

import torch

from architectures import Architecture
from architectures.networks.s4d_variants import S4ModelSeq2Seq


class TimeDomainS4Denoiser(Architecture):
    """S4D sequence-to-sequence denoiser, with no downstream head.

    Takes whitened strain ``(B, num_ifos, L)`` and returns a denoised strain
    of the same shape. Unlike the denoise-and-classify architectures this
    returns a single tensor rather than a tuple, so the training task sees
    only the reconstruction and nothing pulls the representation toward a
    detection statistic.

    Args:
        num_ifos: number of interferometers, i.e. input and output channels.
        d_model: width of the S4D stack.
        d_state: SSM state dimension per channel.
        n_layers: number of S4D blocks.
        dropout: dropout applied inside each block.
        prenorm: normalise before the block rather than after.
        num_groups: use GroupNorm with this many groups instead of LayerNorm.
        dt_min: lower bound of the sampled S4D timescale.
        dt_max: upper bound of the sampled S4D timescale.
    """

    def __init__(
        self,
        num_ifos: int,
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
        self.model = S4ModelSeq2Seq(
            d_input=num_ifos,
            d_output=num_ifos,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            num_groups=num_groups,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.model(X)
