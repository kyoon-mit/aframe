from typing import Optional

import torch
import torch.nn as nn

from architectures import Architecture
from architectures.networks.s4d_variants import S4ModelSeq2Seq


class S4ModelSeq2SeqFiLM(S4ModelSeq2Seq):
    """S4D seq-to-seq with feature-wise linear modulation on every block.

    Each block's output becomes ``gamma * x + beta``, with gamma and beta
    produced per channel per layer from a conditioning vector (Perez et
    al., AAAI 2018). With no conditioning the modulation is the identity.

    Args:
        cond_dim: length of the conditioning vector. 0 disables FiLM and
            makes this class identical to its parent.
        cond_hidden: width of the network that produces gamma and beta.
    """

    def __init__(
        self,
        *args,
        cond_dim: int = 0,
        cond_hidden: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cond_dim = cond_dim
        if cond_dim:
            d_model = self.encoder.out_features
            n_layers = len(self.s4_layers)
            self.film = nn.Sequential(
                nn.Linear(cond_dim, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, 2 * n_layers * d_model),
            )
            # start at the identity: gamma = 1, beta = 0
            nn.init.zeros_(self.film[-1].weight)
            nn.init.zeros_(self.film[-1].bias)

    def forward(
        self, x: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, d_input, L)
            cond: (B, cond_dim), or None for the unconditioned pass.

        Returns:
            (B, d_output, L)
        """
        gammas = betas = None
        if self.cond_dim and cond is not None:
            n_layers = len(self.s4_layers)
            d_model = self.encoder.out_features
            film = self.film(cond).view(-1, n_layers, 2, d_model)
            gammas = 1.0 + film[:, :, 0]  # (B, n_layers, d_model)
            betas = film[:, :, 1]

        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)  # (B, d_model, L)
        for index, (layer, norm, dropout) in enumerate(
            zip(self.s4_layers, self.norms, self.dropouts, strict=True)
        ):
            if self.prenorm:
                z = self._apply_norm(norm, x)
                z = dropout(layer(z))
                x = x + z
            else:
                z = dropout(layer(x))
                x = self._apply_norm(norm, z + x)
            if gammas is not None:
                gamma = gammas[:, index].unsqueeze(-1)  # (B, d_model, 1)
                beta = betas[:, index].unsqueeze(-1)
                x = gamma * x + beta
        x = x.transpose(-1, -2)
        x = self.decoder(x)
        return x.transpose(-1, -2)


class TimeDomainS4DenoiserFiLM(Architecture):
    """``TimeDomainS4Denoiser`` with FiLM-modulated blocks.

    Same arguments as ``TimeDomainS4Denoiser`` plus the two below.
    ``forward`` takes an optional conditioning vector.

    Args:
        cond_dim: length of the conditioning vector. 0 disables FiLM.
        cond_hidden: width of the network that produces the modulation.
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
        cond_dim: int = 0,
        cond_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.model = S4ModelSeq2SeqFiLM(
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
            cond_dim=cond_dim,
            cond_hidden=cond_hidden,
        )

    def forward(
        self, X: torch.Tensor, cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(X, cond)
