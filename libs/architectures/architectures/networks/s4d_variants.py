import torch
from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.nn.ssm.s4d import S4Model
from torch import nn


class S4ModelPrenorm(S4Model):
    """S4Model with pre-norm residual blocks (norm before each S4D layer)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts, strict=True
        ):
            z = norm(x.transpose(-1, -2)).transpose(-1, -2)
            z = dropout(layer(z))
            x = x + z
        x = x.transpose(-1, -2)
        x = x.mean(dim=1)
        return self.decoder(x)


class S4ModelResNetMLPDecoder(S4Model):
    """S4Model backbone with a ResNet1D + MLP readout head in place of the
    mean-pool + linear decoder."""

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 256,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        prenorm: bool = False,
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__(
            d_input=d_input,
            d_output=d_output,
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dropout=dropout,
            dt_min=dt_min,
            dt_max=dt_max,
        )
        self.prenorm = prenorm
        self.resnet = ResNet1D(
            in_channels=d_model,
            layers=list(resnet_layers),
            classes=resnet_latent_dim,
        )
        layers: list[nn.Module] = []
        width = resnet_latent_dim
        for _ in range(mlp_depth):
            layers += [nn.Linear(width, mlp_width), nn.GELU()]
            width = mlp_width
        layers += [nn.Linear(width, d_output)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(-1, -2)
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(
            self.s4_layers, self.norms, self.dropouts, strict=True
        ):
            if self.prenorm:
                z = norm(x.transpose(-1, -2)).transpose(-1, -2)
                z = dropout(layer(z))
                x = x + z
            else:
                z = dropout(layer(x))
                x = norm((z + x).transpose(-1, -2)).transpose(-1, -2)
        h = self.resnet(x)
        return self.mlp(h)
