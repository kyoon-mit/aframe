from typing import Literal, Optional

import torch
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.ssm.s4d import S4Model

from architectures import Architecture
from architectures.base import JaxArchitecture
from architectures.networks.s4d_variants import (
    S4ModelDenoiseRegress,
    S4ModelResNetMLPDecoder,
    S4ModelSeq2Seq,
)


class RegressionArchitecture(Architecture):
    """
    Base class for regression architectures.
    Forward pass returns a tensor of shape ``(N, num_params)``.
    """

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class MultiTaskArchitecture(Architecture):
    """
    Base class for multi-task architectures.
    Forward pass returns ``(logits, param_estimates)`` where
    ``logits`` has shape ``(N, 1)`` and ``param_estimates``
    has shape ``(N, num_params)``.
    """

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class RegressionTimeDomainResNet(ResNet1D, RegressionArchitecture):
    """
    ResNet1D backbone for injection parameter regression.
    Outputs a tensor of shape ``(N, num_params)``.

    Args:
        num_params:
            Number of parameters to regress. Must match the length
            of ``param_names`` in the model config.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_params: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos,
            layers=layers,
            classes=num_params,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class RegressionTimeDomainS4DenoiseRegress(RegressionArchitecture):
    """S4D denoiser (seq-to-seq) feeding an S4D regressor.

    ``forward`` returns ``(x_denoised, param_estimates)``: a cleaned I/Q
    sequence ``(N, num_ifos, L)`` and the regression head
    ``(N, d_output)``. Pairs with ``DenoisedGaussianNLLRegression``.
    """

    def __init__(
        self,
        num_ifos: int,
        d_output: int = 2,
        denoiser_d_model: int = 128,
        denoiser_d_state: int = 64,
        denoiser_n_layers: int = 4,
        denoiser_dropout: float = 0.2,
        regressor_d_model: int = 128,
        regressor_d_state: int = 64,
        regressor_n_layers: int = 4,
        regressor_dropout: float = 0.2,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        detach_denoiser: bool = False,
        # linked from the CLI but unused here
        sample_rate: Optional[float] = None,
        kernel_length: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = S4ModelDenoiseRegress(
            denoiser=S4ModelSeq2Seq,
            regressor=S4Model,
            denoiser_params={
                "d_input": num_ifos,
                "d_output": num_ifos,
                "d_model": denoiser_d_model,
                "d_state": denoiser_d_state,
                "n_layers": denoiser_n_layers,
                "dropout": denoiser_dropout,
                "dt_min": dt_min,
                "dt_max": dt_max,
            },
            regressor_params={
                "d_input": num_ifos,
                "d_output": d_output,
                "d_model": regressor_d_model,
                "d_state": regressor_d_state,
                "n_layers": regressor_n_layers,
                "dropout": regressor_dropout,
                "dt_min": dt_min,
                "dt_max": dt_max,
            },
            detach_denoiser=detach_denoiser,
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)


class RegressionTimeDomainS4DenoiseRegressResNetMLP(RegressionArchitecture):
    """S4D seq-to-seq denoiser feeding an S4D + ResNet1D/MLP regressor.

    Same as ``RegressionTimeDomainS4DenoiseRegress`` but the regressor head
    is ``S4ModelResNetMLPDecoder`` (S4 layers -> ResNet1D -> MLP) instead of
    mean-pool + linear. ``forward`` returns ``(x_denoised, param_estimates)``.
    """

    def __init__(
        self,
        num_ifos: int,
        d_output: int = 2,
        denoiser_d_model: int = 128,
        denoiser_d_state: int = 64,
        denoiser_n_layers: int = 4,
        denoiser_dropout: float = 0.2,
        regressor_d_model: int = 128,
        regressor_d_state: int = 64,
        regressor_n_layers: int = 4,
        regressor_dropout: float = 0.2,
        regressor_prenorm: bool = False,
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        detach_denoiser: bool = False,
        sample_rate: Optional[float] = None,
        kernel_length: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model = S4ModelDenoiseRegress(
            denoiser=S4ModelSeq2Seq,
            regressor=S4ModelResNetMLPDecoder,
            denoiser_params={
                "d_input": num_ifos,
                "d_output": num_ifos,
                "d_model": denoiser_d_model,
                "d_state": denoiser_d_state,
                "n_layers": denoiser_n_layers,
                "dropout": denoiser_dropout,
                "dt_min": dt_min,
                "dt_max": dt_max,
            },
            regressor_params={
                "d_input": num_ifos,
                "d_output": d_output,
                "d_model": regressor_d_model,
                "d_state": regressor_d_state,
                "n_layers": regressor_n_layers,
                "dropout": regressor_dropout,
                "prenorm": regressor_prenorm,
                "resnet_layers": resnet_layers,
                "resnet_latent_dim": resnet_latent_dim,
                "mlp_width": mlp_width,
                "mlp_depth": mlp_depth,
                "dt_min": dt_min,
                "dt_max": dt_max,
            },
            detach_denoiser=detach_denoiser,
        )

    def forward(self, X: torch.Tensor):
        return self.model(X)


class MultiTaskTimeDomainResNet(MultiTaskArchitecture):
    """
    Shared ResNet1D backbone with separate classification and regression heads.
    Returns ``(logits, param_estimates)`` where logits has shape ``(N, 1)``
    and param_estimates has shape ``(N, num_params)``.

    Args:
        embedding_dim:
            Dimensionality of the shared backbone's output embedding,
            i.e. the input size to both heads.
        num_params:
            Number of parameters to regress. Must match the length
            of ``param_names`` in the model config.
    """

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_params: int,
        layers: list[int],
        embedding_dim: int = 512,
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__()
        self.backbone = ResNet1D(
            num_ifos,
            layers=layers,
            classes=embedding_dim,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )
        self.clf_head = torch.nn.Linear(embedding_dim, 1)
        self.reg_head = torch.nn.Linear(embedding_dim, num_params)

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(X)
        return self.clf_head(h), self.reg_head(h)


try:
    import equinox as eqx
    import jax.random as jr

    from architectures.networks.linoss import LinOSSModel

    class RegressionTimeDomainLinOSS(JaxArchitecture, eqx.Module):
        """JAX LinOSS regression architecture.

        Wraps :class:`LinOSSModel` with a ``num_ifos``-first signature so
        the LightningCLI can link it from the datamodule. Forward returns
        ``(logits, state)`` where logits has shape ``(d_output,)``.
        """

        model: LinOSSModel

        def __init__(
            self,
            num_ifos: int,
            d_output: int = 2,
            d_model: int = 64,
            d_state: int = 64,
            n_layers: int = 4,
            dropout: float = 0.2,
            r_min: float = 0.9,
            theta_max: float = 3.14159265359,
            seed: int = 0,
            sample_rate: float = None,
            kernel_length: float = None,
        ):
            key = jr.PRNGKey(seed)
            self.model = LinOSSModel(
                d_input=num_ifos,
                d_output=d_output,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                dropout=dropout,
                key=key,
                r_min=r_min,
                theta_max=theta_max,
            )

        def __call__(self, x, state, key=None):
            return self.model(x, state, key=key)

except ImportError:

    class RegressionTimeDomainLinOSS(JaxArchitecture):
        """Stub raised when JAX/equinox are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "JAX and equinox are required for RegressionTimeDomainLinOSS. "
                "Install them with: uv sync --extra jax"
            )


try:
    import equinox as _eqx
    import jax.random as _jr

    from architectures.networks.linoss_variants import (
        LinOSSModelResNetMLPDecoder,
    )

    class RegressionTimeDomainLinOSSResNetMLPDecoder(
        JaxArchitecture, _eqx.Module
    ):
        """JAX LinOSS with a ResNet1D + MLP readout head."""

        model: LinOSSModelResNetMLPDecoder

        def __init__(
            self,
            num_ifos: int,
            d_output: int = 2,
            d_model: int = 64,
            d_state: int = 64,
            n_layers: int = 4,
            dropout: float = 0.2,
            r_min: float = 0.9,
            theta_max: float = 3.14159265359,
            seed: int = 0,
            sample_rate: float = None,
            kernel_length: float = None,
        ):
            self.model = LinOSSModelResNetMLPDecoder(
                d_input=num_ifos,
                d_output=d_output,
                d_model=d_model,
                d_state=d_state,
                n_layers=n_layers,
                dropout=dropout,
                key=_jr.PRNGKey(seed),
                r_min=r_min,
                theta_max=theta_max,
            )

        def __call__(self, x, state, key=None):
            return self.model(x, state, key=key)

except ImportError:

    class RegressionTimeDomainLinOSSResNetMLPDecoder(JaxArchitecture):
        """Stub raised when JAX/equinox are not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "JAX and equinox are required. uv sync --extra jax"
            )
