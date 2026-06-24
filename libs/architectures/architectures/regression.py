from typing import Literal, Optional

import torch
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D

from architectures import Architecture
from architectures.base import JaxArchitecture


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


class RegressionTimeDomainLinOSS(JaxArchitecture):
    """JAX/Equinox LinOSS architecture for BNS parameter regression.

    Wraps ``LinOSS`` with ``d_output=2`` (chirp-mass mean + pre-Softplus
    log-variance) so it can be used with ``JaxRegressionAframe``.
    """

    linoss: "LinOSS"  # noqa: F821

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        time_in_features: int,
        time_hidden_dim: int,
        time_num_blocks: int,
        time_dropout_rate: float,
        time_state_dim: int,
        time_r_min: float,
        time_theta_max: float,
        time_num_res_blocks: int,
        time_conv_kernel_size: int,
        time_conv_position: str = "pre",
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        resnet_kernel_size: int = 21,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        d_output: int = 2,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr
        from architectures.networks.linoss import LinOSS

        self.linoss = LinOSS(
            time_in_features=time_in_features,
            time_hidden_dim=time_hidden_dim,
            time_num_blocks=time_num_blocks,
            time_dropout_rate=time_dropout_rate,
            time_state_dim=time_state_dim,
            time_r_min=time_r_min,
            time_theta_max=time_theta_max,
            time_num_res_blocks=time_num_res_blocks,
            time_conv_kernel_size=time_conv_kernel_size,
            time_conv_position=time_conv_position,
            resnet_layers=resnet_layers,
            resnet_latent_dim=resnet_latent_dim,
            resnet_kernel_size=resnet_kernel_size,
            resnet_norm_groups=resnet_norm_groups,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            d_output=d_output,
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.linoss(X, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X, state, key=key)


class RegressionTimeDomainLinOSS2(JaxArchitecture):
    """JAX/Equinox LinOSS2 architecture for BNS parameter regression.

    Wraps the generalized ``LinOSS2`` (ported from discretax PR #74) with
    ``d_output=2`` (chirp-mass mean + pre-Softplus log-variance) so it can be
    used with ``JaxRegressionAframe``.

    Compared to ``RegressionTimeDomainLinOSS``, the sequence mixer additionally
    exposes the discretization, initialization, stability projection,
    multi-head and input-normalization options of the LinOSS2 mixer.
    """

    linoss: "LinOSS2"  # noqa: F821

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        time_in_features: int,
        time_hidden_dim: int,
        time_num_blocks: int,
        time_dropout_rate: float,
        time_state_dim: int,
        time_r_min: float,
        time_theta_max: float,
        time_num_res_blocks: int,
        time_conv_kernel_size: int,
        time_conv_position: str = "pre",
        time_discretization: str = "IMEX",
        time_initialization: str = "AG",
        time_damping: bool = True,
        time_stability: str = "oscillatory",
        time_projection_eps: float = 0.0,
        time_input_normalization: bool = False,
        time_num_heads: int = 1,
        time_use_head_output_projection: bool = False,
        time_A_max: float = 1.0,
        time_G_max: float = 1.0,
        dtype: str = "float32",
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        resnet_kernel_size: int = 21,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        d_output: int = 2,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr

        from architectures.networks.linoss2 import LinOSS2

        self.linoss = LinOSS2(
            time_in_features=time_in_features,
            time_hidden_dim=time_hidden_dim,
            time_num_blocks=time_num_blocks,
            time_dropout_rate=time_dropout_rate,
            time_state_dim=time_state_dim,
            time_r_min=time_r_min,
            time_theta_max=time_theta_max,
            time_num_res_blocks=time_num_res_blocks,
            time_conv_kernel_size=time_conv_kernel_size,
            time_conv_position=time_conv_position,
            time_discretization=time_discretization,
            time_initialization=time_initialization,
            time_damping=time_damping,
            time_stability=time_stability,
            time_projection_eps=time_projection_eps,
            time_input_normalization=time_input_normalization,
            time_num_heads=time_num_heads,
            time_use_head_output_projection=time_use_head_output_projection,
            time_A_max=time_A_max,
            time_G_max=time_G_max,
            dtype=dtype,
            resnet_layers=resnet_layers,
            resnet_latent_dim=resnet_latent_dim,
            resnet_kernel_size=resnet_kernel_size,
            resnet_norm_groups=resnet_norm_groups,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            d_output=d_output,
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.linoss(X, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X, state, key=key)
