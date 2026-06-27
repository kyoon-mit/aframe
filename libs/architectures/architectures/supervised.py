from typing import Literal, Optional

from architectures import Architecture
from architectures.base import JaxArchitecture
from architectures.networks import S4Model, WaveNet, Xylophone
from jaxtyping import Float
from ml4gw.nn.resnet.resnet_1d import NormLayer, ResNet1D
from ml4gw.nn.resnet.resnet_2d import ResNet2D
from torch import Tensor
import torch


class SupervisedArchitecture(Architecture):
    """
    Dummy class for registering available architectures
    for supervised learning problems. Supervised architectures
    are expected to return a single, real-valued logit
    corresponding to a detection statistic.
    """

    def forward(
        self, X: Float[Tensor, "batch channels ..."]
    ) -> Float[Tensor, " batch"]:
        raise NotImplementedError


class SupervisedTimeDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
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
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedFrequencyDomainResNet(ResNet1D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
    ) -> None:
        super().__init__(
            num_ifos * 2,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainXylophone(Xylophone, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            classes=1,
            norm_layer=norm_layer,
        )


class SupervisedTimeDomainWaveNet(WaveNet, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        res_channels: int,
        layers_per_block: int,
        num_blocks: int,
        kernel_size: int = 2,
        norm_layer: Optional[NormLayer] = None,
    ):
        super().__init__(
            num_ifos,
            res_channels,
            layers_per_block,
            num_blocks,
            kernel_size=kernel_size,
            norm_layer=norm_layer,
        )


class SupervisedSpectrogramDomainResNet(ResNet2D, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
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
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )


class SupervisedS4Model(S4Model, SupervisedArchitecture):
    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        d_output: int = 1,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
        prenorm: bool = True,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        lr: Optional[float] = None,
    ) -> None:
        length = int(kernel_length * sample_rate)
        super().__init__(
            length=length,
            d_input=num_ifos,
            d_output=d_output,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            prenorm=prenorm,
            dt_min=dt_min,
            dt_max=dt_max,
            lr=lr,
        )


class SupervisedMultiModalResNet(SupervisedArchitecture):
    """
    MultiModal embedding network that embeds time, frequency, and PSD data.
    We pass the data through their own ResNets defined by their layers
    and context dims, then concatenate the output embeddings.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        freq_classes: int,
        time_layers: list[int],
        freq_layers: list[int],
        time_kernel_size: int = 3,
        freq_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.freq_psd_resnet = ResNet1D(
            in_channels=int(num_ifos * 3),
            layers=freq_layers,
            classes=freq_classes,
            kernel_size=freq_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

        self.classifier = torch.nn.Linear(time_classes + freq_classes, 1)

    def forward(self, X, X_fft):
        time_domain_output = self.time_domain_resnet(X)
        freq_domain_output = self.freq_psd_resnet(X_fft)
        concat = torch.cat([time_domain_output, freq_domain_output], dim=-1)
        return self.classifier(concat)


class SupervisedTimeSpectrogramResNet(SupervisedArchitecture):
    """
    Spectrogram and Time Domain ResNet that processes a combination of
    timeseries and spectrogram image data.
    """

    def __init__(
        self,
        num_ifos: int,
        time_classes: int,
        spec_classes: int,
        time_layers: list[int],
        spec_layers: list[int],
        time_kernel_size: int = 3,
        spec_kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        time_norm_layer: Optional[NormLayer] = None,
        spec_norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ):
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos,
            layers=time_layers,
            classes=time_classes,
            kernel_size=time_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=time_norm_layer,
        )

        self.spectrogram_resnet = ResNet2D(
            in_channels=num_ifos,
            layers=spec_layers,
            classes=spec_classes,
            kernel_size=spec_kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=spec_norm_layer,
        )

    def forward(self, X, X_spec):
        time_domain_output = self.time_domain_resnet(X)
        spec_domain_output = self.spectrogram_resnet(X_spec)
        return time_domain_output, spec_domain_output


class SupervisedTimeDomainOriginalLinOSS(JaxArchitecture):
    """JAX/Equinox OriginalLinOSS architecture for binary classification.

    Wraps the plain ``OriginalLinOSS`` model (faithful port of discretax's
    ``LinOSS``: linear encoder + stacked LinOSS ``StandardBlock`` blocks + MLP
    head, with **no** pooling backbone and **no** intermediate ResNet).

    The sequence mixer is the generalized real-pair LinOSS mixer from discretax
    PR #74, exposing the full feature set: IM/IMEX/IMEX2/IMEX3/EX
    discretizations, AG/RT initialization, oscillatory/stable stability
    projection, multi-head mixing with optional output projection, LRU-style
    input normalization and a configurable compute dtype.

    ``d_output=1`` produces a single detection logit so it can be used with
    ``JaxClassificationAframe``.
    """

    linoss: "OriginalLinOSS"  # noqa: F821

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
        time_prenorm: bool = True,
        dtype: str = "float32",
        mlp_width: int = 64,
        mlp_depth: int = 2,
        d_output: int = 1,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr

        from architectures.networks.original_linoss import OriginalLinOSS

        self.linoss = OriginalLinOSS(
            time_in_features=time_in_features,
            time_hidden_dim=time_hidden_dim,
            time_num_blocks=time_num_blocks,
            time_dropout_rate=time_dropout_rate,
            time_state_dim=time_state_dim,
            time_r_min=time_r_min,
            time_theta_max=time_theta_max,
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
            time_prenorm=time_prenorm,
            dtype=dtype,
            mlp_width=mlp_width,
            mlp_depth=mlp_depth,
            d_output=d_output,
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.linoss(X, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X, state, key=key)


class SupervisedHeterodyneTimeDomainResNet(SupervisedArchitecture):
    """
    Time Domain ResNet that processes a Heterodyned timeseries.

    Args:
        num_chirp_masses (int):
            Number of chirp masses used to define the input channel
            dimension (in_channels = num_ifos x num_chirp_masses).
    """

    def __init__(
        self,
        num_ifos: int,
        num_chirp_masses: int,
        layers: list[int],
        kernel_size: int = 3,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        stride_type: Optional[list[Literal["stride", "dilation"]]] = None,
        norm_layer: Optional[NormLayer] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.time_domain_resnet = ResNet1D(
            in_channels=num_ifos * num_chirp_masses,
            layers=layers,
            classes=1,
            kernel_size=kernel_size,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            stride_type=stride_type,
            norm_layer=norm_layer,
        )

    def forward(self, X):
        return self.time_domain_resnet(X)
