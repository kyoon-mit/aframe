from typing import Literal, Optional

from architectures import Architecture, JaxArchitecture
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


class SupervisedTimeDomainLinOSS(JaxArchitecture):
    from architectures.networks.linoss import LinOSS

    linoss: LinOSS

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
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.linoss(X, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X, state, key=key)


class SupervisedTimeDomainOriginalLinOSS(JaxArchitecture):
    """Wrapper around OriginalLinOSS for supervised GW detection.

    Accepts the standard aframe input shape (num_ifos, time), transposes to
    (time, num_ifos), and delegates to OriginalLinOSS which returns a scalar
    detection logit.
    """

    from architectures.networks.original_linoss import OriginalLinOSS

    linoss: OriginalLinOSS

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        hidden_dim: int = 64,
        num_blocks: int = 6,
        ssm_size: int = 64,
        discretization: str = "IM",
        drop_rate: float = 0.05,
        conv_type: str = "none",
        conv_kernel_size: int = 4,
        conv_expansion: int = 2,
        use_mlp_head: bool = False,
        mlp_hidden_dim: int = 64,
        mlp_depth: int = 3,
        norm_pos: str = "pre",
        norm_type: str = "batch",
        encoder_type: str = "linear",
        encoder_num_layers: int = 2,
        encoder_kernel_size: int = 8,
        patch_size: int = 16,
        encoder_kernel_sizes: Optional[list] = None,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr
        from architectures.networks.original_linoss import OriginalLinOSS

        self.linoss = OriginalLinOSS(
            N=num_ifos,
            H=hidden_dim,
            num_blocks=num_blocks,
            ssm_size=ssm_size,
            discretization=discretization,
            drop_rate=drop_rate,
            conv_type=conv_type,
            conv_kernel_size=conv_kernel_size,
            conv_expansion=conv_expansion,
            use_mlp_head=use_mlp_head,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_depth=mlp_depth,
            key=jr.PRNGKey(seed),
            norm_pos=norm_pos,
            norm_type=norm_type,
            encoder_type=encoder_type,
            encoder_num_layers=encoder_num_layers,
            encoder_kernel_size=encoder_kernel_size,
            patch_size=patch_size,
            encoder_kernel_sizes=encoder_kernel_sizes,
        )

    def __call__(self, X, state, key=None):
        # X: (num_ifos, time) → transpose to (time, num_ifos)
        return self.linoss(X.T, state, key=key)

    def forward(self, X, state, key=None):
        return self.linoss(X.T, state, key=key)


class SupervisedHeterodyneLinOSS(JaxArchitecture):
    """JAX/Equinox architecture for heterodyne input via SSM channel attention.

    Each interferometer has ``num_chirp_masses`` channels (one per heterodyne
    template). A shared small LinOSS encodes each channel; the embeddings
    drive a cross-attention that reduces the N channels to ``n_out`` weighted
    combinations. Those ``num_ifos × n_out`` timeseries are then processed by
    the standard LinOSS detection backbone.
    """

    from architectures.networks.heterodyne_linoss import HeterodyneLinOSS

    model: HeterodyneLinOSS

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_chirp_masses: int,
        encoder_d_model: int = 32,
        encoder_state_dim: int = 32,
        n_out: int = 4,
        d_k: int = 32,
        d_v: int = 8,
        temporal_kernel_size: int = 8,
        temporal_stride: int = 8,
        encoder_r_min: float = 0.9,
        encoder_theta_max: float = 3.14159265359,
        time_hidden_dim: int = 32,
        time_num_blocks: int = 3,
        time_dropout_rate: float = 0.0,
        time_state_dim: int = 32,
        time_r_min: float = 0.9,
        time_theta_max: float = 3.14159265359,
        time_num_res_blocks: int = 0,
        time_conv_kernel_size: int = 4,
        time_conv_position: str = "pre",
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        resnet_kernel_size: int = 7,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr
        from architectures.networks.heterodyne_linoss import HeterodyneLinOSS

        self.model = HeterodyneLinOSS(
            num_ifos=num_ifos,
            num_chirp_masses=num_chirp_masses,
            encoder_d_model=encoder_d_model,
            encoder_state_dim=encoder_state_dim,
            n_out=n_out,
            d_k=d_k,
            d_v=d_v,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            encoder_r_min=encoder_r_min,
            encoder_theta_max=encoder_theta_max,
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
            key=jr.PRNGKey(seed),
        )

    def __call__(self, X, state, key=None):
        return self.model(X, state, key)

    def forward(self, X, state, key=None):
        return self.model(X, state, key)


class SupervisedHeterodynePoolingLinOSS(JaxArchitecture):
    """Heterodyne LinOSS with pooling backbone + conv ResNet head.

    Feeds all num_ifos * num_chirp_masses heterodyned channels directly into
    PoolingLinossHeavyBackbone (no attention reduction). A conv ResNet1D then
    reads out a scalar logit from the downsampled sequence.

    Input shape: (num_ifos * num_chirp_masses, time)
    """

    from architectures.networks.linoss import LinOSS

    linoss: "LinOSS"

    def __init__(
        self,
        num_ifos: int,
        sample_rate: float,
        kernel_length: float,
        num_chirp_masses: int,
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
        resnet_kernel_size: int = 7,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        *,
        seed: int = 0,
    ) -> None:
        import jax.random as jr
        from architectures.networks.linoss import LinOSS

        self.linoss = LinOSS(
            time_in_features=num_ifos * num_chirp_masses,
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
