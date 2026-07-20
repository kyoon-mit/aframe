"""LinOSS with a ResNet1D + MLP readout head (JAX/equinox).

The oscillator backbone is the lean prenorm LinOSS stack; the readout is
the ResNet1D + MLP head from the original heavy LinOSS.
"""

import math

import equinox as eqx
import jax
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from architectures.networks.linoss import LinOSSMixer


class _Downsample1D(eqx.Module):
    conv: eqx.nn.Conv1d
    norm: eqx.nn.GroupNorm

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int,
        norm_groups: int,
        *,
        key: PRNGKeyArray,
    ):
        self.conv = eqx.nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=1,
            stride=stride,
            use_bias=False,
            key=key,
        )
        self.norm = eqx.nn.GroupNorm(norm_groups, out_ch)

    def __call__(self, x: Array) -> Array:
        return self.norm(self.conv(x))


class BasicBlock1D(eqx.Module):
    """Residual basic block with two 1D convolutions and GroupNorm."""

    conv1: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv1d
    norm2: eqx.nn.GroupNorm
    downsample: _Downsample1D | None

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        norm_groups: int = 16,
        downsample: "_Downsample1D | None" = None,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jr.split(key)
        padding = kernel_size // 2
        self.conv1 = eqx.nn.Conv1d(
            inplanes,
            planes,
            kernel_size,
            stride=stride,
            padding=padding,
            use_bias=False,
            key=k1,
        )
        self.norm1 = eqx.nn.GroupNorm(norm_groups, planes)
        self.conv2 = eqx.nn.Conv1d(
            planes,
            planes,
            kernel_size,
            padding=padding,
            use_bias=False,
            key=k2,
        )
        self.norm2 = eqx.nn.GroupNorm(norm_groups, planes)
        self.downsample = downsample

    def __call__(self, x: Array) -> Array:
        identity = x
        out = jax.nn.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return jax.nn.relu(out + identity)


def _make_layer_1d(
    inplanes: int,
    planes: int,
    num_blocks: int,
    kernel_size: int,
    stride: int = 1,
    norm_groups: int = 16,
    *,
    key: PRNGKeyArray,
) -> tuple[list[BasicBlock1D], int]:
    # key[0] → downsample, key[1..] → blocks
    keys = jr.split(key, num_blocks + 1)

    downsample = None
    if stride != 1 or inplanes != planes:
        downsample = _Downsample1D(
            inplanes, planes, stride, norm_groups, key=keys[0]
        )

    blocks = [
        BasicBlock1D(
            inplanes,
            planes,
            kernel_size,
            stride=stride,
            norm_groups=norm_groups,
            downsample=downsample,
            key=keys[1],
        )
    ]
    for i in range(1, num_blocks):
        blocks.append(
            BasicBlock1D(
                planes,
                planes,
                kernel_size,
                norm_groups=norm_groups,
                key=keys[i + 1],
            )
        )

    return blocks, planes


class ResNet1D(eqx.Module):
    """1D ResNet (BasicBlock) matching ml4gw ResNet1D with GroupNorm.

    Input:  (num_ifos, time)
    Output: (classes,)
    """

    conv1: eqx.nn.Conv1d
    norm1: eqx.nn.GroupNorm
    maxpool: eqx.nn.MaxPool1d
    res_layers: list  # list[list[BasicBlock1D]]
    avgpool: eqx.nn.AdaptiveAvgPool1d
    fc: eqx.nn.Linear

    def __init__(
        self,
        in_channels: int,
        layers: tuple[int, ...],
        classes: int,
        kernel_size: int = 3,
        norm_groups: int = 16,
        *,
        key: PRNGKeyArray,
    ):
        k_stem, k_layers, k_fc = jr.split(key, 3)
        layer_keys = jr.split(k_layers, len(layers))

        self.conv1 = eqx.nn.Conv1d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            use_bias=False,
            key=k_stem,
        )
        self.norm1 = eqx.nn.GroupNorm(norm_groups, 64)
        self.maxpool = eqx.nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        inplanes = 64
        res_layers = []
        for i, (num_blocks, k) in enumerate(
            zip(layers, layer_keys, strict=True)
        ):
            planes = 64 * (2**i)
            stride = 1 if i == 0 else 2
            blocks, inplanes = _make_layer_1d(
                inplanes,
                planes,
                num_blocks,
                kernel_size,
                stride,
                norm_groups,
                key=k,
            )
            res_layers.append(blocks)

        self.res_layers = res_layers
        self.avgpool = eqx.nn.AdaptiveAvgPool1d(target_shape=1)
        self.fc = eqx.nn.Linear(inplanes, classes, key=k_fc)

    def __call__(self, x: Array) -> Array:
        # x: (num_ifos, time)
        x = jax.nn.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        for layer in self.res_layers:
            for block in layer:
                x = block(x)
        x = self.avgpool(x)  # (channels, 1)
        x = x[..., 0]  # (channels,)
        return self.fc(x)  # (classes,)


class LinOSSModelResNetMLPDecoder(eqx.Module):
    """LinOSS prenorm backbone -> ResNet1D -> MLP head.

    Input:  (d_input, T)
    Output: (d_output,)
    """

    encoder: eqx.nn.Linear
    mixers: list
    norms: list
    dropout: eqx.nn.Dropout
    resnet: ResNet1D
    mlp: eqx.nn.MLP

    def __init__(
        self,
        d_input: int,
        d_output: int,
        d_model: int = 64,
        d_state: int = 64,
        n_layers: int = 4,
        dropout: float = 0.2,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = math.pi,
        resnet_layers: tuple[int, ...] = (2, 2, 2),
        resnet_latent_dim: int = 64,
        resnet_kernel_size: int = 3,
        resnet_norm_groups: int = 16,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        keys = jr.split(key, 3 + n_layers)
        self.encoder = eqx.nn.Linear(d_input, d_model, key=keys[0])
        self.mixers = [
            LinOSSMixer(
                d_model,
                d_state,
                key=keys[1 + i],
                r_min=r_min,
                theta_max=theta_max,
            )
            for i in range(n_layers)
        ]
        self.norms = [eqx.nn.LayerNorm(d_model) for _ in range(n_layers)]
        self.dropout = eqx.nn.Dropout(dropout)
        self.resnet = ResNet1D(
            in_channels=d_model,
            layers=tuple(resnet_layers),
            classes=resnet_latent_dim,
            kernel_size=resnet_kernel_size,
            norm_groups=resnet_norm_groups,
            key=keys[-2],
        )
        self.mlp = eqx.nn.MLP(
            in_size=resnet_latent_dim,
            out_size=d_output,
            width_size=mlp_width,
            depth=mlp_depth,
            key=keys[-1],
        )

    def __call__(self, x: Array, state=None, key: PRNGKeyArray = None):
        x = x.T  # (T, d_input)
        y = jax.vmap(self.encoder)(x)  # (T, d_model)

        if key is None:
            key = jr.PRNGKey(0)
        layer_keys = jr.split(key, len(self.mixers))
        for mixer, norm, k in zip(
            self.mixers, self.norms, layer_keys, strict=True
        ):
            z = mixer(jax.vmap(norm)(y))  # prenorm
            z = self.dropout(z, key=k)
            y = y + z

        y = y.T  # (d_model, T)
        h = self.resnet(y)  # (resnet_latent_dim,)
        logits = self.mlp(h)  # (d_output,)
        return logits, state
