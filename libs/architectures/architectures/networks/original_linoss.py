"""
OriginalLinOSS: a faithful port of discretax's *plain* ``LinOSS`` model
(camail-official/discretax#74) — a linear encoder, a stack of ``StandardBlock``
blocks (BatchNorm + LinOSS sequence mixer + GLU channel mixer) at constant
hidden dimension, a mean-over-time reduction and an MLP head.

Unlike :mod:`architectures.networks.linoss` and
:mod:`architectures.networks.linoss2`, this module deliberately uses **no**
pooling/channel-expanding backbone and **no** intermediate ResNet.

The sequence mixer is the generalized real-pair :class:`LinOSSSequenceMixer`
ported from discretax PR #74 (reused from
:mod:`architectures.networks.linoss2`). It exposes the full feature set added
in that PR:

  - Discretizations: IM, IMEX, IMEX2, IMEX3, EX.
  - Damped-init strategies: AG (uniform A/G) and RT (radius/theta annulus).
  - Stability projection: "oscillatory" or "stable".
  - Multi-head mixing with an optional learned output projection.
  - LRU-style input normalization (per-mode input gain).
  - Configurable compute dtype.
"""

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from architectures.networks.linoss import GLU
from architectures.networks.linoss2 import LinOSSSequenceMixer

logger = logging.getLogger(__name__)


# --- StandardBlock + OriginalLinOSS model ------------------------


class StandardBlock(eqx.Module):
    """Standard residual block: norm + sequence mixer + GLU channel mixer.

    Faithful port of discretax's ``StandardBlock`` (the block used by S5, LRU
    and LinOSS): a BatchNorm normalization, the generalized
    :class:`LinOSSSequenceMixer` sequence mixer, a GLU channel mixer, dropout
    and a residual skip connection. Operates at a constant hidden dimension (no
    pooling, no channel expansion).

    The BatchNorm uses ``axis_name="batch"``; the surrounding harness vmaps the
    model over the batch axis with that name.
    """

    norm: eqx.nn.BatchNorm
    mixer: LinOSSSequenceMixer
    glu: GLU
    drop: eqx.nn.Dropout
    prenorm: bool = eqx.field(static=True)

    def __init__(
        self,
        in_features: int,
        state_dim: int,
        dropout_rate: float,
        discretization: str = "IMEX",
        initialization: str = "AG",
        damping: bool = True,
        stability: str = "oscillatory",
        projection_eps: float = 0.0,
        input_normalization: bool = False,
        num_heads: int = 1,
        use_head_output_projection: bool = False,
        A_max: float = 1.0,
        G_max: float = 1.0,
        dtype: jnp.dtype = jnp.float32,
        prenorm: bool = True,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        sm_k, glu_k = jr.split(key, 2)
        self.norm = eqx.nn.BatchNorm(
            input_size=in_features,
            axis_name="batch",
            channelwise_affine=False,
            mode="ema",
        )
        self.mixer = LinOSSSequenceMixer(
            in_features,
            key=sm_k,
            state_dim=state_dim,
            discretization=discretization,
            initialization=initialization,
            damping=damping,
            stability=stability,
            projection_eps=projection_eps,
            input_normalization=input_normalization,
            r_min=r_min,
            theta_max=theta_max,
            num_heads=num_heads,
            use_head_output_projection=use_head_output_projection,
            A_max=A_max,
            G_max=G_max,
            dtype=dtype,
        )
        self.glu = GLU(in_features, in_features, key=glu_k)
        self.drop = eqx.nn.Dropout(p=dropout_rate)
        self.prenorm = prenorm

    def __call__(self, x: Array, state: eqx.nn.State, key: PRNGKeyArray):
        _, d1, d2 = jr.split(key, 3)
        skip = x

        if self.prenorm:
            x, state = self.norm(x.T, state)
            x = x.T

        x = self.mixer(x, key)
        x = self.drop(jax.nn.gelu(x), key=d1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=d2)

        x = skip + x

        if not self.prenorm:
            x, state = self.norm(x.T, state)
            x = x.T

        return x, state


class OriginalLinOSS(eqx.Module):
    """OriginalLinOSS model: linear encoder + LinOSS blocks + MLP head.

    Faithful port of discretax's plain ``LinOSS`` model (no pooling backbone,
    no intermediate ResNet). A linear encoder lifts the input to
    ``time_hidden_dim``; a stack of :class:`StandardBlock` blocks mixes the
    sequence at constant hidden dimension using the generalized
    :class:`LinOSSSequenceMixer`; the time axis is reduced by mean and an MLP
    head maps to ``d_output``.
    """

    encoder: eqx.nn.Linear
    blocks: list[StandardBlock]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        time_in_features: int,
        time_hidden_dim: int,
        time_num_blocks: int,
        time_dropout_rate: float,
        time_state_dim: int,
        time_r_min: float,
        time_theta_max: float,
        mlp_width: int,
        mlp_depth: int,
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
        dtype: jnp.dtype = jnp.float32,
        d_output: int = 1,
        *,
        key: PRNGKeyArray,
    ):
        k_enc, k_blocks, k_mlp = jr.split(key, 3)
        self.encoder = eqx.nn.Linear(
            time_in_features, time_hidden_dim, use_bias=False, key=k_enc
        )

        blk_ks = jr.split(k_blocks, time_num_blocks)
        self.blocks = [
            StandardBlock(
                in_features=time_hidden_dim,
                state_dim=time_state_dim,
                dropout_rate=time_dropout_rate,
                discretization=time_discretization,
                initialization=time_initialization,
                damping=time_damping,
                stability=time_stability,
                projection_eps=time_projection_eps,
                input_normalization=time_input_normalization,
                num_heads=time_num_heads,
                use_head_output_projection=time_use_head_output_projection,
                A_max=time_A_max,
                G_max=time_G_max,
                dtype=dtype,
                prenorm=time_prenorm,
                key=bk,
                r_min=time_r_min,
                theta_max=time_theta_max,
            )
            for bk in blk_ks
        ]

        self.mlp = eqx.nn.MLP(
            in_size=time_hidden_dim,
            out_size=d_output,
            width_size=mlp_width,
            depth=mlp_depth,
            key=k_mlp,
        )

    def __call__(
        self,
        x_time: Array,
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        # (channels, time) -> (time, channels)
        x_time = x_time.T

        block_keys = jr.split(key, len(self.blocks))
        y = jax.vmap(self.encoder)(x_time)
        for block, bk in zip(self.blocks, block_keys, strict=True):
            y, state = block(y, state, bk)

        # Reduce over the time dimension, then map to outputs.
        y = jnp.mean(y, axis=0)
        logits = self.mlp(y)
        return logits, state
