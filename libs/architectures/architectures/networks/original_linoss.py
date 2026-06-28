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

    norm: eqx.nn.BatchNorm | eqx.nn.LayerNorm | eqx.nn.GroupNorm
    mixer: LinOSSSequenceMixer
    glu: GLU
    drop: eqx.nn.Dropout
    prenorm: bool = eqx.field(static=True)
    norm_type: str = eqx.field(static=True)

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
        step_init: str = "logit_normal",
        dt_min: float = 1e-3,
        dt_max: float = 0.1,
        norm_groups: int = 16,
        dtype: jnp.dtype = jnp.float32,
        prenorm: bool = True,
        norm_type: str = "batch",
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        sm_k, glu_k = jr.split(key, 2)
        if norm_type == "batch":
            self.norm = eqx.nn.BatchNorm(
                input_size=in_features,
                axis_name="batch",
                channelwise_affine=False,
                mode="ema",
            )
        elif norm_type == "layer":
            self.norm = eqx.nn.LayerNorm(in_features)
        elif norm_type == "group":
            self.norm = eqx.nn.GroupNorm(
                groups=norm_groups, channels=in_features
            )
        else:
            raise ValueError(
                "norm_type must be 'batch', 'layer' or 'group', got "
                f"{norm_type!r}"
            )
        self.norm_type = norm_type
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
            step_init=step_init,
            dt_min=dt_min,
            dt_max=dt_max,
            dtype=dtype,
        )
        self.glu = GLU(in_features, in_features, key=glu_k)
        self.drop = eqx.nn.Dropout(p=dropout_rate)
        self.prenorm = prenorm

    def _apply_norm(self, x: Array, state: eqx.nn.State):
        """Apply the configured normalization to (time, channels) ``x``.

        BatchNorm normalizes each channel over (batch, time); GroupNorm
        normalizes within channel groups over (channels-in-group, time). Both
        expect (channels, time). LayerNorm normalizes over the channel/feature
        axis per time step and is vmapped over time.
        """
        if self.norm_type in ("batch", "group"):
            x, state = self.norm(x.T, state)
            return x.T, state
        x, state = jax.vmap(self.norm)(x, state)
        return x, state

    def __call__(self, x: Array, state: eqx.nn.State, key: PRNGKeyArray):
        _, d1, d2 = jr.split(key, 3)
        skip = x

        if self.prenorm:
            x, state = self._apply_norm(x, state)

        x = self.mixer(x, key)
        x = self.drop(jax.nn.gelu(x), key=d1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=d2)

        x = skip + x

        if not self.prenorm:
            x, state = self._apply_norm(x, state)

        return x, state


class OriginalLinOSS(eqx.Module):
    """OriginalLinOSS model: linear encoder + LinOSS blocks + MLP head.

    Faithful port of discretax's plain ``LinOSS`` model (no pooling backbone,
    no intermediate ResNet). A linear encoder lifts the input to
    ``time_hidden_dim``; a stack of :class:`StandardBlock` blocks mixes the
    sequence at constant hidden dimension using the generalized
    :class:`LinOSSSequenceMixer`.

    The temporal readout is selected by ``readout``:

      - ``"mean"`` (default): reduce the time axis by mean and apply an MLP
        head to map to ``d_output``. This reproduces the original model.
      - ``"max"``: apply a per-timestep MLP to every time step (producing
        ``readout_channels`` channels), take the max over time for each
        channel, then combine the per-channel maxima with a final MLP. This
        encourages the model to localize the signal in time rather than
        averaging over it.
    """

    encoder: eqx.nn.Linear
    blocks: list[StandardBlock]
    mlp: eqx.nn.MLP | None
    timestep_mlp: eqx.nn.MLP | None
    pool_mlp: eqx.nn.MLP | None
    readout: str = eqx.field(static=True)

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
        time_step_init: str = "logit_normal",
        time_dt_min: float = 1e-3,
        time_dt_max: float = 0.1,
        time_prenorm: bool = True,
        time_norm_type: str = "batch",
        time_norm_groups: int = 16,
        dtype: jnp.dtype = jnp.float32,
        d_output: int = 1,
        readout: str = "mean",
        readout_channels: int | None = None,
        *,
        key: PRNGKeyArray,
    ):
        if readout not in ("mean", "max"):
            raise ValueError(
                f"readout must be 'mean' or 'max', got {readout!r}"
            )
        self.readout = readout

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
                step_init=time_step_init,
                dt_min=time_dt_min,
                dt_max=time_dt_max,
                norm_groups=time_norm_groups,
                dtype=dtype,
                prenorm=time_prenorm,
                norm_type=time_norm_type,
                key=bk,
                r_min=time_r_min,
                theta_max=time_theta_max,
            )
            for bk in blk_ks
        ]

        if readout == "mean":
            self.mlp = eqx.nn.MLP(
                in_size=time_hidden_dim,
                out_size=d_output,
                width_size=mlp_width,
                depth=mlp_depth,
                key=k_mlp,
            )
            self.timestep_mlp = None
            self.pool_mlp = None
        else:  # readout == "max"
            n_channels = (
                time_hidden_dim
                if readout_channels is None
                else readout_channels
            )
            k_ts, k_pool = jr.split(k_mlp, 2)
            self.mlp = None
            # Per-timestep MLP mapping the hidden features to n_channels.
            self.timestep_mlp = eqx.nn.MLP(
                in_size=time_hidden_dim,
                out_size=n_channels,
                width_size=mlp_width,
                depth=mlp_depth,
                key=k_ts,
            )
            # Combine the per-channel maxima (over time) into the output.
            self.pool_mlp = eqx.nn.MLP(
                in_size=n_channels,
                out_size=d_output,
                width_size=mlp_width,
                depth=mlp_depth,
                key=k_pool,
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

        if self.readout == "mean":
            # Reduce over the time dimension, then map to outputs.
            y = jnp.mean(y, axis=0)
            logits = self.mlp(y)
        else:  # readout == "max"
            # Per-timestep MLP, max over time per channel, then combine.
            y = jax.vmap(self.timestep_mlp)(y)  # (time, n_channels)
            y = jnp.max(y, axis=0)  # (n_channels,)
            logits = self.pool_mlp(y)
        return logits, state
