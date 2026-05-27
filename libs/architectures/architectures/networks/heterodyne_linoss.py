"""
HeterodyneLinOSS: attention-based channel selection for heterodyned GW data.

For each interferometer (H1, L1) the input contains N chirp-mass channels.
A joint SSM encoder processes all N channels together at each time step,
allowing the model to learn cross-chirp-mass patterns (e.g. a real signal
appears as a ridge across adjacent chirp-mass channels) before computing K/Q.

  - K, Q: joint SSM on (T', N) → pool over T → (N,)
           → Linear(1, d_k) per channel
  - V:    per-channel SSM (vmapped), full sequence → (N, T', d_v)

Cross-attention (Q × K) produces n_out weighted sums of V, reducing N
chirp-mass channels to n_out per ifo. The n_out × d_v × num_ifos combined
channels feed the standard LinOSS backbone.

Input shape: (num_ifos * num_chirp_masses, time)
  channels are grouped by ifo: [H1_m0..H1_mN, L1_m0..L1_mN, ...]
Output: scalar logit, updated eqx.nn.State
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PRNGKeyArray

from architectures.networks.linoss import LinOSS, LinOSSMixer


class ChirpMassJointSSMEncoder(eqx.Module):
    """Encode all N chirp-mass channels jointly through one SSM call.

    Channels are stacked as features at each time step so the SSM can learn
    cross-chirp-mass context (e.g. a signal appears coherently across adjacent
    chirp-mass channels). After pooling over time, each channel gets back its
    own scalar summary that is informed by all other channels.

    (N, time') → transpose → (time', N) → Linear(N→d_model) →
    LinOSSMixer → LayerNorm → mean over time → Linear(d_model→N) → (N,)
    """

    input_proj: eqx.nn.Linear  # N → d_model
    mixer: LinOSSMixer
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear  # d_model → N

    def __init__(
        self,
        num_chirp_masses: int,
        d_model: int,
        state_dim: int,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.input_proj = eqx.nn.Linear(num_chirp_masses, d_model, key=k1)
        self.mixer = LinOSSMixer(
            d_model, state_dim, key=k2, r_min=r_min, theta_max=theta_max
        )
        self.norm = eqx.nn.LayerNorm(d_model)
        self.output_proj = eqx.nn.Linear(d_model, num_chirp_masses, key=k3)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        # x: (N, time') → (N,)
        x = x.T  # (time', N)
        x = jax.vmap(self.input_proj)(x)  # (time', d_model)
        x = self.mixer(x, key)  # (time', d_model)
        x = jax.vmap(self.norm)(x)  # (time', d_model)
        x = x.mean(axis=0)  # (d_model,) — global avg pool
        return self.output_proj(x)  # (N,)


class ChirpMassSSMValueEncoder(eqx.Module):
    """Encode a single chirp-mass channel to a full temporal value sequence.

    Same structure as ChirpMassSSMEncoder but returns the entire sequence
    (time, d_v) rather than a globally-pooled vector, so the attention
    output retains the time dimension.

    (time,) → linear → LinOSSMixer → LayerNorm → output_proj → (time, d_v)
    """

    input_proj: eqx.nn.Linear
    mixer: LinOSSMixer
    norm: eqx.nn.LayerNorm
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        d_model: int,
        state_dim: int,
        d_v: int,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.input_proj = eqx.nn.Linear(1, d_model, key=k1)
        self.mixer = LinOSSMixer(
            d_model, state_dim, key=k2, r_min=r_min, theta_max=theta_max
        )
        self.norm = eqx.nn.LayerNorm(d_model)
        self.output_proj = eqx.nn.Linear(d_model, d_v, key=k3)

    def __call__(self, x: Array, key: PRNGKeyArray) -> Array:
        # x: (time,) → (time, d_v)
        x = x[:, None]  # (time, 1)
        x = jax.vmap(self.input_proj)(x)  # (time, d_model)
        x = self.mixer(x, key)  # (time, d_model)
        x = jax.vmap(self.norm)(x)  # (time, d_model)
        return jax.vmap(self.output_proj)(x)  # (time, d_v)


class HeterodyneChirpMassAttention(eqx.Module):
    """Reduce N chirp-mass channels to n_out via SSM-driven cross-attention.

    For a single interferometer:
      1. Encode each chirp-mass channel with a shared SSM → (N, d_emb).
      2. Project embeddings to per-channel K and Q_all.
      3. Aggregate Q_all into n_out representative queries via a learned
         soft-selection (query_mix ∈ ℝ^{n_out × N}, softmax over N).
      4. Cross-attention: Q (n_out, d_k) × K^T (d_k, N) → weights (n_out, N).
      5. Encode channels into value space via a separate SSM: (N, time, d_v).
      6. Weighted combination: einsum(weights, V) → (n_out, time, d_v),
         transposed and reshaped to (n_out * d_v, time).

    Q and K come from a single joint SSM call on all N channels at once,
    giving each channel cross-chirp-mass context before attention weights
    are computed. V uses a separate per-channel SSM (vmapped) for its own
    temporal representation.

    A shared temporal conv (stride s, kernel k) is applied first, reducing
    sequence length T → T' ≈ T/s to keep SSM memory manageable.
    """

    temporal_conv: eqx.nn.Conv1d  # (1→1) shared across channels
    encoder: ChirpMassJointSSMEncoder  # joint Q/K encoder: (N, T') → (N,)
    value_encoder: (
        ChirpMassSSMValueEncoder  # per-channel V encoder: (T',) → (T', d_v)
    )
    proj_q: eqx.nn.Linear  # (1, d_k) — applied per channel scalar
    proj_k: eqx.nn.Linear  # (1, d_k) — applied per channel scalar
    query_mix: Array  # (n_out, num_chirp_masses) — soft channel selector
    d_k: int = eqx.field(static=True)
    d_v: int = eqx.field(static=True)

    def __init__(
        self,
        num_chirp_masses: int,
        d_model: int,
        state_dim: int,
        n_out: int,
        d_k: int,
        d_v: int,
        temporal_kernel_size: int = 8,
        temporal_stride: int = 8,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        k_tconv, k_enc, k_val, k_q, k_k, k_mix = jr.split(key, 6)
        # Shared 1-D conv downsamples every channel the same way.
        # kernel_size == stride → non-overlapping windows,
        # output length = T // stride.
        self.temporal_conv = eqx.nn.Conv1d(
            1,
            1,
            kernel_size=temporal_kernel_size,
            stride=temporal_stride,
            padding=0,
            key=k_tconv,
        )
        self.encoder = ChirpMassJointSSMEncoder(
            num_chirp_masses,
            d_model,
            state_dim,
            key=k_enc,
            r_min=r_min,
            theta_max=theta_max,
        )
        self.value_encoder = ChirpMassSSMValueEncoder(
            d_model,
            state_dim,
            d_v,
            key=k_val,
            r_min=r_min,
            theta_max=theta_max,
        )
        # K and Q are projected from per-channel scalars; Linear(1, d_k) is
        # shared across all channels (same weights, applied via vmap).
        self.proj_q = eqx.nn.Linear(1, d_k, key=k_q)
        self.proj_k = eqx.nn.Linear(1, d_k, key=k_k)
        # Small init so attention starts near-uniform
        self.query_mix = jr.normal(k_mix, (n_out, num_chirp_masses)) * 0.02
        self.d_k = d_k
        self.d_v = d_v

    def __call__(
        self,
        x_ifo: Array,  # (num_chirp_masses, time)
        key: PRNGKeyArray,
    ) -> Array:  # (n_out * d_v, time')
        n_channels = x_ifo.shape[0]
        k_enc, k_v = jr.split(key)
        enc_keys_v = jr.split(k_v, n_channels)

        # Temporal downsampling: (N, T) → (N, T')
        x_ds = jax.vmap(lambda c: self.temporal_conv(c[None])[0])(
            x_ifo
        )  # (N, T')

        # One joint SSM call on all N channels → per-channel scalars with
        # cross-chirp-mass context baked in
        summaries = self.encoder(x_ds, k_enc)  # (N,)

        # Per-channel K and Q from the cross-channel-aware scalars
        K = jax.vmap(self.proj_k)(summaries[:, None])  # (N, d_k)
        Q_all = jax.vmap(self.proj_q)(summaries[:, None])  # (N, d_k)

        # Soft-select n_out representative queries from the channel Q vectors
        mix = jax.nn.softmax(self.query_mix, axis=-1)  # (n_out, N)
        Q = mix @ Q_all  # (n_out, d_k)

        # Scaled dot-product attention weights
        attn = jax.nn.softmax(
            Q @ K.T / math.sqrt(self.d_k), axis=-1
        )  # (n_out, N)

        # Per-channel SSM encodes downsampled channels into value sequences
        V = jax.vmap(self.value_encoder)(x_ds, enc_keys_v)  # (N, T', d_v)

        # Weighted combination: (n_out, N) × (N, T', d_v) → (n_out, T', d_v)
        out = jnp.einsum("qn,ntd->qtd", attn, V)

        # Reshape to (n_out * d_v, T') so the backbone sees flat channels
        out = out.transpose(0, 2, 1)  # (n_out, d_v, T')
        return out.reshape(-1, out.shape[-1])  # (n_out * d_v, T')


class HeterodyneLinOSS(eqx.Module):
    """Full heterodyne detection model: chirp-mass attention + LinOSS backbone.

    Input  shape: (num_ifos * num_chirp_masses, time)
      Channels are grouped by interferometer:
      [H1_m0, …, H1_mN, L1_m0, …, L1_mN, …]

    Forward pass:
      1. Reshape to (num_ifos, num_chirp_masses, time).
      2. Apply shared HeterodyneChirpMassAttention per ifo (vmapped).
         → (num_ifos, n_out * d_v, time)
      3. Flatten to (num_ifos * n_out * d_v, time).
      4. Pass through the standard LinOSS backbone → scalar logit.
    """

    channel_attention: HeterodyneChirpMassAttention
    backbone: LinOSS
    num_ifos: int = eqx.field(static=True)

    def __init__(
        self,
        num_ifos: int,
        num_chirp_masses: int,
        # Attention / encoder params
        encoder_d_model: int = 32,
        encoder_state_dim: int = 32,
        n_out: int = 4,
        d_k: int = 32,
        d_v: int = 8,
        temporal_kernel_size: int = 8,
        temporal_stride: int = 8,
        encoder_r_min: float = 0.9,
        encoder_theta_max: float = jnp.pi,
        # LinOSS backbone params
        time_hidden_dim: int = 32,
        time_num_blocks: int = 3,
        time_dropout_rate: float = 0.0,
        time_state_dim: int = 32,
        time_r_min: float = 0.9,
        time_theta_max: float = jnp.pi,
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
        key: PRNGKeyArray,
    ):
        k_attn, k_backbone = jr.split(key)
        self.num_ifos = num_ifos

        self.channel_attention = HeterodyneChirpMassAttention(
            num_chirp_masses=num_chirp_masses,
            d_model=encoder_d_model,
            state_dim=encoder_state_dim,
            n_out=n_out,
            d_k=d_k,
            d_v=d_v,
            temporal_kernel_size=temporal_kernel_size,
            temporal_stride=temporal_stride,
            key=k_attn,
            r_min=encoder_r_min,
            theta_max=encoder_theta_max,
        )

        # Each ifo: n_out * d_v channels after attention + value projection
        self.backbone = LinOSS(
            time_in_features=num_ifos * n_out * d_v,
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
            key=k_backbone,
        )

    def __call__(
        self,
        x: Array,  # (num_ifos * num_chirp_masses, time)
        state: eqx.nn.State,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        k_attn, k_back = jr.split(key)

        # (num_ifos * num_chirp_masses, time)
        # → (num_ifos, num_chirp_masses, time)
        x = x.reshape(self.num_ifos, -1, x.shape[-1])

        # Shared attention weights applied independently per ifo
        ifo_keys = jr.split(k_attn, self.num_ifos)
        # → (num_ifos, n_out * d_v, time)
        x = jax.vmap(self.channel_attention)(x, ifo_keys)

        # Flatten to (num_ifos * n_out * d_v, time) for the backbone
        x = x.reshape(-1, x.shape[-1])

        return self.backbone(x, state, k_back)
