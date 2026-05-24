"""
Reference / baseline LinOSS implementation.

Faithful adaptation of the original LinOSS paper architecture:
  - GLU activations
  - LinOSS-IM or LinOSS-IMEX discretisation via parallel scan
  - BatchNorm (requires vmap axis_name="batch" in the training loop)
  - Mean-pool over time → linear logit head (no softmax, suitable for BCE)

Input to OriginalLinOSS.__call__: (T, N)  — time-first, N input features
Output: ((1,), state)                      — single detection logit
"""

import math
from typing import List, Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray


def _simple_uniform_init(rng, shape, std=1.0):
    return random.uniform(rng, shape) * 2.0 * std - std


class OriginalGLU(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear

    def __init__(self, input_dim: int, output_dim: int, *, key: PRNGKeyArray):
        k1, k2 = jr.split(key, 2)
        self.w1 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=k1)
        self.w2 = eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=k2)

    def __call__(self, x: Array) -> Array:
        return self.w1(x) * jax.nn.sigmoid(self.w2(x))


# ---------------------------------------------------------------------------
# Optional add-on modules
# ---------------------------------------------------------------------------


class CausalConv1D(eqx.Module):
    """Depthwise-style causal 1-D convolution applied before the SSM.

    Input/output shape: (T, H) — time-first.
    Pads the left by (kernel_size - 1) so that no future context leaks in.
    """

    conv: eqx.nn.Conv1d
    pad: int = eqx.field(static=True)

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.conv = eqx.nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=0,
            key=key,
        )
        self.pad = kernel_size - 1

    def __call__(self, x: Array) -> Array:
        # x: (T, H) → (H, T) for Conv1d
        x = x.T
        x = jnp.pad(x, ((0, 0), (self.pad, 0)))
        x = self.conv(x)
        return x.T  # (T, H)


class MambaConv1D(eqx.Module):
    """Mamba-style bottleneck causal conv block: (T, H) → (T, H).

    Mirrors CausalBottleneck1D from the LinOSS heavy backbone:
        Linear H → H*expansion  (no bias)
        LayerNorm + GELU
        CausalConv1d H*expansion  (no bias, left-padded)
        LayerNorm + GELU
        Linear H*expansion → H  (no bias)
        LayerNorm
        + residual skip

    No state threading required — LayerNorm is called without state.
    """

    proj_in: eqx.nn.Linear
    norm1: eqx.nn.LayerNorm
    conv: eqx.nn.Conv1d
    conv_pad: int = eqx.field(static=True)
    norm2: eqx.nn.LayerNorm
    proj_out: eqx.nn.Linear
    norm3: eqx.nn.LayerNorm

    def __init__(
        self,
        channels: int,
        kernel_size: int = 4,
        expansion: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2, k3 = jr.split(key, 3)
        mid = channels * expansion
        self.proj_in = eqx.nn.Linear(channels, mid, use_bias=False, key=k1)
        self.norm1 = eqx.nn.LayerNorm(mid)
        self.conv = eqx.nn.Conv1d(
            mid, mid, kernel_size, use_bias=True, padding=0, groups=mid, key=k2
        )
        self.conv_pad = kernel_size - 1
        self.norm2 = eqx.nn.LayerNorm(mid)
        self.proj_out = eqx.nn.Linear(mid, channels, use_bias=False, key=k3)
        self.norm3 = eqx.nn.LayerNorm(channels)

    def __call__(self, x: Array) -> Array:
        # x: (T, H)
        identity = x
        x = jax.vmap(self.proj_in)(x)  # (T, mid)
        x = jax.nn.silu(jax.vmap(self.norm1)(x))
        xt = jnp.pad(x.T, ((0, 0), (self.conv_pad, 0)))
        x = jax.nn.silu(jax.vmap(self.norm2)(self.conv(xt).T))
        x = jax.vmap(self.norm3)(jax.vmap(self.proj_out)(x))
        return identity + x


# ---------------------------------------------------------------------------
# Encoder modules  (project N input features → H model dimension)
# ---------------------------------------------------------------------------


class LinearEncoder(eqx.Module):
    """Pointwise linear projection N → H, applied independently per timestep.

    This is the default encoder — equivalent to the original linear_encoder.
    Input/output shape: (T, N) → (T, H).
    """

    linear: eqx.nn.Linear

    def __init__(self, in_dim: int, out_dim: int, *, key: PRNGKeyArray):
        self.linear = eqx.nn.Linear(in_dim, out_dim, key=key)

    def __call__(self, x: Array) -> Array:
        return jax.vmap(self.linear)(x)  # (T, H)


class ConvEncoder(eqx.Module):
    """Stack of causal 1-D convolutions: (T, N) → (T, H).

    Layer 0 projects N → H with a causal conv of ``kernel_size``.
    Subsequent layers are H → H causal convs with residual connections.
    Left-padding preserves strict causality at every layer.
    """

    first_conv: eqx.nn.Conv1d
    first_pad: int = eqx.field(static=True)
    res_convs: List[eqx.nn.Conv1d]
    res_pads: List[int] = eqx.field(static=True)
    norms: List[eqx.nn.LayerNorm]

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int,
        kernel_size: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, max(num_layers, 1))
        self.first_conv = eqx.nn.Conv1d(
            in_dim, out_dim, kernel_size, padding=0, key=keys[0]
        )
        self.first_pad = kernel_size - 1
        self.res_convs = [
            eqx.nn.Conv1d(
                out_dim, out_dim, kernel_size, padding=0, key=keys[i]
            )
            for i in range(1, num_layers)
        ]
        self.res_pads = [kernel_size - 1] * max(0, num_layers - 1)
        self.norms = [eqx.nn.LayerNorm(out_dim) for _ in range(num_layers)]

    def __call__(self, x: Array) -> Array:
        # x: (T, N)
        xt = x.T  # (N, T)
        xt = jnp.pad(xt, ((0, 0), (self.first_pad, 0)))
        xt = jax.nn.gelu(self.first_conv(xt))  # (H, T)
        xt = jax.vmap(self.norms[0])(xt.T).T  # LayerNorm per timestep
        for conv, pad, norm in zip(
            self.res_convs, self.res_pads, self.norms[1:], strict=False
        ):
            skip = xt
            xp = jnp.pad(xt, ((0, 0), (pad, 0)))
            xt = jax.nn.gelu(conv(xp))
            xt = jax.vmap(norm)(xt.T).T
            xt = xt + skip
        return xt.T  # (T, H)


class PatchEncoder(eqx.Module):
    """Non-overlapping patch tokenizer: (T, N) → (T // patch_size, H).

    Splits the time-series into non-overlapping patches of length
    ``patch_size``, flattens each patch (patch_size * N), and linearly
    embeds it to H.  This reduces the sequence length fed to the SSM
    blocks by ``patch_size``, cutting parallel-scan cost proportionally.
    E.g. at 2048 Hz × 4 s = 8 192 samples, patch_size=16 gives 512 tokens.
    """

    proj: eqx.nn.Linear
    patch_size: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        patch_size: int,
        *,
        key: PRNGKeyArray,
    ):
        self.proj = eqx.nn.Linear(patch_size * in_dim, out_dim, key=key)
        self.patch_size = patch_size

    def __call__(self, x: Array) -> Array:
        # x: (T, N)
        T, N = x.shape
        P = self.patch_size
        T_trunc = (T // P) * P
        patches = x[:T_trunc].reshape(T_trunc // P, P * N)  # (T//P, P*N)
        return jax.vmap(self.proj)(patches)  # (T//P, H)


class MultiScaleConvEncoder(eqx.Module):
    """Parallel causal convolutions at multiple kernel sizes: (T, N) → (T, H).

    Each branch applies a causal 1-D conv with a different kernel size,
    producing ``H // len(kernel_sizes)`` features.  The branches are
    concatenated along the channel axis and projected to H, allowing the
    model to simultaneously capture short and long-range local structure.
    """

    branches: List[eqx.nn.Conv1d]
    pads: List[int] = eqx.field(static=True)
    out_proj: eqx.nn.Linear
    branch_dim: int = eqx.field(static=True)

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_sizes: List[int],
        *,
        key: PRNGKeyArray,
    ):
        n = len(kernel_sizes)
        branch_dim = out_dim // n
        keys = jr.split(key, n + 1)
        self.branches = [
            eqx.nn.Conv1d(in_dim, branch_dim, k, padding=0, key=keys[i])
            for i, k in enumerate(kernel_sizes)
        ]
        self.pads = [k - 1 for k in kernel_sizes]
        self.out_proj = eqx.nn.Linear(branch_dim * n, out_dim, key=keys[-1])
        self.branch_dim = branch_dim

    def __call__(self, x: Array) -> Array:
        # x: (T, N) → (N, T) for Conv1d
        xt = x.T
        outs = []
        for conv, pad in zip(self.branches, self.pads, strict=False):
            xp = jnp.pad(xt, ((0, 0), (pad, 0)))
            outs.append(jax.nn.gelu(conv(xp)))  # each: (branch_dim, T)
        cat = jnp.concatenate(outs, axis=0).T  # (T, branch_dim * n)
        return jax.vmap(self.out_proj)(cat)  # (T, H)


class MLPHead(eqx.Module):
    """Multi-layer perceptron readout head.

    Replaces the single linear layer after mean-pooling.
    Architecture: input → (hidden → GELU) × (num_layers-1) → 1
    """

    layers: List[eqx.nn.Linear]

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        *,
        key: PRNGKeyArray,
    ):
        keys = jr.split(key, num_layers)
        layers = []
        for i, k in enumerate(keys):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = 1 if i == num_layers - 1 else hidden_dim
            layers.append(eqx.nn.Linear(in_dim, out_dim, key=k))
        self.layers = layers

    def __call__(self, x: Array) -> Array:
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


# ---------------------------------------------------------------------------
# Parallel-scan operators (identical to original paper)
# ---------------------------------------------------------------------------


@jax.vmap
def _original_binary_operator(q_i, q_j):
    """Associative operator for the 2nd-order LinOSS state recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    N = A_i.size // 4
    iA = A_i[0 * N : 1 * N]
    iB = A_i[1 * N : 2 * N]
    iC = A_i[2 * N : 3 * N]
    iD = A_i[3 * N : 4 * N]
    jA = A_j[0 * N : 1 * N]
    jB = A_j[1 * N : 2 * N]
    jC = A_j[2 * N : 3 * N]
    jD = A_j[3 * N : 4 * N]
    A_new = jnp.concatenate(
        [
            jA * iA + jB * iC,
            jA * iB + jB * iD,
            jC * iA + jD * iC,
            jC * iB + jD * iD,
        ]
    )
    b1, b2 = b_i[:N], b_i[N:]
    b_new = jnp.concatenate([jA * b1 + jB * b2, jC * b1 + jD * b2])
    return A_new, b_new + b_j


def _apply_linoss_im(A_diag, B, C_tilde, input_sequence, step):
    """Compute LinOSS-IM outputs for an (L, H) input sequence."""
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    schur_comp = 1.0 / (1.0 + step**2.0 * A_diag)
    M_IM_11 = 1.0 - step**2.0 * A_diag * schur_comp
    M_IM_12 = -1.0 * step * A_diag * schur_comp
    M_IM_21 = step * schur_comp
    M_IM_22 = schur_comp
    M_IM = jnp.concatenate([M_IM_11, M_IM_12, M_IM_21, M_IM_22])
    M_IM_elements = M_IM * jnp.ones(
        (input_sequence.shape[0], 4 * A_diag.shape[0])
    )
    F1 = M_IM_11 * Bu_elements * step
    F2 = M_IM_21 * Bu_elements * step
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_IM_elements, F)
    )
    ys = xs[:, A_diag.shape[0] :]
    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


_DAMPED_DISCRETIZATIONS = frozenset(
    {"DampedIMEX1", "DampedIMEX2", "DampedIM", "DampedEX"}
)


def _apply_linoss_damped_imex1(
    A_diag, G_diag, B, C_tilde, input_sequence, step
):
    """Damped LinOSS-IMEX1.

    Recurrence: z_{k+1} = z_k + dt(-Ax_k - Gz_{k+1} + Bu_{k+1}),
                x_{k+1} = x_k + dt*z_{k+1}.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    N = A_diag.shape[0]
    ones = jnp.ones_like(A_diag)
    S = ones + step * G_diag
    M_11 = ones / S
    M_12 = -step * A_diag / S
    M_21 = step / S
    M_22 = ones - step**2 * A_diag / S
    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * N))
    F1 = step / S * Bu_elements
    F2 = step**2 / S * Bu_elements
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_elements, F)
    )
    ys = xs[:, N:]
    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def _apply_linoss_damped_imex2(
    A_diag, G_diag, B, C_tilde, input_sequence, step
):
    """Damped LinOSS-IMEX2.

    Recurrence: z_{k+1} = z_k + dt(-Ax_k - Gz_k + Bu_{k+1}),
                x_{k+1} = x_k + dt*z_{k+1}.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    N = A_diag.shape[0]
    ones = jnp.ones_like(A_diag)
    M_11 = ones - step * G_diag
    M_12 = -step * A_diag
    M_21 = step * (ones - step * G_diag)
    M_22 = ones - step**2 * A_diag
    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * N))
    F1 = step * Bu_elements
    F2 = step**2 * Bu_elements
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_elements, F)
    )
    ys = xs[:, N:]
    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def _apply_linoss_damped_im(A_diag, G_diag, B, C_tilde, input_sequence, step):
    """Damped LinOSS-IM.

    Recurrence: z_{k+1} = z_k + dt(-Ax_{k+1} - Gz_{k+1} + Bu_{k+1}),
                x_{k+1} = x_k + dt*z_{k+1}.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    N = A_diag.shape[0]
    ones = jnp.ones_like(A_diag)
    S = ones + step * G_diag + step**2 * A_diag
    M_11 = ones / S
    M_12 = -step * A_diag / S
    M_21 = step / S
    M_22 = (ones + step * G_diag) / S
    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * N))
    F1 = step / S * Bu_elements
    F2 = step**2 / S * Bu_elements
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_elements, F)
    )
    ys = xs[:, N:]
    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def _apply_linoss_damped_ex(A_diag, G_diag, B, C_tilde, input_sequence, step):
    """Damped LinOSS-EX.

    Recurrence: z_{k+1} = z_k + dt(-Ax_k - Gz_k + Bu_{k+1}),
                x_{k+1} = x_k + dt*z_k.
    """
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    N = A_diag.shape[0]
    ones = jnp.ones_like(A_diag)
    M_11 = ones - step * G_diag
    M_12 = -step * A_diag
    M_21 = step * ones
    M_22 = ones
    M = jnp.concatenate([M_11, M_12, M_21, M_22])
    M_elements = M * jnp.ones((input_sequence.shape[0], 4 * N))
    F1 = step * Bu_elements
    F2 = jnp.zeros_like(F1)
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_elements, F)
    )
    ys = xs[:, N:]
    return jax.vmap(lambda x: (C_tilde @ x).real)(ys)


def _apply_linoss_imex(A_diag, B, C, input_sequence, step):
    """Compute LinOSS-IMEX outputs for an (L, H) input sequence."""
    Bu_elements = jax.vmap(lambda u: B @ u)(input_sequence)
    A_ = jnp.ones_like(A_diag)
    B_ = -1.0 * step * A_diag
    C_ = step
    D_ = 1.0 - (step**2.0) * A_diag
    M_IMEX = jnp.concatenate([A_, B_, C_, D_])
    M_IMEX_elements = M_IMEX * jnp.ones(
        (input_sequence.shape[0], 4 * A_diag.shape[0])
    )
    F1 = Bu_elements * step
    F2 = Bu_elements * (step**2.0)
    F = jnp.hstack((F1, F2))
    _, xs = jax.lax.associative_scan(
        _original_binary_operator, (M_IMEX_elements, F)
    )
    ys = xs[:, A_diag.shape[0] :]
    return jax.vmap(lambda x: (C @ x).real)(ys)


# ---------------------------------------------------------------------------
# LinOSS layer and block
# ---------------------------------------------------------------------------


class OriginalLinOSSLayer(eqx.Module):
    A_diag: Array
    B: Array
    C: Array
    D: Array
    steps: Array
    G_diag: Optional[Array]
    discretization: str = eqx.field(static=True)

    def __init__(
        self,
        ssm_size: int,
        H: int,
        discretization: str,
        *,
        key: PRNGKeyArray,
    ):
        B_key, C_key, D_key, A_key, step_key, G_key = jr.split(key, 6)
        self.A_diag = random.uniform(A_key, shape=(ssm_size,))
        self.B = _simple_uniform_init(
            B_key, shape=(ssm_size, H, 2), std=1.0 / math.sqrt(H)
        )
        self.C = _simple_uniform_init(
            C_key, shape=(H, ssm_size, 2), std=1.0 / math.sqrt(ssm_size)
        )
        self.D = normal(stddev=1.0)(D_key, (H,))
        self.steps = random.uniform(step_key, shape=(ssm_size,))
        self.discretization = discretization
        if discretization in _DAMPED_DISCRETIZATIONS:
            self.G_diag = random.uniform(G_key, shape=(ssm_size,))
        else:
            self.G_diag = None

    def __call__(self, input_sequence: Array) -> Array:
        A_diag = nn.relu(self.A_diag)
        B_complex = self.B[..., 0] + 1j * self.B[..., 1]
        C_complex = self.C[..., 0] + 1j * self.C[..., 1]
        steps = nn.sigmoid(self.steps)
        if self.discretization == "IMEX":
            ys = _apply_linoss_imex(
                A_diag, B_complex, C_complex, input_sequence, steps
            )
        elif self.discretization == "DampedIMEX1":
            G_diag = nn.relu(self.G_diag)
            ys = _apply_linoss_damped_imex1(
                A_diag, G_diag, B_complex, C_complex, input_sequence, steps
            )
        elif self.discretization == "DampedIMEX2":
            G_diag = nn.relu(self.G_diag)
            ys = _apply_linoss_damped_imex2(
                A_diag, G_diag, B_complex, C_complex, input_sequence, steps
            )
        elif self.discretization == "DampedIM":
            G_diag = nn.relu(self.G_diag)
            ys = _apply_linoss_damped_im(
                A_diag, G_diag, B_complex, C_complex, input_sequence, steps
            )
        elif self.discretization == "DampedEX":
            G_diag = nn.relu(self.G_diag)
            ys = _apply_linoss_damped_ex(
                A_diag, G_diag, B_complex, C_complex, input_sequence, steps
            )
        else:  # IM (default)
            ys = _apply_linoss_im(
                A_diag, B_complex, C_complex, input_sequence, steps
            )
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return ys + Du


class OriginalLinOSSBlock(eqx.Module):
    norm: eqx.nn.LayerNorm | eqx.nn.BatchNorm
    causal_conv: Optional[Union[CausalConv1D, MambaConv1D]]
    ssm: OriginalLinOSSLayer
    glu: OriginalGLU
    drop: eqx.nn.Dropout
    norm_pos: str = eqx.field(static=True, default="pre")  # "pre" or "post"
    norm_type: str = eqx.field(
        static=True, default="layer"
    )  # "layer" or "batch"
    conv_type: str = eqx.field(
        static=True, default="none"
    )  # "none" | "simple" | "bottleneck"

    def __init__(
        self,
        ssm_size: int,
        H: int,
        discretization: str,
        drop_rate: float = 0.05,
        conv_type: str = "none",
        conv_kernel_size: int = 4,
        conv_expansion: int = 2,
        norm_pos: str = "pre",
        norm_type: str = "batch",
        *,
        key: PRNGKeyArray,
    ):
        ssm_key, glu_key, conv_key = jr.split(key, 3)
        self.norm_type = norm_type
        if norm_type == "batch":
            self.norm = eqx.nn.BatchNorm(
                input_size=H,
                axis_name="batch",
                channelwise_affine=False,
            )
        elif norm_type == "layer":
            self.norm = eqx.nn.LayerNorm(H)
        else:
            raise ValueError(f"Invalid norm_type: {norm_type}")

        self.conv_type = conv_type
        if conv_type == "simple":
            self.causal_conv = CausalConv1D(H, conv_kernel_size, key=conv_key)
        elif conv_type == "bottleneck":
            self.causal_conv = MambaConv1D(
                H, conv_kernel_size, conv_expansion, key=conv_key
            )
        else:
            self.causal_conv = None
        self.ssm = OriginalLinOSSLayer(
            ssm_size, H, discretization, key=ssm_key
        )
        self.glu = OriginalGLU(H, H, key=glu_key)
        self.drop = eqx.nn.Dropout(p=drop_rate)
        self.norm_pos = norm_pos

    def apply_norm(
        self, x: Array, state: eqx.nn.State
    ) -> tuple[Array, eqx.nn.State]:
        if self.norm_type == "batch":
            x, state = self.norm(x.T, state)
            return x.T, state
        elif self.norm_type == "layer":
            return jax.vmap(self.norm)(x, state)
        else:
            raise ValueError(f"Invalid norm_type: {self.norm_type}")

    def __call__(
        self, x: Array, state: eqx.nn.State, *, key: PRNGKeyArray
    ) -> tuple[Array, eqx.nn.State]:
        d1, d2 = jr.split(key, 2)
        skip = x
        # BatchNorm expects (H, L), so transpose

        if self.norm_pos == "pre":
            x, state = self.apply_norm(x, state)

        if self.causal_conv is not None:
            x = self.causal_conv(x)
        x = self.ssm(x)

        if self.norm_pos == "post":
            x, state = self.apply_norm(x, state)

        x = self.drop(jax.nn.gelu(x), key=d1)
        x = jax.vmap(self.glu)(x)
        x = self.drop(x, key=d2)
        return skip + x, state


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class OriginalLinOSS(eqx.Module):
    """Original LinOSS architecture adapted for binary GW detection.

    Input:  (T, N) — time-first, N = num_ifos input features
    Output: ((1,), state) — single detection logit via mean-pool + head

    Architecture:
        encoder  N → H  (see encoder_type)
        num_blocks × OriginalLinOSSBlock
            (BatchNorm/LayerNorm + [CausalConv1D] + LinOSS-IM/IMEX + GLU)
        mean-pool over T
        linear_layer  H → 1  (or MLPHead if use_mlp_head=True)

    encoder_type options:
        "linear"         — pointwise linear projection (default).
        "conv"           — stack of causal 1-D convolutions with residuals;
                           controlled by encoder_num_layers /
                           encoder_kernel_size.
        "patch"          — non-overlapping patch embedding that reduces the
                           sequence length by patch_size before the SSM layers.
        "multiscale_conv" — parallel causal convs at different kernel sizes
                           (encoder_kernel_sizes) concatenated and
                           projected to H.

    Note: BatchNorm in each block requires the vmap in the training loop to
    use axis_name="batch" so that batch statistics are collected correctly.
    """

    encoder: Union[
        LinearEncoder, ConvEncoder, PatchEncoder, MultiScaleConvEncoder
    ]
    encoder_type: str = eqx.field(static=True)
    blocks: List[OriginalLinOSSBlock]
    linear_layer: Union[eqx.nn.Linear, MLPHead]
    stateful: bool = True
    nondeterministic: bool = True

    def __init__(
        self,
        N: int,
        H: int,
        num_blocks: int,
        ssm_size: int,
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
        encoder_kernel_sizes: Optional[List[int]] = None,
        *,
        key: PRNGKeyArray,
    ):
        enc_key, *block_keys, head_key = jr.split(key, num_blocks + 2)
        self.encoder_type = encoder_type
        if encoder_type == "linear":
            self.encoder = LinearEncoder(N, H, key=enc_key)
        elif encoder_type == "conv":
            self.encoder = ConvEncoder(
                N, H, encoder_num_layers, encoder_kernel_size, key=enc_key
            )
        elif encoder_type == "patch":
            self.encoder = PatchEncoder(N, H, patch_size, key=enc_key)
        elif encoder_type == "multiscale_conv":
            scales = (
                encoder_kernel_sizes
                if encoder_kernel_sizes is not None
                else [4, 8, 16]
            )
            self.encoder = MultiScaleConvEncoder(N, H, scales, key=enc_key)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type!r}")
        self.blocks = [
            OriginalLinOSSBlock(
                ssm_size,
                H,
                discretization,
                drop_rate=drop_rate,
                conv_type=conv_type,
                conv_kernel_size=conv_kernel_size,
                conv_expansion=conv_expansion,
                key=k,
                norm_pos=norm_pos,
                norm_type=norm_type,
            )
            for k in block_keys
        ]
        self.linear_layer = (
            MLPHead(H, mlp_hidden_dim, mlp_depth, key=head_key)
            if use_mlp_head
            else eqx.nn.Linear(H, 1, key=head_key)
        )

    def __call__(
        self,
        x: Array,
        state: eqx.nn.State,
        *,
        key: PRNGKeyArray,
    ) -> tuple[Array, eqx.nn.State]:
        # x: (T, N)
        drop_keys = jr.split(key, len(self.blocks))
        x = self.encoder(x)  # (T_out, H)
        for block, k in zip(self.blocks, drop_keys, strict=False):
            x, state = block(x, state, key=k)
        x = jnp.mean(x, axis=0)  # (H,)  — mean-pool over time
        x = self.linear_layer(x)  # (1,)
        return x, state
