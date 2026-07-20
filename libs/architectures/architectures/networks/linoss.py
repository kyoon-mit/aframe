"""Minimal LinOSS (linear oscillatory state-space) model in JAX/equinox.

Ported from the damped LinOSS-IMEX mixer. Kept lean: the oscillator core
(``LinOSSMixer``) plus a simple encoder -> mixer stack -> mean-pool ->
linear head model (``LinOSSModel``), mirroring the shape of ml4gw's S4Model.
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import nn, random
from jax.nn.initializers import normal
from jaxtyping import Array, PRNGKeyArray


@jax.vmap
def _binary_op(q_i, q_j):
    """Associative operator for the 2nd-order state recurrence."""
    A_i, b_i = q_i
    A_j, b_j = q_j
    N = A_i.size // 4
    iA = A_i[:N]
    iB = A_i[N : 2 * N]
    iC = A_i[2 * N : 3 * N]
    iD = A_i[3 * N :]
    jA = A_j[:N]
    jB = A_j[N : 2 * N]
    jC = A_j[2 * N : 3 * N]
    jD = A_j[3 * N :]
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


def _damped_linoss_imex(A_diag, G_diag, B, x, step):
    """Damped LinOSS-IMEX via parallel scan.

    Args:
        A_diag: (P,)   diagonal stiffness
        G_diag: (P,)   diagonal damping
        B:      (P, H) complex input matrix
        x:      (T, H) real input sequence
        step:   (P,)   discretisation step-sizes

    Returns:
        ys: (T, P) complex, the velocity component of the state.
    """
    P = A_diag.shape[0]
    L = x.shape[0]
    Bu = jax.vmap(lambda u: B @ u)(x)  # (T, P) complex
    S = 1.0 + step * G_diag
    M = jnp.concatenate(
        [1 / S, -step * A_diag / S, step / S, 1.0 - step**2 * A_diag / S]
    )
    M_elems = M * jnp.ones((L, 4 * P))
    F = jnp.hstack([Bu * step / S, Bu * step**2 / S])
    _, xs = jax.lax.associative_scan(_binary_op, (M_elems, F))
    return xs[:, P:]  # velocity component


def _theta_to_A(theta, G, steps):
    """Map oscillation angle to initial A_diag for damped LinOSS-IMEX."""
    cos2 = jnp.cos(theta) ** (-2)
    sq = jnp.sqrt(steps**4 * cos2 + steps**5 * G * cos2)
    tan2 = jnp.tan(theta) ** 2
    base = -(steps**2) * (-4 - 2 * steps * G - (4 + 2 * steps * G) * tan2)
    denom = 2 * steps**4 * (1 + tan2)
    A_plus = (4 * sq + base) / denom
    A_minus = (-4 * sq + base) / denom
    return jnp.where(theta > jnp.pi / 2, A_plus, A_minus)


class LinOSSMixer(eqx.Module):
    """Damped LinOSS-IMEX sequence mixer, maps (T, H) -> (T, H)."""

    A_diag: Array  # (P,)
    G_diag: Array  # (P,)
    B: Array  # (P, H, 2) complex stored as real/imag pair
    C: Array  # (H, P, 2)
    D: Array  # (H,)
    steps: Array  # (P,) pre-sigmoid log-steps

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        *,
        key: PRNGKeyArray,
        r_min: float = 0.9,
        theta_max: float = jnp.pi,
    ):
        A_k, G_k, B_k, C_k, D_k, s_k = jr.split(key, 6)
        P, H = state_dim, input_dim

        self.steps = normal(stddev=0.5)(s_k, (P,))
        steps = nn.sigmoid(self.steps)

        # initialise magnitudes in [r_min, 1] for stability
        mags = jnp.sqrt(
            random.uniform(G_k, (P,)) * (1.0 - r_min**2) + r_min**2
        )
        self.G_diag = (1 - mags**2) / (steps * mags**2)
        G = nn.relu(self.G_diag)

        theta = random.uniform(A_k, (P,)) * theta_max
        self.A_diag = _theta_to_A(theta, G, steps)

        std_B = 1.0 / math.sqrt(H)
        self.B = random.uniform(B_k, (P, H, 2)) * 2 * std_B - std_B
        std_C = 1.0 / math.sqrt(P)
        self.C = random.uniform(C_k, (H, P, 2)) * 2 * std_C - std_C
        self.D = normal(stddev=1.0)(D_k, (H,))

    def __call__(self, x: Array, key: PRNGKeyArray = None) -> Array:
        steps = nn.sigmoid(self.steps)
        G = nn.relu(self.G_diag)

        # clamp A to the valid stability region
        lo = (2 + steps * G - 2 * jnp.sqrt(1 + steps * G)) / steps**2
        hi = (2 + steps * G + 2 * jnp.sqrt(1 + steps * G)) / steps**2
        A = lo + nn.relu(self.A_diag - lo) - nn.relu(self.A_diag - hi)

        Bc = self.B[..., 0] + 1j * self.B[..., 1]
        Cc = self.C[..., 0] + 1j * self.C[..., 1]

        ys = _damped_linoss_imex(A, G, Bc, x, steps)  # (T, P) complex
        Cy = jax.vmap(lambda y: (Cc @ y).real)(ys)  # (T, H)
        Du = jax.vmap(lambda u: self.D * u)(x)  # (T, H)
        return Cy + Du


class LinOSSModel(eqx.Module):
    """Lean LinOSS sequence model, analogous to ml4gw's S4Model.

    Input:  (d_input, T)   real strain, channels-first
    Output: (d_output,)    pooled prediction

    encoder Linear -> n_layers x (LinOSSMixer + LayerNorm + dropout +
    residual) -> mean-pool over time -> decoder Linear.
    """

    encoder: eqx.nn.Linear
    mixers: list
    norms: list
    dropout: eqx.nn.Dropout
    decoder: eqx.nn.Linear

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
    ):
        keys = jr.split(key, 2 + n_layers)
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
        self.decoder = eqx.nn.Linear(d_model, d_output, key=keys[-1])

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

        y = jnp.mean(y, axis=0)  # (d_model,)
        logits = self.decoder(y)  # (d_output,)
        return logits, state
