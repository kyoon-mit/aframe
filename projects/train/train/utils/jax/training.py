import equinox as eqx
import jax.numpy as jnp
import optax

from jaxtyping import Array, PRNGKeyArray


@eqx.filter_vmap(in_axes=(None, 0, None, 0), out_axes=(0, None))
def jax_fwd_batch(
    model: eqx.Module,
    X_time: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    return model(X_time, state, key=key)


def _bce_loss(logits: Array, y: Array) -> Array:
    """Binary cross entropy from combined outputs."""
    return jnp.mean(
        optax.sigmoid_binary_cross_entropy(logits, y.astype(logits.dtype))
    )


def jax_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y = y.reshape(-1, 1)
    loss = _bce_loss(logits, y)
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)

    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_loss_fn, has_aux=True
    )(diff_model, static_model, state, X_time, y, key)

    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def jax_snr_weighted_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    snr_weights: Array,
    fp_weight: float,
    snr_weight_power: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple[eqx.nn.State, Array]]:
    """BCE weighted by SNR for signals and ``fp_weight`` for background.

    Per-sample weights:
        - signal (y == 1): ``snr ^ snr_weight_power``
        - background (y == 0): ``fp_weight``

    Weights are normalised by their batch mean so the loss magnitude
    stays comparable to unweighted BCE regardless of hyperparameters.
    """
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X_time, state, key)
    y_flat = y.reshape(-1, 1)
    per_sample = optax.sigmoid_binary_cross_entropy(
        logits, y_flat.astype(logits.dtype)
    ).squeeze(-1)  # (N,)

    fg_mask = y.reshape(-1).astype(jnp.bool_)
    raw_w = jnp.where(
        fg_mask,
        snr_weights**snr_weight_power,
        fp_weight,
    )
    w = raw_w / jnp.mean(raw_w)
    loss = jnp.mean(w * per_sample)
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_snr_weighted_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X_time: Array,
    y: Array,
    snr_weights: Array,
    fp_weight: float,
    snr_weight_power: float,
    opt_state,
    opt_update,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)

    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_snr_weighted_loss_fn, has_aux=True
    )(
        diff_model,
        static_model,
        state,
        X_time,
        y,
        snr_weights,
        fp_weight,
        snr_weight_power,
        key,
    )

    updates, new_opt_state = opt_update(grads, opt_state, diff_model)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


@eqx.filter_jit
def jax_inference(
    model: eqx.Module,
    X_time: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    inference_model = eqx.tree_inference(model, value=True)
    return jax_fwd_batch(inference_model, X_time, state, key)
