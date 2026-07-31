import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


@eqx.filter_vmap(
    in_axes=(None, 0, None, 0), out_axes=(0, None), axis_name="batch"
)
def jax_fwd_batch(
    model: eqx.Module,
    X_time: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    return model(X_time, state, key=key)


def _beta_nll_loss(
    mean: Array, target: Array, var: Array, beta: float
) -> Array:
    """Gaussian NLL with stop-gradient variance weighting (BetaNLL)."""
    nll = 0.5 * (jnp.log(var) + (target - mean) ** 2 / var)
    return jnp.mean(jax.lax.stop_gradient(var) ** beta * nll)


def jax_regression_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X: Array,
    chirp_mass: Array,
    beta: float,
    lambda_spread: float,
    key: PRNGKeyArray,
) -> tuple[Array, tuple]:
    model = eqx.combine(diff_model, static_model)
    outputs, new_state = jax_fwd_batch(model, X, state, key)  # (B, d_output)
    n_vars = outputs.shape[-1] // 2
    mean = outputs[:, :n_vars]
    var = jax.nn.softplus(outputs[:, n_vars:])
    nll = _beta_nll_loss(mean, chirp_mass, var, beta)
    spread = jnp.mean(
        jax.nn.softplus(jnp.var(chirp_mass, axis=0) - jnp.var(mean, axis=0))
    )
    loss = nll + lambda_spread * spread
    return loss, (new_state, nll, spread, mean, var)


def _scale_by_group_lr(updates, labels, lr_other, lr_ssm):
    """Scale the adamw direction by ``-lr``, using ``lr_ssm`` for LinOSS
    mixer leaves and ``lr_other`` elsewhere. ``lr_*`` are supplied per step
    from the torch scheduler, so lr control lives in the same place (and the
    same ``LearningRateMonitor``) as the S4D models."""

    def scale(u, lab):
        lr = lr_ssm if lab == "ssm" else lr_other
        return -lr * u

    return jax.tree_util.tree_map(scale, updates, labels)


@eqx.filter_jit
def jax_apply_regression_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X: Array,
    chirp_mass: Array,
    beta: float,
    lambda_spread: float,
    opt_state,
    opt_update,
    lr_other: float,
    lr_ssm: float,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, nll, spread, mean, var)), grads = (
        eqx.filter_value_and_grad(jax_regression_loss_fn, has_aux=True)(
            diff_model,
            static_model,
            state,
            X,
            chirp_mass,
            beta,
            lambda_spread,
            key,
        )
    )
    directions, new_opt_state = opt_update(grads, opt_state, diff_model)
    labels = ssm_param_labels(diff_model)
    updates = _scale_by_group_lr(directions, labels, lr_other, lr_ssm)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return (
        new_model,
        new_state,
        new_opt_state,
        {
            "loss": loss,
            "nll": nll,
            "spread": spread,
            "mean": mean,
            "var": var,
        },
    )


@eqx.filter_jit
def jax_inference(
    model: eqx.Module,
    X: Array,
    state: eqx.nn.State,
    key: PRNGKeyArray,
) -> tuple[Array, eqx.nn.State]:
    inference_model = eqx.tree_inference(model, value=True)
    return jax_fwd_batch(inference_model, X, state, key)


def _bce_loss(logits: Array, y: Array) -> Array:
    import optax

    return jnp.mean(
        optax.sigmoid_binary_cross_entropy(logits, y.astype(logits.dtype))
    )


def jax_classification_loss_fn(
    diff_model: eqx.Module,
    static_model: eqx.Module,
    state: eqx.nn.State,
    X: Array,
    y: Array,
    key: PRNGKeyArray,
) -> tuple[Array, tuple]:
    model = eqx.combine(diff_model, static_model)
    logits, new_state = jax_fwd_batch(model, X, state, key)
    loss = _bce_loss(logits, y.reshape(-1, 1))
    return loss, (new_state, loss)


@eqx.filter_jit
def jax_apply_classification_training_step(
    model: eqx.Module,
    model_filter_spec,
    state: eqx.nn.State,
    X: Array,
    y: Array,
    opt_state,
    opt_update,
    lr_other: float,
    lr_ssm: float,
    key: PRNGKeyArray,
) -> tuple[eqx.Module, eqx.nn.State, object, dict]:
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, _)), grads = eqx.filter_value_and_grad(
        jax_classification_loss_fn, has_aux=True
    )(diff_model, static_model, state, X, y, key)
    directions, new_opt_state = opt_update(grads, opt_state, diff_model)
    labels = ssm_param_labels(diff_model)
    updates = _scale_by_group_lr(directions, labels, lr_other, lr_ssm)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return new_model, new_state, new_opt_state, {"loss": loss}


def ssm_param_labels(params):
    """Label each leaf ``"ssm"`` if it lives inside the LinOSS mixers
    (``model.mixers``), else ``"other"`` -- used by ``optax.multi_transform``
    to give the SSM params their own learning rate."""

    def tag(path, _leaf):
        return (
            "ssm"
            if any(
                isinstance(k, jax.tree_util.GetAttrKey) and k.name == "mixers"
                for k in path
            )
            else "other"
        )

    return jax.tree_util.tree_map_with_path(tag, params)


def jax_denoise_cls_loss_fn(
    diff_model, static_model, state, X, X_clean, y, lambda_denoise, key
):
    model = eqx.combine(diff_model, static_model)
    (x_denoised, logits), new_state = jax_fwd_batch(model, X, state, key)
    bce = _bce_loss(logits, y.reshape(-1, 1))
    denoise = jnp.mean((x_denoised - X_clean) ** 2)
    loss = bce + lambda_denoise * denoise
    return loss, (new_state, bce, denoise)


@eqx.filter_jit
def jax_apply_denoise_cls_training_step(
    model,
    model_filter_spec,
    state,
    X,
    X_clean,
    y,
    lambda_denoise,
    opt_state,
    opt_update,
    lr_other,
    lr_ssm,
    key,
):
    diff_model, static_model = eqx.partition(model, model_filter_spec)
    (loss, (new_state, bce, denoise)), grads = eqx.filter_value_and_grad(
        jax_denoise_cls_loss_fn, has_aux=True
    )(diff_model, static_model, state, X, X_clean, y, lambda_denoise, key)
    directions, new_opt_state = opt_update(grads, opt_state, diff_model)
    labels = ssm_param_labels(diff_model)
    updates = _scale_by_group_lr(directions, labels, lr_other, lr_ssm)
    new_model = eqx.combine(
        eqx.apply_updates(diff_model, updates), static_model
    )
    return (
        new_model,
        new_state,
        new_opt_state,
        {"loss": loss, "bce": bce, "denoise": denoise},
    )
