import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import torch

from train.metrics import TimeSlideAUROC
from train.model.regression_kevin import RegressionAframe, _log_gaussian_nll
from architectures.base import Architecture, JaxArchitecture

from train.utils.jax.param_count import print_param_tree
from train.utils.jax.convert import tensor_to_jax_array, jax_array_to_tensor
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    jax_apply_regression_training_step,
    jax_inference,
)


logger = logging.getLogger(__name__)


class JaxRegressionAframe(RegressionAframe):
    """JAX/equinox version of :class:`RegressionAframe`.

    Trains a pre-built :class:`JaxArchitecture` with the same GaussianNLL
    (BetaNLL) + spread objective on the chirp mass as ``RegressionAframe``,
    but runs the forward/backward pass in JAX while Lightning drives the
    training loop with ``automatic_optimization = False``.

    The architecture must output ``d_output`` values per sample: the first
    half are the means, the second half the pre-Softplus variances.

    ``score``, ``compute_loss``, ``validation_step`` and ``test_step`` are
    inherited unchanged from ``RegressionAframe`` -- they only depend on
    ``self(X)``, which is overridden here to call the JAX model.
    """

    def __init__(
        self,
        arch: JaxArchitecture,
        d_output: int,
        metric: TimeSlideAUROC,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        clip_grad_norm: float = 10.0,
        warmup_steps: int = 1000,
        beta_nll: float = 0.5,
        lambda_spread: float = 0.0,
        y_mean: list[float] | None = None,
        y_std: list[float] | None = None,
        normalize_input: bool = False,
        target_param: str | None = None,
        val_target_param: str | None = None,
        seed: int = 42,
        load_from_checkpoint: str | None = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        super().__init__(
            arch=Architecture(),
            d_output=d_output,
            metric=metric,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            beta_nll=beta_nll,
            lambda_spread=lambda_spread,
            y_mean=y_mean,
            y_std=y_std,
            normalize_input=normalize_input,
            target_param=target_param,
            val_target_param=val_target_param,
        )
        self.automatic_optimization = False
        # Kept as plain floats for the (jit-traced) JAX loss function.
        self.beta_nll = beta_nll

        logger.info(f"JAX devices: {jax.devices()}")

        key = jr.PRNGKey(seed)

        self.jax_model = arch
        self.jax_model_state = eqx.nn.State(self.jax_model)
        self.jax_model_filter_spec = jax.tree_util.tree_map(
            eqx.is_inexact_array, self.jax_model
        )

        self.scheduler = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=max_steps,
            end_value=learning_rate * 0.01,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(clip_grad_norm),
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=self.scheduler, weight_decay=weight_decay
            ),
        )

        diff_model, _ = eqx.partition(
            self.jax_model, self.jax_model_filter_spec
        )
        self.opt_state = self.optimizer.init(diff_model)

        if load_from_checkpoint is not None:
            self.jax_model, self.jax_model_state, loaded_opt = load_model(
                load_from_checkpoint,
                self.jax_model,
                self.opt_state,
                self.jax_model_state,
            )
            if reset_optimizer_on_load:
                diff_model, _ = eqx.partition(
                    self.jax_model, self.jax_model_filter_spec
                )
                self.opt_state = self.optimizer.init(diff_model)
                logger.info(
                    "Optimizer state reset after checkpoint load"
                    " (fresh LR schedule starting from step 0)."
                )
            else:
                self.opt_state = loaded_opt
                logger.info(
                    "Optimizer state restored from checkpoint"
                    " (continuing LR schedule from saved step)."
                )

        self.rng_key = key

        print_param_tree(self.jax_model, depth=4, color="blue")

    def configure_optimizers(self):  # type: ignore
        pass

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = tensor_to_jax_array(X)

        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X.shape[0])

        outputs, new_state = jax_inference(
            self.jax_model, X, self.jax_model_state, keys
        )
        self.jax_model_state = new_state

        return jax_array_to_tensor(outputs)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Detection score: negative mean predicted variance.

        Lower uncertainty → higher score.
        """
        outputs = self(self._prepare_input(X))
        _, var_pre = outputs.chunk(2, dim=-1)
        return -self.var_activation(var_pre).mean(dim=-1)

    def training_step(self, batch):
        X, _, params = batch

        target = self.resolve_target(params, self.target_param)
        y_norm = self._normalize_target(target).reshape(-1, self.n_vars)

        X = tensor_to_jax_array(self._prepare_input(X))
        y_norm = tensor_to_jax_array(y_norm)

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, X.shape[0])

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_regression_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X,
            y_norm,
            self.beta_nll,
            self.lambda_spread,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        nll = metrics["nll"]
        var = metrics["var"]
        mean = metrics["mean"]
        indiv_mse = jnp.mean((mean - y_norm) ** 2, axis=0)

        _log_gaussian_nll(
            self,
            "train",
            float(nll),
            jax_array_to_tensor(indiv_mse),
            jax_array_to_tensor(var),
        )
        self.log(
            "train/loss",
            float(metrics["loss"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/spread_penalty",
            float(metrics["spread"]),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/lr",
            float(self.opt_state[1].hyperparams["learning_rate"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Needed for lightning if self.automatic_optimization = False
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)
