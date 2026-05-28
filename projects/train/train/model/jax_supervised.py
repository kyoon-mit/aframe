import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import torch

from train.metric import LightningStepOutput
from train.metrics import TimeSlideAUROC
from train.model.base import AframeBase
from architectures.base import Architecture, JaxArchitecture

from train.utils.jax.param_count import print_param_tree
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    jax_apply_training_step,
    jax_apply_snr_weighted_training_step,
    jax_apply_focal_training_step,
    jax_apply_hnm_training_step,
    jax_apply_pauc_training_step,
    jax_apply_focal_hnm_training_step,
    jax_inference,
)


logger = logging.getLogger(__name__)


class JaxSupervisedAframe(AframeBase):
    """"""

    def __init__(
        self,
        arch: JaxArchitecture,
        metric: TimeSlideAUROC,
        learning_rate: float = 1e-4,
        pct_lr_ramp: float = 0.1,
        weight_decay: float = 0.0,
        max_steps: int = 500_000,
        clip_grad_norm: float = 10.0,
        warmup_steps: int = 1000,
        seed: int = 42,
        load_from_checkpoint: str | None = None,
        reset_optimizer_on_load: bool = True,
    ):
        super().__init__(
            arch=Architecture(),
            metric=metric,
            learning_rate=learning_rate,
            pct_lr_ramp=pct_lr_ramp,
            weight_decay=weight_decay,
        )
        self.automatic_optimization = False

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

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_time_j = jnp.asarray(X.cpu().numpy())

        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X_time_j.shape[0])

        logits, new_state = jax_inference(
            self.jax_model, X_time_j, self.jax_model_state, keys
        )
        self.jax_model_state = new_state

        return torch.from_numpy(np.array(logits, copy=True))

    def validation_step(self, batch, _):
        shift, X_bg, X_inj, params_tensor = batch

        y_bg = self.forward(X_bg)

        num_views, batch_size, *shape = X_inj.shape
        X_inj_flat = X_inj.view(num_views * batch_size, *shape)

        y_fg_flat = self.forward(X_inj_flat)
        y_fg = y_fg_flat.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)
        self.log(
            "val/valid_auroc",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        y_bg_np = np.array(y_bg.cpu()).flatten()
        y_fg_np = np.array(y_fg.cpu()).flatten()

        param_names = self.trainer.datamodule.val_param_names
        params_dict = {
            name: params_tensor[:, i].cpu().numpy()
            for i, name in enumerate(param_names)
        }

        return LightningStepOutput(
            targets=np.concatenate(
                [np.zeros(len(y_bg_np)), np.ones(len(y_fg_np))]
            ),
            outputs=np.concatenate([y_bg_np, y_fg_np]),
            bg_outputs=y_bg_np,
            params=params_dict,
        )


class JaxFocalLossSupervisedAframe(JaxSupervisedAframe):
    """Jax supervised model trained with focal loss.

    Focal loss down-weights easy examples so the gradient concentrates on
    hard/misclassified samples:  ``L = (1 - p_t)^gamma * BCE``.
    Higher ``gamma`` focuses more aggressively on hard cases.
    """

    def __init__(self, gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_focal_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.gamma,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)


class JaxHardNegMiningSupervisedAframe(JaxSupervisedAframe):
    """Jax supervised model with hard negative mining.

    BCE is computed over all positives and the ``n_hard_negs`` highest-scoring
    background samples per batch. Easy negatives are excluded from the
    gradient, forcing the model to improve its ranking at the hard end.
    """

    def __init__(self, n_hard_negs: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.n_hard_negs = n_hard_negs

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_hnm_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.n_hard_negs,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)


class JaxPAUCSupervisedAframe(JaxSupervisedAframe):
    """Jax supervised model trained with a pAUC surrogate loss.

    Optimises a squared-hinge pairwise loss over (positive, hard-negative)
    pairs where hard negatives are the ``n_hard_negs`` highest-scoring
    background samples in each batch. This directly targets the low-FPR
    regime of the ROC curve rather than the full AUC.

    The ``margin`` parameter sets the desired score gap: a positive must
    outscore each hard negative by at least ``margin`` for zero loss.
    """

    def __init__(self, n_hard_negs: int = 16, margin: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.n_hard_negs = n_hard_negs
        self.margin = margin

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_pauc_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.n_hard_negs,
            self.margin,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)


class JaxFocalHNMSupervisedAframe(JaxSupervisedAframe):
    """Focal loss combined with hard negative mining.

    Applies focal down-weighting (concentrates on hard examples globally)
    AND restricts the loss to the top-``n_hard_negs`` hardest negatives per
    batch (ignores trivially easy negatives entirely).
    """

    def __init__(self, gamma: float = 2.0, n_hard_negs: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.n_hard_negs = n_hard_negs

    def training_step(self, batch):
        X_time, y = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_focal_hnm_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            self.gamma,
            self.n_hard_negs,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)


class JaxSNRWeightedSupervisedAframe(JaxSupervisedAframe):
    """Jax supervised model with SNR-weighted training loss.

    The per-sample training loss is weighted as follows:

    - **Signal** (y == 1): ``weight = snr ^ snr_weight_power``
    - **Background** (y == 0): ``weight = fp_weight``

    Weights are normalised by their batch mean so the loss magnitude
    stays comparable to unweighted BCE regardless of hyperparameter
    choices.

    Setting ``fp_weight > 1`` makes the model penalise false positives
    more heavily relative to the average-SNR signal.  Setting
    ``snr_weight_power > 0`` additionally up-weights high-SNR signals so
    the model is encouraged to detect events that clearly exceed the noise
    floor.

    """

    def __init__(
        self,
        fp_weight: float = 5.0,
        snr_weight_power: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fp_weight = fp_weight
        self.snr_weight_power = snr_weight_power

    def training_step(self, batch):
        X_time, y, snr_weights = batch
        X_time = jnp.asarray(X_time.cpu().numpy())
        y = jnp.asarray(y.cpu().numpy())
        snr_weights = jnp.asarray(snr_weights.cpu().numpy())

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X_time.shape[0]
        keys = jr.split(k, batch_size)

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_snr_weighted_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_time,
            y,
            snr_weights,
            self.fp_weight,
            self.snr_weight_power,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        loss = metrics["loss"].item()
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
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

        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

        return torch.tensor(0.0)
