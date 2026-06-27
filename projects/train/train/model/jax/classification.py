import logging

import equinox as eqx
import jax
import jax.random as jr
import numpy as np
import optax
import torch

from train.metric import LightningStepOutput
from train.metrics import TimeSlideAUROC
from train.model.classification import AframeClassification
from architectures.base import Architecture, JaxArchitecture

from train.utils.jax.param_count import print_param_tree
from train.utils.jax.convert import tensor_to_jax_array, jax_array_to_tensor
from train.utils.jax.load_model import load_model
from train.utils.jax.training import (
    jax_apply_training_step,
    jax_apply_segment_training_step,
    jax_inference,
    pool_window_logits,
)


logger = logging.getLogger(__name__)


class JaxClassificationAframe(AframeClassification):
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
        segment_window_length: float | None = None,
        segment_num_windows: int = 1,
        segment_pool: str = "logsumexp",
        segment_pool_temperature: float = 0.1,
        consistency_weight: float = 0.0,
        consistency_type: str = "tv",
    ):
        super().__init__(
            arch=Architecture(),
            metric=metric,
            learning_rate=learning_rate,
            pct_lr_ramp=pct_lr_ramp,
            weight_decay=weight_decay,
        )
        self.automatic_optimization = False

        # Segment-level (sliding-window) training/eval. When
        # ``segment_window_length`` is set, the model is fed a kernel longer
        # than its native window, unfolds it into ``segment_num_windows``
        # overlapping sub-windows, scores each, and integrates/pools the
        # per-window logits into a single segment logit (see
        # ``train.utils.jax.training``). ``None`` disables it (single-window
        # behaviour, fully backward compatible).
        self.segment_window_length = segment_window_length
        self.segment_num_windows = segment_num_windows
        self.segment_pool = segment_pool
        self.segment_pool_temperature = segment_pool_temperature
        self.consistency_weight = consistency_weight
        self.consistency_type = consistency_type
        self._segment_enabled = segment_window_length is not None
        # The window length in samples is derived lazily from the datamodule's
        # (model) sample rate the first time it is needed -- this avoids
        # duplicating sample_rate on the model (which would also collide with
        # the datamodule's ``sample_rate`` hparam at logging time) and stays
        # correct under optional resampling (``model_sample_rate``).
        self._window_samples = None

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

    def score(self, X: torch.Tensor) -> torch.Tensor:
        return self(X)

    def _advance_optimizers(self):
        # Needed for lightning when self.automatic_optimization = False
        optimizers = self.optimizers()
        if isinstance(optimizers, (list, tuple)):
            for opt in optimizers:
                opt.step()
        else:
            optimizers.step()

    def _get_window_samples(self) -> int:
        """Window length in samples, derived from the datamodule's rate."""
        if self._window_samples is None:
            sr = self.trainer.datamodule.model_sample_rate
            self._window_samples = int(round(self.segment_window_length * sr))
        return self._window_samples

    def _unfold(self, X: torch.Tensor) -> torch.Tensor:
        """Unfold a long kernel ``(B, C, T)`` into the batch axis.

        Returns ``(B * N, C, W)`` in sample-major order
        (``[s0w0, s0w1, ..., s1w0, ...]``) so per-window logits reshape back to
        ``(B, N)``. The ``N`` windows of length ``W`` start at evenly spaced
        integer offsets from 0 to ``T - W``.
        """
        B, C, T = X.shape
        W = self._get_window_samples()
        N = self.segment_num_windows
        if T < W:
            raise ValueError(
                f"kernel length {T} samples is shorter than the segment "
                f"window {W} samples (segment_window_length="
                f"{self.segment_window_length}s)"
            )
        if N == 1:
            starts = [0]
        else:
            starts = [round(i * (T - W) / (N - 1)) for i in range(N)]
        windows = torch.stack([X[:, :, s : s + W] for s in starts], dim=1)
        return windows.reshape(B * N, C, W)

    def _segment_forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.shape[0]
        N = self.segment_num_windows
        X_win = tensor_to_jax_array(self._unfold(X))

        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X_win.shape[0])

        logits, new_state = jax_inference(
            self.jax_model, X_win, self.jax_model_state, keys
        )
        self.jax_model_state = new_state

        logits_bn = logits.reshape(B, N)
        pooled = pool_window_logits(
            logits_bn, self.segment_pool, self.segment_pool_temperature
        )
        return jax_array_to_tensor(pooled)

    def _segment_training_step(self, batch):
        X, y, _ = batch
        N = self.segment_num_windows
        X_win = tensor_to_jax_array(self._unfold(X))
        y_j = tensor_to_jax_array(y)

        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, X_win.shape[0])

        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_segment_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            X_win,
            y_j,
            N,
            self.segment_pool,
            self.segment_pool_temperature,
            self.consistency_weight,
            self.consistency_type,
            self.opt_state,
            self.optimizer.update,
            keys,
        )

        for name in ("loss", "bce", "consistency"):
            self.log(
                f"train/{name}",
                metrics[name].item(),
                on_step=True,
                on_epoch=True,
                prog_bar=(name == "loss"),
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

        self._advance_optimizers()
        return torch.tensor(0.0)

    def training_step(self, batch):
        if self._segment_enabled:
            return self._segment_training_step(batch)

        batch = tensor_to_jax_array(batch)
        X, y, params = batch

        self.rng_key, k = jr.split(self.rng_key)
        batch_size = X.shape[0]
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
            X,
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

        self._advance_optimizers()

        return torch.tensor(0.0)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self._segment_enabled:
            return self._segment_forward(X)

        X = tensor_to_jax_array(X)

        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X.shape[0])

        logits, new_state = jax_inference(
            self.jax_model, X, self.jax_model_state, keys
        )
        self.jax_model_state = new_state

        return jax_array_to_tensor(logits)

    def validation_step(self, batch, _):
        shift, X_bg, X_inj, params = batch

        y_bg = self.forward(X_bg)

        num_views, batch_size, *shape = X_inj.shape
        X_inj_flat = X_inj.view(num_views * batch_size, *shape)

        y_fg_flat = self.forward(X_inj_flat)
        y_fg = y_fg_flat.view(num_views, batch_size).mean(0)

        self.metric.update(shift, y_bg, y_fg)
        metric_name = self.metric.__class__.__name__
        self.log(
            f"validation/{metric_name}",
            self.metric,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        y_bg_np = np.array(y_bg.cpu()).flatten()
        y_fg_np = np.array(y_fg.cpu()).flatten()
        params_np = {k: v.cpu().numpy() for k, v in params.items()}

        return LightningStepOutput(
            targets=np.concatenate(
                [np.zeros(len(y_bg_np)), np.ones(len(y_fg_np))]
            ),
            outputs=np.concatenate([y_bg_np, y_fg_np]),
            bg_outputs=y_bg_np,
            params=params_np,
        )
