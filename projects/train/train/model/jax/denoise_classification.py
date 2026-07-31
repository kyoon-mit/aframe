from typing import Optional

import jax.random as jr

from architectures.base import JaxArchitecture
from train.metrics import TimeSlideAUROC
from train.model.jax.classification import JaxClassificationAframe
from train.utils.jax.convert import jax_array_to_tensor, tensor_to_jax_array
from train.utils.jax.training import (
    jax_apply_denoise_cls_training_step,
    jax_inference,
)


class JaxDenoiseClassificationAframe(JaxClassificationAframe):
    """JAX LinOSS joint denoiser + detection classifier.

    Arch returns ``(x_denoised, logits)``. Trains on
    ``BCE(logits, y) + lambda_denoise * MSE(x_denoised, X_clean)`` using the
    ``DenoisingTimeDomainSupervisedAframeDataset`` batch
    ``(X, X_clean, y, params)``. Validation uses the logit head only, so the
    inherited ``TimeSlideAUROC`` path is unchanged.

    Explicit signature (not ``**kwargs``) so the LightningCLI can build the
    nested ``lr_scheduler`` / ``metric`` classes.
    """

    def __init__(
        self,
        arch: JaxArchitecture,
        metric: TimeSlideAUROC,
        metric_full: Optional[TimeSlideAUROC] = None,
        lambda_denoise: float = 1.0,
        learning_rate: float = 1e-4,
        ssm_lr: Optional[float] = None,
        weight_decay: float = 0.0,
        clip_grad_norm: float = 10.0,
        max_steps: int = 500_000,
        warmup_steps: int = 1000,
        lr_scheduler=None,
        lr_scheduler_interval: str = "epoch",
        normalize_input: bool = False,
        seed: int = 42,
        load_from_checkpoint: Optional[str] = None,
        reset_optimizer_on_load: bool = True,
    ) -> None:
        super().__init__(
            arch,
            metric=metric,
            metric_full=metric_full,
            learning_rate=learning_rate,
            ssm_lr=ssm_lr,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            lr_scheduler=lr_scheduler,
            lr_scheduler_interval=lr_scheduler_interval,
            normalize_input=normalize_input,
            seed=seed,
            load_from_checkpoint=load_from_checkpoint,
            reset_optimizer_on_load=reset_optimizer_on_load,
        )
        self.lambda_denoise = lambda_denoise

    def forward(self, X):
        X = tensor_to_jax_array(self._maybe_normalize(X))
        self.rng_key, step_k = jr.split(self.rng_key)
        keys = jr.split(step_k, X.shape[0])
        outputs, new_state = jax_inference(
            self.jax_model, X, self.jax_model_state, keys
        )
        self.jax_model_state = new_state
        _, logits = outputs
        return jax_array_to_tensor(logits)

    def training_step(self, batch):
        X, X_clean, y, _ = batch
        Xj = tensor_to_jax_array(self._maybe_normalize(X))
        Xcj = tensor_to_jax_array(X_clean)
        yj = tensor_to_jax_array(y)

        lr_other, lr_ssm = self._current_lrs()
        self.rng_key, k = jr.split(self.rng_key)
        keys = jr.split(k, Xj.shape[0])
        (
            self.jax_model,
            self.jax_model_state,
            self.opt_state,
            metrics,
        ) = jax_apply_denoise_cls_training_step(
            self.jax_model,
            self.jax_model_filter_spec,
            self.jax_model_state,
            Xj,
            Xcj,
            yj,
            self.lambda_denoise,
            self.opt_state,
            self.optimizer.update,
            lr_other,
            lr_ssm,
            keys,
        )
        self.log(
            "train/loss",
            float(metrics["loss"]),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("train/bce", float(metrics["bce"]), on_epoch=True)
        self.log(
            "train/loss_denoise", float(metrics["denoise"]), on_epoch=True
        )
        self.optimizers().step()
        return None
