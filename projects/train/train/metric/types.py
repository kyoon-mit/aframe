import gc
from dataclasses import dataclass
from typing import Any, Callable, Literal, Protocol, TypedDict

import numpy as np
import wandb
from lightning.pytorch import Callback, LightningModule, Trainer
from torch.utils._pytree import tree_flatten, tree_map
from jaxtyping import PyTree

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BatchedTarget = np.ndarray
BatchedParams = dict[str, np.ndarray]


class LightningStepOutput(TypedDict, total=False):
    targets: Any
    outputs: Any
    params: Any
    bg_outputs: Any


@dataclass
class Loggable:
    value: float | np.ndarray | dict

    def log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        name: str,
        prog_bar: bool,
        batch_size: int,
    ):
        Loggable._log(
            self.value, trainer, pl_module, name, prog_bar, batch_size
        )

    @staticmethod
    def _log(
        value,
        trainer: Trainer,
        pl_module: LightningModule,
        name: str,
        prog_bar: bool,
        batch_size: int,
    ):
        if isinstance(value, Loggable):
            value.log(trainer, pl_module, name, prog_bar, batch_size)
        elif isinstance(value, dict):
            for k, v in value.items():
                Loggable._log(
                    v, trainer, pl_module, f"{name}/{k}", prog_bar, batch_size
                )
        else:
            try:
                value = value.item()
            except AttributeError:
                pass
            pl_module.log(
                name, value, prog_bar=prog_bar, batch_size=batch_size
            )


@dataclass
class ImageLog(Loggable):
    caption: str | None

    def log(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        name: str,
        prog_bar: bool,
        batch_size: int,
    ):
        trainer.logger.experiment.log(
            {name: [wandb.Image(self.value, caption=self.caption)]},
            step=pl_module.global_step,
        )


class Metric(Protocol):
    """A protocol defining the signature for a metric function."""

    def __call__(
        self,
        target: BatchedTarget | PyTree | None = None,
        pred: BatchedTarget | PyTree | None = None,
        params: BatchedParams | None = None,
        bg_pred: BatchedTarget | PyTree | None = None,
        **kwargs: Any,
    ) -> Loggable | np.ndarray | float: ...


class ReductionFunction(Protocol):
    """A protocol defining the signature for a reduction function."""

    def __call__(
        self,
        target: BatchedTarget | PyTree | None = None,
        pred: BatchedTarget | PyTree | None = None,
        params: BatchedParams | None = None,
        **kwargs: Any,
    ) -> Loggable | np.ndarray | float: ...


MetricType = Literal["Accumulated", "PerBatch"]

# ---------------------------------------------------------------------------
# Metric decorator & registry
# ---------------------------------------------------------------------------

_METRIC_REGISTRY: dict[str, Callable] = {}


def metric(
    type: MetricType = "PerBatch",
    stages: tuple[str, ...] = ("train", "val", "test"),
    prog_bar: bool = False,
    every_n_steps: int | None = None,
    keys: list[str] | None = None,
) -> Callable:
    """Decorator that registers a plain function as a Lightning metric.

    The decorated function is returned **unchanged** — it stays a normal
    callable so metric functions can call each other directly (e.g.
    ``f1_score`` calling ``true_positive_rate``).  Metadata is attached
    as private attributes so ``make_callback`` can create a ``CustomMetric``
    Lightning callback from it.

    Example::

        @metric(type="Accumulated", stages=("val", "test"))
        def auc(target, pred, **kwargs):
            return roc_auc_score(target, pred[:, -1])

        # still works as a plain function:
        score = auc(y_true, y_pred)

        # create a Lightning callback:
        cb = make_callback(auc)
    """

    def decorator(fn: Callable) -> Callable:
        fn._metric_type = type
        fn._metric_stages = stages
        fn._metric_prog_bar = prog_bar
        fn._metric_every_n_steps = every_n_steps
        fn._metric_keys = keys
        fn._is_metric = True
        _METRIC_REGISTRY[fn.__name__] = fn
        return fn

    return decorator


def acc_metric(
    stages: tuple[str, ...] = ("val", "test"),
    prog_bar: bool = False,
    every_n_steps: int | None = None,
    keys: list[str] | None = None,
) -> Callable:
    """Shorthand for ``@metric(type='Accumulated', ...)``."""
    return metric(
        type="Accumulated",
        stages=stages,
        prog_bar=prog_bar,
        every_n_steps=every_n_steps,
        keys=keys,
    )


def _get_batch_size(pytree) -> int:
    """Get batch size from the first leaf of a pytree."""
    if isinstance(pytree, np.ndarray):
        return pytree.shape[0]
    leaves, _ = tree_flatten(pytree)
    return leaves[0].shape[0]


class CustomMetric(Callback):
    """Helper class to handle metric logging for each head."""

    acc_outputs: list[LightningStepOutput]

    def __init__(
        self,
        metric: Metric,
        metric_name: str,
        keys: list[str] | None = None,
        prog_bar: bool = False,
        type: MetricType = "PerBatch",
        every_n_steps: int | None = None,
        stages: tuple[str, ...] = ("train", "val", "test"),
        reduce_fn: ReductionFunction | None = None,
    ):
        super().__init__()
        self.metric = metric
        self.metric_name = metric_name
        self.keys = keys  # output keys this metric applies to; None means all
        self.prog_bar = prog_bar
        self.type = type
        self.every_n_steps = every_n_steps
        self.stages = stages
        self.reduce_fn = reduce_fn

        self.acc_outputs = []

    def log_metric(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LightningStepOutput,
        stage: Literal["train", "val", "test"],
    ) -> None:
        if not isinstance(outputs, dict):
            return
        if "targets" not in outputs or "outputs" not in outputs:
            return
        params = outputs.get("params", None)
        targets = outputs["targets"]
        model_outputs = outputs["outputs"]
        model_bg_outputs = outputs.get("bg_outputs", None)

        if isinstance(model_outputs, dict):
            # Apply metric per output key. self.keys restricts which keys are
            # used; None means apply to all keys present in model_outputs.
            keys_to_use = (
                self.keys
                if self.keys is not None
                else list(model_outputs.keys())
            )
            for key in keys_to_use:
                if key not in model_outputs:
                    continue
                head_out = model_outputs[key]
                if model_bg_outputs is not None:
                    head_out_bg = model_bg_outputs[key]
                else:
                    head_out_bg = None

                head_target = (
                    targets[key]
                    if isinstance(targets, dict) and key in targets
                    else targets
                )
                try:
                    metric_value = self.metric(
                        target=head_target,
                        pred=head_out,
                        bg_pred=head_out_bg,
                        params=params,
                    )
                    Loggable._log(
                        metric_value,
                        trainer,
                        pl_module,
                        f"{stage}/{key}/{self.metric_name}",
                        self.prog_bar,
                        batch_size=_get_batch_size(head_target),
                    )
                except Exception as e:
                    print(
                        f"Error logging metric "
                        f"{self.metric_name} for key {key}: {e}"
                    )
        else:
            try:
                metric_value = self.metric(
                    target=targets,
                    pred=model_outputs,
                    params=params,
                )
                Loggable._log(
                    metric_value,
                    trainer,
                    pl_module,
                    f"{stage}/{self.metric_name}",
                    self.prog_bar,
                    batch_size=_get_batch_size(targets),
                )
            except Exception as e:
                print(f"Error logging metric {self.metric_name}: {e}")

    def _on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: LightningStepOutput,
        batch_idx: int,
        stage: Literal["train", "val", "test"],
    ):
        # make all tensors in output numpy arrays

        if stage not in self.stages:
            return
        if self.type == "Accumulated" or stage in ("test", "val"):
            self.acc_outputs.append(outputs)
        else:
            outputs = tree_map(lambda x: np.array(x), outputs)
            if (
                self.every_n_steps is not None
                and batch_idx % self.every_n_steps != 0
            ):
                return
            self.log_metric(
                trainer,
                pl_module,
                outputs,
                stage=stage,
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self._on_batch_end(
            trainer, pl_module, outputs, batch_idx, stage="train"
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._on_batch_end(trainer, pl_module, outputs, batch_idx, stage="val")

    def on_test_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._on_batch_end(
            trainer, pl_module, outputs, batch_idx, stage="test"
        )

    def _on_epoch_end(
        self, trainer, pl_module, stage: Literal["train", "val", "test"]
    ):
        if stage not in self.stages:
            return
        if self.type == "Accumulated" or stage in ("val", "test"):
            acc_outputs = tree_map(
                lambda x: np.array(x) if x is not None else None,
                self.acc_outputs,
            )
            combined_outputs = tree_map(
                lambda *x: (
                    np.concatenate(x, axis=0) if x[0] is not None else None
                ),
                *acc_outputs,
            )

            self.log_metric(
                trainer,
                pl_module,
                outputs=combined_outputs,
                stage=stage,
            )
        self.acc_outputs = []
        gc.collect()

    def on_train_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, stage="train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, stage="val")

    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, stage="test")


def make_callback(fn: Callable) -> CustomMetric:
    """Create a :class:`CustomMetric` Lightning callback from a
    ``@metric()``-decorated function.

    Example::

        trainer = Trainer(callbacks=[make_callback(auc), make_callback(mse)])
    """
    if not getattr(fn, "_is_metric", False):
        raise ValueError(
            f"{fn.__name__!r} is not decorated with @metric(). "
            "Decorate it first or pass the metadata explicitly "
            "to CustomMetric."
        )
    return CustomMetric(
        metric=fn,
        metric_name=fn.__name__,
        type=fn._metric_type,
        stages=fn._metric_stages,
        prog_bar=fn._metric_prog_bar,
        every_n_steps=fn._metric_every_n_steps,
        keys=fn._metric_keys,
    )
