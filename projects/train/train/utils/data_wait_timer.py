import logging
import time

from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class DataWaitTimer(Callback):
    """
    Measures and logs how long each training/validation
    step waits for the next batch.

    The "data wait" is the wall-clock gap between the end of step N and the
    start of step N+1 — i.e. the time the training loop is blocked on the
    dataloader.  A running exponential-moving-average is logged every step;
    a warning is emitted whenever a single wait exceeds `warn_threshold_s`.
    """

    def __init__(self, warn_threshold_s: float = 0.1):
        """
        Args:
            warn_threshold_s: Emit a WARNING if a single data-wait exceeds this
                              many seconds (default 100 ms).
        """
        super().__init__()
        self.warn_threshold_s = warn_threshold_s

        self._train_batch_end_t: float | None = None
        self._val_batch_end_t: float | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._train_batch_end_t is None:
            return
        wait = time.perf_counter() - self._train_batch_end_t

        pl_module.log(
            "train/data_wait_s",
            wait,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        if wait > self.warn_threshold_s:
            log.warning(
                "DL bottleneck: training step %d waited %.3f s for next batch "
                "(threshold %.3f s)",
                trainer.global_step,
                wait,
                self.warn_threshold_s,
            )

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        self._train_batch_end_t = time.perf_counter()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if self._val_batch_end_t is None:
            return
        wait = time.perf_counter() - self._val_batch_end_t

        pl_module.log(
            "val/data_wait_s",
            wait,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        if wait > self.warn_threshold_s:
            log.warning(
                "DL bottleneck: val batch %d waited %.3f s for next batch"
                "(threshold %.3f s)",
                batch_idx,
                wait,
                self.warn_threshold_s,
            )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self._val_batch_end_t = time.perf_counter()

    # Reset at epoch boundaries so inter-epoch gaps aren't counted as data-wait
    def on_train_epoch_start(self, trainer, pl_module):
        self._train_batch_end_t = None

    def on_validation_epoch_start(self, trainer, pl_module):
        self._val_batch_end_t = None
