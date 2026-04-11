import logging
import time

from lightning.pytorch.callbacks import Callback

log = logging.getLogger(__name__)


class SamplesPerSecondTimer(Callback):
    """Measures and logs model throughput in samples per second.

    This calculates the time taken for a full batch step (forward, backward,
    optimizer step) by measuring the time between batch start and end.
    """

    def __init__(self):
        super().__init__()

        self._train_batch_start_t: float | None = None
        self._val_batch_start_t: float | None = None

    def _get_batch_size(self, batch) -> int:
        if isinstance(batch, (tuple, list)):
            if len(batch) > 0 and hasattr(batch[0], "shape"):
                return batch[0].shape[0]
            elif len(batch) > 0:
                try:
                    return len(batch[0])
                except TypeError:
                    return 1
            return 1
        elif isinstance(batch, dict):
            # In validation, we might receive lists/tuples of
            # length batch_size containing dictionary objects
            val = next(iter(batch.values()))
            if isinstance(val, (list, tuple)):
                val = len(val)
            elif hasattr(val, "shape"):  # array
                val = val.shape[0]
            else:
                try:
                    val = len(val)
                except TypeError:
                    val = 1
            return val
        elif hasattr(batch, "shape"):
            return batch.shape[0]
        else:
            try:
                return len(batch)
            except AttributeError:
                pass
            except TypeError:
                pass
        return 1

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._train_batch_start_t = time.perf_counter()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if self._train_batch_start_t is None:
            return

        duration = time.perf_counter() - self._train_batch_start_t
        batch_size = self._get_batch_size(batch)
        throughput = batch_size / duration

        pl_module.log(
            "train/samples_per_second",
            throughput,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        self._val_batch_start_t = time.perf_counter()

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if self._val_batch_start_t is None:
            return

        duration = time.perf_counter() - self._val_batch_start_t
        batch_size = self._get_batch_size(batch)
        throughput = batch_size / duration

        pl_module.log(
            "val/samples_per_second",
            throughput,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
