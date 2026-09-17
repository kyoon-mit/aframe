import h5py
import torch

from train.callbacks import ReferenceEventCallback


class PsdReferenceEventCallback(ReferenceEventCallback):
    """``ReferenceEventCallback`` that hands each event's PSD to the model.

    Reads the ``psds`` dataset the events file carries and sets it on the
    model around the forward pass, the way the training step does, so a
    ``HighpassDenoiser`` filters the plotted output with the same response
    it trains with.
    """

    def on_fit_start(self, trainer, pl_module) -> None:
        super().on_fit_start(trainer, pl_module)
        with h5py.File(self.events_file) as handle:
            if "psds" not in handle:
                raise KeyError(
                    f"{self.events_file} has no psds dataset; rebuild it "
                    "with the current build_plot_events.py"
                )
            psds = torch.tensor(handle["psds"][:], dtype=torch.float64)
        self._psds = psds.to(pl_module.device)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not hasattr(pl_module, "_psds"):
            return super().on_validation_epoch_end(trainer, pl_module)
        pl_module._psds = self._psds
        try:
            return super().on_validation_epoch_end(trainer, pl_module)
        finally:
            pl_module._psds = None
