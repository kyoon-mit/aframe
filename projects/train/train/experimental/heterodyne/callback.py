"""Plot callbacks that hand the denoiser its heterodyne conditioning.

The plain plot callbacks call the module on a stored batch of strain,
which is all an ordinary denoiser needs. A heterodyne denoiser also needs
the chirp mass and the coalescence time of every row it is given, so
these subclasses supply them around the inherited drawing code.
"""

from typing import Optional

import torch

from train.callbacks import DenoiserEvolutionCallback, ReferenceEventCallback


class _Conditioned:
    """Set the module's conditioning for the duration of a plot."""

    def _params_for(self, pl_module):
        raise NotImplementedError

    def _plot(self, super_method, trainer, pl_module):
        params = self._params_for(pl_module)
        if params is None:
            return
        pl_module.condition(params)
        try:
            super_method(trainer, pl_module)
        finally:
            pl_module._conditioning = None

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._plot(super().on_validation_epoch_end, trainer, pl_module)

    def on_test_end(self, trainer, pl_module) -> None:
        self._plot(super().on_test_end, trainer, pl_module)


class HeterodyneEvolutionCallback(_Conditioned, DenoiserEvolutionCallback):
    """Evolution plots on the first training batch, with its own params."""

    _params: Optional[dict] = None

    def on_train_batch_start(
        self, trainer, pl_module, batch, batch_idx
    ) -> None:
        if self._fixed_batch is None and len(batch) >= 4:
            n = min(self.n_examples, batch[0].shape[0])
            self._params = {
                key: value[:n].detach().clone()
                for key, value in batch[3].items()
            }
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

    def _params_for(self, pl_module):
        return self._params


class HeterodyneReferenceEventCallback(_Conditioned, ReferenceEventCallback):
    """Reference-event plots, conditioned on the file's own injection.

    Every event in the file is the same waveform at a different
    signal-to-noise ratio, so one chirp mass covers them all, and the
    merger sits at the ``merger_index`` the file records.

    Args:
        chirp_mass: of the injected waveform, in solar masses.
    """

    def __init__(self, *args, chirp_mass: float, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.chirp_mass = chirp_mass

    def _params_for(self, pl_module):
        if self._fixed_batch is None:
            return None
        n = self._fixed_batch[0].shape[0]
        device = pl_module.device
        return {
            "chirp_mass": torch.full(
                (n,), self.chirp_mass, device=device, dtype=torch.float32
            ),
            "coalescence_time": torch.full(
                (n,),
                self.merger_index / self.sample_rate,
                device=device,
                dtype=torch.float32,
            ),
        }
