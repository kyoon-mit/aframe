from train.data.supervised.denoiser import DenoiserOnlyAframeDataset
from train.data.supervised.supervised import SupervisedAframeDataset


class PsdDenoiserDataset(DenoiserOnlyAframeDataset):
    """``DenoiserOnlyAframeDataset`` that also returns the batch's PSDs.

    Batches are ``(X, X_clean, y, params, psds)``. The PSDs are the ones
    the strain was whitened with; the model needs them to reproduce the
    same highpass on its output. Validation is injected through this
    method as well, so both loaders carry them.
    """

    def inject(self, X, waveforms, params):
        X, y, psds, params_out = SupervisedAframeDataset.inject(
            self, X=X, waveforms=waveforms, params=params
        )
        X = self.apply_transforms(X, psds)
        X_clean = self.apply_transforms(self._clean_signal, psds)
        return X, X_clean, y, params_out, psds
