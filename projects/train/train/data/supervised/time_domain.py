import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals, params=None):
        X_bg, X_inj, psds, params = super().build_val_batches(
            background, signals, params
        )
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg, params

    def inject(self, X, waveforms=None):
        X, y, psds = super().inject(X, waveforms)
        X = self.whitener(X, psds)
        return X, y


class SNRWeightedTimeDomainDataset(TimeDomainSupervisedAframeDataset):
    """Time-domain dataset that returns per-sample SNR weights alongside
    (X, y) so the model can weight the training loss by injection SNR
    and apply an asymmetric penalty for false positives.

    The third element of the training batch is a ``(batch_size,)`` float
    tensor where each entry is the injected SNR for true-signal samples
    and ``0`` for background / augmented (swapped/muted) samples.
    """

    def inject(self, X, waveforms=None):
        X, y = super().inject(X, waveforms)
        # _train_snr_weights is set by SupervisedAframeDataset.inject
        snr_weights = self._train_snr_weights
        return X, y, snr_weights
