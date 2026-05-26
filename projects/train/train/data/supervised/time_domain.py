import torch

from train.data.supervised.supervised import SupervisedAframeDataset


class TimeDomainSupervisedAframeDataset(SupervisedAframeDataset):
    def build_val_batches(self, background, signals):
        X_bg, X_inj, psds = super().build_val_batches(background, signals)
        X_bg = self.whitener(X_bg, psds)
        # whiten each view of injections
        X_fg = []
        for inj in X_inj:
            inj = self.whitener(inj, psds)
            X_fg.append(inj)

        X_fg = torch.stack(X_fg)
        return X_bg, X_fg

    def apply_transforms(self, X, psds):
        return self.whitener(X, psds)

    def inject(self, X, waveforms=None, params=None):
        X, y, psds, params_out = super().inject(X, waveforms, params)
        X = self.apply_transforms(X, psds)
        if params is not None:
            return X, y, params_out
        return X, y
