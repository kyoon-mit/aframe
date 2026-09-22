"""Denoiser batches that also carry the merger time of each row.

Heterodyning needs both the chirp mass and the coalescence time, and the
chirp mass alone comes out of the parameter transforms. The coalescence
time is set by the random kernel offset that ``inject`` draws and then
throws away, so this dataset records it and returns it as one more entry
in ``params``.
"""

from train.data.supervised import supervised as supervised_module
from train.data.supervised.denoiser import DenoiserOnlyAframeDataset


class HeterodyneDenoiserDataset(DenoiserOnlyAframeDataset):
    """``DenoiserOnlyAframeDataset`` plus a per-row coalescence time.

    ``params["coalescence_time"]`` is the merger's position in the
    whitened kernel, in seconds from its left edge. Every row must carry
    a signal for it to be defined, so ``waveform_prob`` must be 1 and the
    swap and mute augmentations must be off.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the CLI links this to the network's input width, and the network
        # sees an in-phase and a quadrature channel per interferometer
        self.num_ifos = 2 * len(self.hparams.ifos)

    def inject(self, X, waveforms, params):
        if (
            self.hparams.waveform_prob != 1.0
            or self.hparams.swap_prob != 0.0
            or self.hparams.mute_prob != 0.0
        ):
            raise ValueError(
                "Heterodyning needs a merger in every row: set "
                "waveform_prob to 1 and swap_prob and mute_prob to 0"
            )

        # the kernel offset is drawn inside the base inject, which returns
        # only the kernels, so record it as it is drawn
        offsets = {}
        original = supervised_module.sample_kernels

        def recording(*args, **kwargs):
            kernels, idx = original(*args, return_idx=True, **kwargs)
            offsets["idx"] = idx
            return kernels

        supervised_module.sample_kernels = recording
        try:
            X, X_clean, y, params_out = super().inject(X, waveforms, params)
        finally:
            supervised_module.sample_kernels = original

        # slice_waveforms puts the merger at this index of the sliced
        # response, and whitening then trims half a filter from the left
        unwhitened_kernel_size = (
            int(self.hparams.kernel_length * self.hparams.sample_rate)
            + self.filter_size
        )
        merger_in_response = unwhitened_kernel_size - self.right_pad_size
        merger = merger_in_response - offsets["idx"].to(X.device)
        merger = merger - self.filter_size // 2
        params_out["coalescence_time"] = (
            merger.float() / self.hparams.sample_rate
        )
        return X, X_clean, y, params_out
