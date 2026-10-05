from typing import Optional

import torch
from ml4gw.utils.slicing import sample_kernels

from train import augmentations as aug
from train.data.base import BaseAframeDataset


class SupervisedAframeDataset(BaseAframeDataset):
    def __init__(
        self,
        *args,
        swap_prob: Optional[float] = None,
        mute_prob: Optional[float] = None,
        signal_repeats: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            signal_repeats: number of training samples that share one
                injected signal, each with its own background.
        """
        super().__init__(*args, **kwargs)
        if signal_repeats < 1:
            raise ValueError(
                f"signal_repeats must be >= 1, got {signal_repeats}"
            )
        self.signal_repeats = signal_repeats
        if swap_prob is not None and 0 <= swap_prob <= 1:
            self.swapper = aug.ChannelSwapper(swap_prob)
            self.swap_prob = swap_prob
        elif swap_prob is not None:
            raise ValueError(
                f"swap_prob must be between 0 and 1, got {swap_prob}"
            )
        else:
            self.swapper = None
            self.swap_prob = 0

        if mute_prob is not None and 0 <= mute_prob <= 1:
            self.muter = aug.ChannelMuter(mute_prob)
            self.mute_prob = mute_prob
        elif mute_prob is not None:
            raise ValueError(
                f"mute_frac must be between 0 and 1, got {mute_prob}"
            )
        else:
            self.muter = None
            self.mute_prob = 0

    def _active_snr_sampler(self):
        """The validation sampler while validating, if one is set."""
        trainer = self.trainer
        validating = trainer.validating or trainer.sanity_checking
        if validating and self.val_snr_sampler is not None:
            return self.val_snr_sampler
        return self.snr_sampler

    @property
    def waveforms_per_batch(self) -> int:
        return -(-self.hparams.batch_size // self.signal_repeats)

    @property
    def sample_prob(self):
        return self.hparams.waveform_prob + self.swap_prob + self.mute_prob

    @torch.no_grad()
    def inject(
        self, X, waveforms, params
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
    ]:
        if waveforms is None:
            raise ValueError(
                "Waveforms should be passed to the `inject` method, got None"
            )

        X, psds = self.psd_estimator(X)
        X = self.inverter(X)
        X = self.reverser(X)
        # sample enough waveforms to do true injections,
        # swapping, and muting

        rvs = torch.rand(size=X.shape[:1], device=X.device)
        mask = rvs < self.sample_prob

        # an all-background batch (possible when sample_prob < 1) would
        # send empty tensors through the projector and crash cuFFT
        if not mask.any():
            y = torch.zeros((X.size(0), 1), device=X.device)
            params_out = {
                key: torch.full((X.size(0),), float("nan"), device=X.device)
                for key in list(params) + ["dec", "psi", "phi", "snr"]
            }
            params_out = self.apply_param_transforms(params_out)
            # clean (noise-free) target for the denoising task: all zeros
            # here since no signal was injected
            self._clean_signal = torch.zeros_like(X)
            return X, y, psds, params_out

        # project each distinct signal once, then copy it into R samples,
        # each with its own background
        N = mask.sum().item()
        R = self.signal_repeats if self.trainer.training else 1
        n = -(-N // R)
        dec, psi, phi = self.sample_extrinsic(X[:n])
        idx = torch.randperm(waveforms.shape[0])[:n]
        waveforms = waveforms[idx].to(X.device).float()
        params = {k: v[idx].to(X.device).float() for k, v in params.items()}
        snrs = self._active_snr_sampler().sample((n,)).to(X.device)
        params.update(dec=dec, psi=psi, phi=phi, snr=snrs)

        responses = self.projector(
            dec, psi, phi, cross=waveforms[:, 0], plus=waveforms[:, 1]
        )
        if R > 1:
            responses = responses.repeat_interleave(R, dim=0)[:N]
            params = {k: v.repeat_interleave(R)[:N] for k, v in params.items()}
        responses = self.projector.rescaler(
            responses, psds[mask], params["snr"]
        )

        # If we're loading waveforms from disk, we'll have sliced
        # the waveforms already in `on_before_batch_transfer`
        if not self.waveforms_from_disk:
            responses = self.slice_waveforms(responses)
        kernels = sample_kernels(
            responses, kernel_size=X.size(-1), coincident=True
        )

        # perform augmentations on the responses themselves,
        # keep track of which indices have been augmented
        swap_indices = mute_indices = []
        idx = torch.where(mask)[0]
        if self.swapper is not None:
            kernels, swap_indices = self.swapper(kernels)
        if self.muter is not None:
            kernels, mute_indices = self.muter(kernels)

        # inject the IFO responses
        X[mask] += kernels

        # stash the clean (noise-free) signal for the denoising task, before
        # `mask` is mutated by swap/mute below; zeros on non-injected rows
        clean = torch.zeros_like(X)
        clean[mask] = kernels
        self._clean_signal = clean

        # make labels, turning off injection mask where
        # we swapped or muted
        mask[idx[swap_indices]] = 0
        mask[idx[mute_indices]] = 0
        y = torch.zeros((X.size(0), 1), device=X.device)
        y[mask] += 1

        # return NaN for params that weren't injected
        still_injected = mask[idx]
        params_out = {}
        for key, vals in params.items():
            out = torch.full((X.size(0),), float("nan"), device=X.device)
            out[idx[still_injected]] = vals[still_injected]
            params_out[key] = out

        params_out = self.apply_param_transforms(params_out)

        return X, y, psds, params_out
