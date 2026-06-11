from typing import Optional, Type, Union

import torch
from ml4gw import gw
from ml4gw.distributions import PowerLaw
from ml4gw.types import BatchTensor


class ChannelSwapper(torch.nn.Module):
    """
    Data augmentation module that randomly swaps channels
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels swapped.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            num = num if not num % 2 else num - 1
            num = max(2, num)
            channel = torch.randint(X.shape[1], size=(num // 2,)).repeat(2)
            # swap channels from the first num / 2 elements with the
            # second num / 2 elements
            indices = torch.arange(num)
            target_indices = torch.roll(indices, shifts=num // 2, dims=0)
            X[indices, channel] = X[target_indices, channel]
        return X, indices


class ChannelMuter(torch.nn.Module):
    """
    Data augmentation module that randomly mutes 1 channel
    of a fraction of batch elements.

    Args:
        frac:
            Fraction of batch that will have channels muted.
    """

    def __init__(self, frac: float = 0.5):
        super().__init__()
        self.frac = frac

    def forward(self, X):
        num = int(X.shape[0] * self.frac)
        indices = []
        if num > 0:
            channel = torch.randint(X.shape[1], size=(num,))
            indices = torch.randint(X.shape[0], size=(num,))
            X[indices, channel] = torch.zeros(
                X.shape[-1], device=X.device, dtype=X.dtype
            )

        return X, indices


class SnrRescaler(torch.nn.Module):
    """
    Module that calculates SNRs of injections relative
    to a given ASD and performs augmentation of the waveform
    dataset by rescaling injections such that they have SNRs
    given by `target_snrs`. If this argument is `None`, each
    injection is randomly matched with and scaled to the SNR
    of a different injection from the batch.
    """

    def __init__(
        self,
        sample_rate: float,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.highpass = highpass
        self.lowpass = lowpass

    def forward(
        self,
        responses: gw.WaveformTensor,
        psds: torch.Tensor,
        target_snrs: Union[BatchTensor, float, None],
    ) -> gw.WaveformTensor:
        # we can either specify one PSD for all batch
        # elements, or a PSD for each batch element
        if psds.ndim > 2 and len(psds) != len(responses):
            raise ValueError(
                "Background PSDs must either be two dimensional "
                "or have a PSD specified for every element in the "
                "batch. Expected {}, found {}".format(
                    len(responses), len(psds)
                )
            )

        # interpolate the number of PSD frequency bins down
        # to the value expected by the shape of the waveforms
        num_freqs = responses.size(-1) // 2 + 1
        if psds.size(-1) != num_freqs:
            if psds.ndim == 2:
                psds = psds[None]
                reshape = True
            else:
                reshape = False

            psds = torch.nn.functional.interpolate(psds, size=(num_freqs,))
            if reshape:
                psds = psds.view(-1, num_freqs)

        # compute the SNRs of the existing signals
        snrs = gw.compute_network_snr(
            responses,
            psds,
            self.sample_rate,
            self.highpass,
            self.lowpass,
        )

        if target_snrs is None:
            # if we didn't specify any target SNRs, then shuffle
            # the existing SNRs of the waveforms as they stand
            idx = torch.randperm(len(snrs))
            target_snrs = snrs[idx]
        elif not isinstance(target_snrs, torch.Tensor):
            # otherwise if we provided just a float, assume
            # that it's a lower bound on the desired SNR levels
            target_snrs = snrs.clamp(target_snrs, 1000)

        # reweight the amplitude of the IFO responses
        # in order to achieve the target SNRs
        target_snrs.to(snrs.device)
        weights = target_snrs / snrs
        return responses * weights.view(-1, 1, 1)


class SnrSampler:
    """
    Sample values from a configurable distribution, linearly interpolating
    its parameters from `start_params` to `end_params` over `decay_steps`.
    Parameters in `constant_params` remain fixed throughout training.

    The default astrophysical prior of the SNR distribution is a power law of
    index -3 (see `aframe.priors.end_o3_ratesandpops`). To use this with
    curriculum learning on SNR, set `distribution=ml4gw.distributions.PowerLaw`
    and fix `"index"` and `"maximum"` in `constant_params`, then sweep the
    lower bound by giving `"minimum"` in both `start_params` (its high initial
    value) and `end_params` (its low final value).

    Args:
        constant_params:
            Distribution parameters that remain fixed throughout training.
        start_params:
            Distribution parameters at step 0.
        end_params:
            Distribution parameters after `decay_steps` steps.
        decay_steps:
            Number of steps over which to interpolate from `start_params`
            to `end_params`.
        distribution:
            Distribution class to use for sampling. Defaults to
            `ml4gw.distributions.PowerLaw`.
    """

    def __init__(
        self,
        constant_params: dict,
        start_params: dict,
        end_params: dict,
        decay_steps: int,
        distribution: Type = PowerLaw,
    ):
        self.decay_steps = decay_steps
        self._step = 0
        self.constant_params = constant_params
        self.start_params = start_params
        self.end_params = end_params
        self.dist_cls = distribution

        self.dist = distribution(**constant_params, **start_params)

    def __call__(self, N):
        return self.sample((N,))

    def sample(self, N):
        snrs = self.dist.sample(N)
        self.step()
        return snrs

    def step(self):
        self._step += 1
        if self._step > self.decay_steps:
            return

        frac = self._step / self.decay_steps
        params = {
            k: self.start_params[k]
            + frac * (self.end_params[k] - self.start_params[k])
            for k in self.start_params
        }
        self.dist = self.dist_cls(**self.constant_params, **params)


class WaveformProjector(torch.nn.Module):
    def __init__(
        self,
        ifos: list[str],
        sample_rate: float,
        highpass: Optional[float] = None,
        lowpass: Optional[float] = None,
    ) -> None:
        super().__init__()
        tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("tensors", tensors)
        self.register_buffer("vertices", vertices)

        self.sample_rate = sample_rate
        self.rescaler = SnrRescaler(sample_rate, highpass, lowpass)

    def forward(
        self,
        dec: torch.Tensor,
        psi: torch.Tensor,
        phi: torch.Tensor,
        snrs: Union[torch.Tensor, float, None] = None,
        psds: Optional[torch.Tensor] = None,
        **polarizations: torch.Tensor,
    ) -> torch.Tensor:
        responses = gw.compute_observed_strain(
            dec,
            psi,
            phi,
            detector_tensors=self.tensors,
            detector_vertices=self.vertices,
            sample_rate=self.sample_rate,
            **polarizations,
        )
        if snrs is not None:
            if psds is None:
                raise ValueError(
                    "Must specify background PSDs if projecting to target SNR"
                )
            responses = self.rescaler(responses, psds, snrs)
        return responses
