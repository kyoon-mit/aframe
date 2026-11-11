"""Den-reg batches whose input is raw strain with the learned noise removed."""

import torch
import yaml
from jsonargparse import ArgumentParser
from torch import nn

from train.data.supervised import DenoiserOnlyAframeDataset
from train.data.supervised.supervised import SupervisedAframeDataset
from train.experimental.raw_noise.predict import prediction_errors


def load_noise_model(config: str, ckpt: str) -> nn.Module:
    """The network of a ``RawNextSampleTask`` run, in eval mode."""
    with open(config) as f:
        arch = yaml.safe_load(f)["model"]["init_args"]["arch"]
    parser = ArgumentParser()
    parser.add_subclass_arguments(nn.Module, "arch")
    net = parser.instantiate_classes(parser.parse_object({"arch": arch})).arch
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    net.load_state_dict(
        {
            k[len("model.") :]: v
            for k, v in state["state_dict"].items()
            if k.startswith("model.")
        }
    )
    return net.eval().requires_grad_(False)


class NoiseRemovedAframeDataset(DenoiserOnlyAframeDataset):
    """Emits ``(X, X_clean, y, params)`` like the parent, with the whitener
    replaced by the learned noise removal.

    The whole raw window (spectrum stretch included) is sign flipped and
    time reversed, then the signal is injected as in the parent. The window
    becomes linear prediction errors (``predict.prediction_errors``, with
    the spectrum of its first ``psd_length`` seconds); the noise model
    reads the last ``context`` seconds of them. ``X`` is the errors minus
    the model's mean, over its spread (root of the mixture variance), on
    the same ``kernel_length`` seconds the whitener keeps. ``X_clean`` is
    the errors of the injected signal alone over the same spread.
    ``last_whitened`` holds the parent's whitened ``X`` of the same batch.

    Args:
        noise_config: config.yaml of the noise model run.
        noise_ckpt: its checkpoint.
        context: seconds of prediction errors the noise model reads.
        noise_chunk: windows per noise model forward pass.
    """

    def __init__(
        self,
        *args,
        noise_config: str,
        noise_ckpt: str,
        context: float = 8.0,
        noise_chunk: int = 64,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(
            {
                "noise_config": noise_config,
                "noise_ckpt": noise_ckpt,
                "context": context,
                "noise_chunk": noise_chunk,
            }
        )
        self.noise_model = None
        self.last_whitened = None

    def build_transforms(self):
        super().build_transforms()
        self.noise_model = load_noise_model(
            self.hparams.noise_config, self.hparams.noise_ckpt
        )

    @torch.no_grad()
    def remove_noise(self, errors):
        """Mean and spread (B, C, N) of the noise model on (B, C, N)."""
        flat = errors.reshape(-1, 1, errors.size(-1))
        means, spreads = [], []
        for chunk in flat.split(self.hparams.noise_chunk):
            mean, log_sigma, logits = self.noise_model(chunk)
            weights = torch.softmax(logits, dim=1)
            variance = (weights * torch.exp(2 * log_sigma)).sum(1)
            means.append(mean)
            spreads.append(variance.sqrt())
        return (
            torch.cat(means).reshape(errors.shape),
            torch.cat(spreads).reshape(errors.shape),
        )

    @torch.no_grad()
    def inject(self, X, waveforms, params):
        full = self.reverser(self.inverter(X))
        inverter, reverser = self.inverter, self.reverser
        self.inverter = self.reverser = nn.Identity()
        try:
            X, y, psds, params_out = SupervisedAframeDataset.inject(
                self, X=full, waveforms=waveforms, params=params
            )
        finally:
            self.inverter, self.reverser = inverter, reverser
        clean = self._clean_signal
        self.last_whitened = self.apply_transforms(X, psds)

        tail = X.size(-1)
        full = torch.cat([full[..., :-tail], X], dim=-1)
        clean_full = torch.zeros_like(full)
        clean_full[..., -tail:] = clean

        rate = self.hparams.sample_rate
        second = int(rate)
        errors, _ = prediction_errors(full, psds, rate, second)
        clean_errors, _ = prediction_errors(clean_full, psds, rate, second)
        context = int(self.hparams.context * rate)
        errors = errors[..., -context:]
        clean_errors = clean_errors[..., -context:]
        mean, spread = self.remove_noise(errors)

        crop = int(self.hparams.fduration * rate) // 2
        keep = slice(-tail + crop, -crop)
        X = ((errors - mean) / spread)[..., keep]
        X_clean = (clean_errors / spread)[..., keep]
        return X, X_clean, y, params_out
