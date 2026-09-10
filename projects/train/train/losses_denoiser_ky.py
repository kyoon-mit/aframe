"""Denoiser reconstruction losses.

Split out of ``train.losses``, which now re-exports them so existing
configs keep working.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_FUNCTIONS = (
    "mse",
    "mae",
    "rmse",
    "huber",
    "smooth_l1",
    "smae",
    "logcosh",
)


def loss_helper(
    loss: str,
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """Scalar discrepancy between two tensors, selected by name.

    All options are means over every element, so they stay comparable in
    magnitude and can be swapped without retuning the mixing weight. See
    LOSS_FUNCTIONS for the choices; ``delta`` is the huber/smooth_l1
    transition point (ignored otherwise).
    """
    if loss == "mse":
        return F.mse_loss(pred, target)
    if loss == "mae":
        return F.l1_loss(pred, target)
    if loss == "rmse":
        # clamp keeps the sqrt gradient finite when the error reaches zero
        return F.mse_loss(pred, target).clamp_min(1e-12).sqrt()
    if loss == "huber":
        return F.huber_loss(pred, target, delta=delta)
    if loss in ("smooth_l1", "smae"):
        return F.smooth_l1_loss(pred, target, beta=delta)
    if loss == "logcosh":
        # log(cosh(d)) as |d| + log1p(exp(-2|d|)) - log(2); no overflow
        d = (pred - target).abs()
        return (d + torch.log1p(torch.exp(-2.0 * d)) - math.log(2.0)).mean()
    raise ValueError(f"Unknown loss {loss!r}. Choose one of {LOSS_FUNCTIONS}.")


class TermStashMixin:
    """Stash each loss term as a ``last_<name>_term`` attribute.

    The terms exist to be logged, so they are detached by default: holding
    their graphs would keep a copy of the backward graph alive every step.
    ``term_gradient_norms`` sets ``keep_term_graph`` for the length of one
    call when it needs to differentiate them.
    """

    keep_term_graph = False

    def _stash(self, **terms: torch.Tensor) -> None:
        for name, value in terms.items():
            setattr(
                self,
                f"last_{name}_term",
                value if self.keep_term_graph else value.detach(),
            )


@torch.enable_grad()
def term_gradient_norms(
    loss_fn: nn.Module,
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict:
    """Gradient norm of each loss term with respect to the prediction.

    The mixing weight is only meaningful next to these: two terms can have
    similar values and still contribute gradients that differ by orders of
    magnitude, and it is the gradients that move the weights.

    Differentiates each ``last_<name>_term`` the loss stashed and returns
    ``{<name>_value, <name>_gradnorm}``, or an empty dict for a loss that
    stashes nothing.
    """
    prediction = pred.detach().clone().requires_grad_(True)
    target = target.detach()

    previous = getattr(loss_fn, "keep_term_graph", None)
    if previous is None:
        return {}

    loss_fn.keep_term_graph = True
    try:
        loss_fn(prediction, target)
        terms = {
            name[len("last_") : -len("_term")]: getattr(loss_fn, name)
            for name in dir(loss_fn)
            if name.startswith("last_") and name.endswith("_term")
        }
        differentiable = [
            (name, value)
            for name, value in terms.items()
            if torch.is_tensor(value) and value.requires_grad
        ]

        stats = {}
        for index, (name, value) in enumerate(differentiable):
            stats[f"{name}_value"] = float(value.detach())
            # one graph is shared across the terms, so hold it until the last
            (gradient,) = torch.autograd.grad(
                value,
                prediction,
                retain_graph=index < len(differentiable) - 1,
                allow_unused=True,
            )
            stats[f"{name}_gradnorm"] = (
                0.0 if gradient is None else float(gradient.norm())
            )

        # a term with no gradient path still has a value worth logging
        for name, value in terms.items():
            stats.setdefault(f"{name}_value", float(value.detach()))
    finally:
        loss_fn.keep_term_graph = previous
    return stats


class ScheduledMixtureLoss(TermStashMixin, nn.Module):
    """Mixture of a time-domain and a frequency-domain denoiser loss.

    Aframe tensors are ``(B, C, L)``, so the FFT runs over the last dim ``L``.
    ``alpha`` is externally mutable so the training task can schedule it
    epoch-by-epoch (0 = pure spectral, 1 = pure time-domain).

    The two terms live on different scales and that mismatch moves during
    training. A time-domain MSE grows with the square of the waveform
    amplitude, while an ``msle`` spectral term compares log magnitudes and
    so is amplitude-invariant: measured over four decades of amplitude the
    ratio between them swings by six orders of magnitude. Under an SNR
    curriculum the amplitudes shift as training runs, so a fixed ``alpha``
    silently re-weights the two objectives.

    Dividing each term by a statistic of the current batch fixes the scale
    but breaks on noise-only targets, where that statistic is zero: the
    quotient is then either a clamp-limited spike or a division by zero.
    Instead the time term is divided by ``target_scale``, a running mean of
    the target power kept as a buffer and updated only from batches that
    carry signal. It is a slow-moving constant rather than a per-batch
    quantity, so it tracks the curriculum without ever collapsing to zero,
    and a batch of pure background is scored against the same scale as any
    other batch -- its error stays finite and meaningful.

    Args:
        alpha: initial weight for the time term. Updated in place each epoch.
        density: if True, divide the time term by the running target scale
            so it becomes dimensionless and comparable to the spectral term.
            If False, no scaling; rely on alpha for balance.
        scale_momentum: EMA momentum for the running target scale. Higher
            values track the curriculum more slowly.
        time_loss: time-domain discrepancy, one of LOSS_FUNCTIONS.
        spectral_loss: 'mse' (default) compares |FFT| directly; 'msle'
            compares log(|FFT| + log_floor) to compress dynamic range so loud
            bins stop dominating; any LOSS_FUNCTIONS name applies that
            discrepancy to plain magnitudes.
        log_floor: floor inside the log for 'msle'.
        sig_thresh: a row whose clean target has a smaller norm than this
            carries no signal, and its spectral error is measured on plain
            magnitudes rather than through the log.
        log_base: base of the logarithm for 'msle'. Changing it rescales the
            spectral term by 1/ln(base)**2 -- base 10 makes it ~5.3x smaller
            than natural log -- so it shifts where alpha balances the two
            terms rather than changing the shape of either.
        huber_delta: transition point for 'huber'/'smooth_l1', both terms.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        density: bool = True,
        time_loss: str = "mse",
        spectral_loss: str = "mse",
        log_floor: float = 1e-9,
        log_base: float = 10.0,
        huber_delta: float = 1.0,
        scale_momentum: float = 0.99,
        sig_thresh: float = 0.5,
    ):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if time_loss not in LOSS_FUNCTIONS:
            raise ValueError(
                f"time_loss must be one of {LOSS_FUNCTIONS}, got {time_loss!r}"
            )
        if spectral_loss not in LOSS_FUNCTIONS + ("msle",):
            raise ValueError(
                f"spectral_loss must be 'msle' or one of {LOSS_FUNCTIONS}, "
                f"got {spectral_loss!r}"
            )
        if log_floor <= 0.0:
            raise ValueError(f"log_floor must be > 0, got {log_floor}")
        if log_base <= 1.0:
            raise ValueError(f"log_base must be > 1, got {log_base}")
        if huber_delta <= 0.0:
            raise ValueError(f"huber_delta must be > 0, got {huber_delta}")
        if not 0.0 <= scale_momentum < 1.0:
            raise ValueError(
                f"scale_momentum must be in [0, 1), got {scale_momentum}"
            )
        self.alpha = alpha  # mutable; updated each epoch by the task
        self.density = density
        self.time_loss = time_loss
        self.spectral_loss = spectral_loss
        self.log_floor = log_floor
        self.log_base = log_base
        self._log_base_scale = math.log(log_base)
        self.huber_delta = huber_delta
        self.scale_momentum = scale_momentum
        self.sig_thresh = sig_thresh
        # running mean of the target's own "error against predicting zero",
        # saved with the model so a resumed run keeps the same normalization
        self.register_buffer("target_scale", torch.ones(()))
        self.register_buffer(
            "scale_initialized", torch.zeros((), dtype=torch.bool)
        )

    @torch.no_grad()
    def _update_scale(self, target: torch.Tensor) -> None:
        """Track the typical target scale across batches.

        The update is skipped for a batch whose targets are all (or nearly
        all) zero, since such a batch says nothing about the scale of a
        signal and would drag the running value toward zero.
        """
        batch_scale = loss_helper(
            self.time_loss,
            torch.zeros_like(target),
            target,
            self.huber_delta,
        )
        if batch_scale <= 0.0:
            return
        if not self.scale_initialized:
            self.target_scale.fill_(float(batch_scale))
            self.scale_initialized.fill_(True)
        else:
            momentum = self.scale_momentum
            self.target_scale.mul_(momentum).add_(
                batch_scale * (1.0 - momentum)
            )

    def _term(
        self, loss: str, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """One term of the mixture, reduced over the whole batch.

        ``loss`` selects the discrepancy, so ``time_loss`` and
        ``spectral_loss`` are honored rather than assumed to be MSE.
        """
        return loss_helper(loss, pred, target, self.huber_delta)

    def _msle_term(
        self,
        pred_mag: torch.Tensor,
        target_mag: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Spectral error, scored differently where the target is empty.

        The log exists to compress a spectrum that spans decades, so that
        quiet bins are not drowned out by loud ones. A noise-only row has
        no such spectrum: its target is zero everywhere, and the log's
        floor then decides what counts as silent. At ``log_floor`` 1e0 the
        difference between emitting 1e-2 and emitting nothing is 4e-3 in
        log, squared away to 2e-5, so the row is scored as already
        perfect. Lowering the floor restores the distinction but puts the
        prediction in the region where the log's 1/x derivative drives the
        gradient up as the output falls, which is the wrong direction to
        converge from.

        So the log is used only where there is a distribution to compare,
        and rows whose target is empty are scored on plain magnitudes,
        where zero is an ordinary value and the gradient falls off as the
        output approaches it.
        """
        row_energy = target.reshape(target.shape[0], -1).pow(2).sum(-1)
        has_signal = row_energy.sqrt() > self.sig_thresh
        if bool(has_signal.all()):
            return self._term("mse", *self._as_log(pred_mag, target_mag))

        terms, weights = [], []
        if bool(has_signal.any()):
            log_pred, log_target = self._as_log(
                pred_mag[has_signal], target_mag[has_signal]
            )
            terms.append(self._term("mse", log_pred, log_target))
            weights.append(int(has_signal.sum()))

        empty = ~has_signal
        terms.append(self._term("mse", pred_mag[empty], target_mag[empty]))
        weights.append(int(empty.sum()))

        total = sum(weights)
        return sum(
            term * (weight / total)
            for term, weight in zip(terms, weights, strict=True)
        )

    def _as_log(
        self, pred_mag: torch.Tensor, target_mag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Magnitudes as log_b(|X| + floor)."""
        return (
            torch.log(pred_mag + self.log_floor) / self._log_base_scale,
            torch.log(target_mag + self.log_floor) / self._log_base_scale,
        )

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """pred, target: (B, C, L). Returns scalar loss."""
        if self.training and self.density:
            self._update_scale(target)

        time_term = self._term(self.time_loss, pred, target)
        if self.density:
            # divide by a slow-moving constant, not a statistic of this
            # batch, so a background-only batch is scored on the same
            # footing instead of against a vanishing denominator
            time_term = time_term / self.target_scale.clamp_min(1e-12)

        pred_mag = torch.fft.rfft(pred, dim=-1).abs()
        target_mag = torch.fft.rfft(target, dim=-1).abs()

        if self.spectral_loss == "msle":
            spectral_term = self._msle_term(pred_mag, target_mag, target)
        else:
            spectral_term = self._term(
                self.spectral_loss, pred_mag, target_mag
            )

        mix = self.alpha * time_term + (1.0 - self.alpha) * spectral_term

        # expose raw components (pre-alpha) so the task can log them and
        # pick alpha from their relative scale
        self._stash(time=time_term, spectral=spectral_term)
        return mix


class CorrelationDenoiseLoss(nn.Module):
    """Scale-invariant matched-filter denoiser loss.

    Plain MSE between the denoised and clean whitened strain collapses to
    "predict zero" for low-SNR (SNR 4-8) signals: their power is tiny next to
    the noise, so silence already yields a low MSE and the denoiser never
    learns to recover the buried chirp. Matched filtering, the optimal
    low-SNR detector, cares only about the template SHAPE, not its amplitude.

    This loss rewards template overlap: ``1 - NCC`` where NCC is the
    normalized cross-correlation (cosine similarity over time) between
    denoised and clean, per (sample, detector). Being scale-free it does not
    reward zero output. It is applied only to rows that actually carry signal
    (``||clean|| ~ optimal SNR > sig_thresh``); a small MSE term over all rows
    (``mse_weight``) both suppresses noise-only rows to zero and keeps the
    recovered amplitude calibrated so the regressor can read absolute scale.
    """

    def __init__(
        self,
        mse_weight: float = 0.1,
        sig_thresh: float = 0.5,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.sig_thresh = sig_thresh
        self.eps = eps

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        i = inputs - inputs.mean(dim=-1, keepdim=True)
        t = targets - targets.mean(dim=-1, keepdim=True)
        t_norm = t.norm(dim=-1)  # (B, C) ~ optimal matched-filter SNR
        i_norm = i.norm(dim=-1)
        ncc = (i * t).sum(dim=-1) / (i_norm * t_norm + self.eps)  # (B, C)
        sig = t_norm > self.sig_thresh
        if sig.any():
            corr = (1.0 - ncc[sig]).mean()
        else:
            corr = inputs.new_zeros(())
        if self.mse_weight:
            return corr + self.mse_weight * F.mse_loss(inputs, targets)
        return corr


class SNRWeightedMSELoss(nn.Module):
    """Per-row SNR-focal MSE.

    Plain MSE is dominated by the loud (high-SNR) rows, so the denoiser nails
    the easy signals and neglects the SNR 4-8 rows that are the actual
    unsolved regime. Here each row's reconstruction error is weighted by
    ``(ref/(snr+ref))**gamma``, shifting capacity onto low-SNR signals while
    keeping high-SNR rows nonzero. ``snr`` is read from the clean-target norm
    (``||whitened h|| ~ optimal SNR``); noise-only rows (norm ~ 0) get the max
    weight so noise is still driven to zero. Weights are renormalized to mean
    1 so the ``lambda_denoise`` scale stays comparable to a plain MSE.
    """

    def __init__(
        self, gamma: float = 1.0, ref_snr: float = 8.0, eps: float = 1e-8
    ):
        super().__init__()
        self.gamma = gamma
        self.ref_snr = ref_snr
        self.eps = eps

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        row_mse = ((inputs - targets) ** 2).mean(dim=-1)  # (B, C)
        snr = targets.norm(dim=-1)  # (B, C)
        w = (self.ref_snr / (snr + self.ref_snr)) ** self.gamma
        w = w / w.mean().clamp(min=self.eps)
        return (w * row_mse).mean()


class MixtureMSESpectralLoss(nn.Module):
    """MSE (time domain) + spectral MSE (frequency domain) mixture.

    Spectral term compares the magnitude of the real FFT along the sequence
    axis. Aframe tensors are ``(B, C, L)``, so the FFT is taken over the last
    dim ``L``.

    Args:
        alpha: Weight for the time-domain MSE. Spectral term weighted by
            ``(1 - alpha)``. Defaults to ``0.5``.
    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        mse_loss = self.mse(inputs, targets)
        inputs_mag = torch.abs(torch.fft.rfft(inputs, dim=-1))
        targets_mag = torch.abs(torch.fft.rfft(targets, dim=-1))
        spectral_loss = self.mse(inputs_mag, targets_mag)
        return self.alpha * mse_loss + (1.0 - self.alpha) * spectral_loss
