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


class ScheduledMixtureLoss(nn.Module):
    """Mixture of a time-domain and a frequency-domain denoiser loss.

    Aframe tensors are ``(B, C, L)``, so the FFT runs over the last dim ``L``.
    ``alpha`` is externally mutable so the training task can schedule it
    epoch-by-epoch (0 = pure spectral, 1 = pure time-domain).

    Both terms are reduced over the whole batch. When ``density`` is set,
    each term is divided by the same discrepancy measured against a zero
    prediction -- mean(err) / mean(scale) -- so loud events dominate both
    numerator and denominator, as they did before the per-row variant.

    Args:
        alpha: initial weight for the time term. Updated in place each epoch.
        density: if True, divide each term by the batch-wide discrepancy
            against a zero prediction, making both terms dimensionless
            "error relative to predicting nothing" so alpha=0.5 is equal
            weight. If False, no scaling; rely on alpha for balance.
        time_loss: time-domain discrepancy, one of LOSS_FUNCTIONS.
        spectral_loss: 'mse' (default) compares |FFT| directly; 'msle'
            compares log(|FFT| + log_floor) to compress dynamic range so loud
            bins stop dominating; any LOSS_FUNCTIONS name applies that
            discrepancy to plain magnitudes.
        log_floor: floor inside the log for 'msle'.
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
        self.alpha = alpha  # mutable; updated each epoch by the task
        self.density = density
        self.time_loss = time_loss
        self.spectral_loss = spectral_loss
        self.log_floor = log_floor
        self.log_base = log_base
        self._log_base_scale = math.log(log_base)
        self.huber_delta = huber_delta

    def _term(
        self, loss: str, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """One term of the mixture, reduced over the whole batch and
        optionally made dimensionless.

        ``loss`` selects the discrepancy, so ``time_loss`` and
        ``spectral_loss`` are honored rather than assumed to be MSE.
        """
        value = loss_helper(loss, pred, target, self.huber_delta)
        if self.density:
            scale = loss_helper(
                loss, torch.zeros_like(target), target, self.huber_delta
            )
            value = value / scale.clamp_min(1e-8)
        return value

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """pred, target: (B, C, L). Returns scalar loss."""
        time_term = self._term(self.time_loss, pred, target)

        pred_mag = torch.fft.rfft(pred, dim=-1).abs()
        target_mag = torch.fft.rfft(target, dim=-1).abs()

        if self.spectral_loss == "msle":
            # log_b(x) = ln(x) / ln(b)
            pred_mag = (
                torch.log(pred_mag + self.log_floor) / self._log_base_scale
            )
            target_mag = (
                torch.log(target_mag + self.log_floor) / self._log_base_scale
            )
            spectral_term = self._term("mse", pred_mag, target_mag)
        else:
            spectral_term = self._term(
                self.spectral_loss, pred_mag, target_mag
            )

        mix = self.alpha * time_term + (1.0 - self.alpha) * spectral_term

        # expose raw components (pre-alpha) so the task can log them and
        # pick alpha from their relative scale
        self.last_time_term = time_term.detach()
        self.last_spectral_term = spectral_term.detach()
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


def soft_pauc_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    fpr_frac: float = 0.05,
    beta: float = 10.0,
) -> torch.Tensor:
    """Differentiable partial-AUROC surrogate at low false-positive rate.

    Detection at low FAR is set by the loudest background. This penalizes
    signals that fail to outscore the top ``fpr_frac`` fraction of background
    logits in the batch (a soft ranking hinge). Returns 0 if the batch lacks
    positives or negatives.

    Args:
        logits: raw detection logits, shape ``(N, 1)`` or ``(N,)``.
        y: binary labels, same shape.
        fpr_frac: fraction of loudest negatives to rank against (low-FAR
            focus; smaller = stricter tail).
        beta: softplus sharpness; higher approaches a hard hinge.
    """
    logits = logits.reshape(-1)
    y = y.reshape(-1)
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        return logits.new_zeros(())
    k = max(1, int(neg.numel() * fpr_frac))
    top_neg = torch.topk(neg, k).values
    # pairwise: want pos > top_neg; penalize (top_neg - pos) > 0
    diff = top_neg.view(-1, 1) - pos.view(1, -1)  # (k, P)
    return torch.nn.functional.softplus(beta * diff).mean() / beta
