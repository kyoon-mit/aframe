import torch
import torch.nn as nn
import torch.nn.functional as F


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
