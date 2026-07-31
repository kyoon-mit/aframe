import torch
import torch.nn as nn


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
