"""Losses for the detection classifier.

Denoiser reconstruction losses live in ``train.losses_denoiser_ky``.
"""

import torch


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
