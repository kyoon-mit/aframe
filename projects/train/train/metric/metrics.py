import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .types import BatchedParams, BatchedTarget, acc_metric, metric


def _scores(pred: BatchedTarget) -> np.ndarray:
    """Return a flat 1-D score array regardless of whether pred is 1-D (raw
    scalar per sample) or 2-D (log-softmax with shape (N, 2)).  For the
    latter the positive-class column is extracted."""
    pred = np.asarray(pred)
    if pred.ndim == 2:
        return pred[:, -1]
    return pred.flatten()


def _binary_at_fpr(
    target: BatchedTarget, pred: BatchedTarget, target_fpr: float = 1e-3
) -> np.ndarray:
    """Binarise scores by finding the decision threshold at *target_fpr* on the
    ROC curve.  Returns an integer array of predicted labels."""
    scores = _scores(pred)
    fpr, _, thresholds = roc_curve(target.flatten(), scores)
    idx = np.argmin(np.abs(fpr - target_fpr))
    return (scores >= thresholds[idx]).astype(int)


# ------------------------------------------------------------
#   ROC / AUC metrics  (work on accumulated epoch data)
# ------------------------------------------------------------


@acc_metric(stages=("val", "test"), prog_bar=True)
def auc_roc(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    auc = roc_auc_score(target.flatten(), _scores(pred))
    return float(auc) if not np.isnan(auc) else 0.0


@acc_metric(stages=("val", "test"), prog_bar=True)
def auc_roc_at_fpr_0_001(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    auc = roc_auc_score(target.flatten(), _scores(pred), max_fpr=1e-3)
    return float(auc) if not np.isnan(auc) else 0.0


@acc_metric(stages=("val", "test"))
def tpr_at_fpr_0_001(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    fpr, tpr, _ = roc_curve(target.flatten(), _scores(pred))
    idx = np.argmin(np.abs(fpr - 1e-3))
    return float(tpr[idx])


@acc_metric(stages=("val", "test"))
def fpr_at_0_5_tpr(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    fpr, tpr, _ = roc_curve(target.flatten(), _scores(pred))
    idx = np.argmin(np.abs(tpr - 0.5))
    return float(fpr[idx])


# ------------------------------------------------------------
#   Score distribution metrics  (per-batch, train + val)
# ------------------------------------------------------------


@metric()
def mean_fg_score(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """Mean model score for foreground (injected signal) samples."""
    scores = _scores(pred)
    mask = target.flatten() == 1
    return float(np.mean(scores[mask])) if np.any(mask) else float("nan")


@metric()
def mean_bg_score(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """Mean model score for background samples."""
    scores = _scores(pred)
    mask = target.flatten() == 0
    return float(np.mean(scores[mask])) if np.any(mask) else float("nan")


@metric()
def score_separation(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """Difference between mean fg and mean bg scores — a quick proxy for
    how well the model separates the two classes."""
    return mean_fg_score(target, pred) - mean_bg_score(target, pred)


# ------------------------------------------------------------
#   Threshold-based classification metrics
#   (accumulated — threshold is derived from the full epoch)
# ------------------------------------------------------------


@acc_metric(stages=("val", "test"))
def accuracy(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """Accuracy at the FPR=1e-3 decision threshold."""
    pred_binary = _binary_at_fpr(target, pred)
    return float(np.mean(target.flatten() == pred_binary))


@acc_metric(stages=("val", "test"))
def accuracy_pos_class(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """Recall (TPR) at the FPR=1e-3 decision threshold."""
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    mask = t == 1
    return (
        float(np.mean(t[mask] == pred_binary[mask]))
        if np.any(mask)
        else float("nan")
    )


@acc_metric(stages=("val", "test"))
def accuracy_neg_class(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    """True-negative rate at the FPR=1e-3 decision threshold."""
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    mask = t == 0
    return (
        float(np.mean(t[mask] == pred_binary[mask]))
        if np.any(mask)
        else float("nan")
    )


@acc_metric(stages=("val", "test"))
def false_positive_rate(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    return float(np.mean((t == 0) & (pred_binary == 1)))


@acc_metric(stages=("val", "test"))
def false_negative_rate(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    return float(np.mean((t == 1) & (pred_binary == 0)))


@acc_metric(stages=("val", "test"))
def true_positive_rate(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    return float(np.mean((t == 1) & (pred_binary == 1)))


@acc_metric(stages=("val", "test"))
def true_negative_rate(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    pred_binary = _binary_at_fpr(target, pred)
    t = target.flatten()
    return float(np.mean((t == 0) & (pred_binary == 0)))


@acc_metric(stages=("val", "test"))
def f1_score(
    target: BatchedTarget, pred: BatchedTarget, **kwargs
) -> np.ndarray:
    tp = true_positive_rate(target, pred)
    fp = false_positive_rate(target, pred)
    fn = false_negative_rate(target, pred)
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


# ------------------------------------------------------------
#   Misc
# ------------------------------------------------------------


@metric()
def snr_magnitude(params: BatchedParams, **kwargs) -> np.ndarray:
    snr = params["snr"]
    return float(np.mean(snr[~np.isnan(snr)]))
