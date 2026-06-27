"""The two scorers and a tiny inference helper.

* ``TinyCNN`` -- a fully-convolutional 1-D CNN over a window; ~a couple
  thousand parameters.  Because it is convolutional it can be slid cheaply
  over a whole
  segment at evaluation time.
* the approach-2 model is just an sklearn classifier on
  :func:`scorer.features.window_features`, handled in ``train``/``evaluate``.
"""

import numpy as np
import torch
import torch.nn as nn


class TinyCNN(nn.Module):
    def __init__(self, channels=16, kernel=7):
        super().__init__()
        pad = kernel // 2
        self.body = nn.Sequential(
            nn.Conv1d(1, channels, kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=pad),
            nn.ReLU(),
        )
        # global max + mean pooling -> small head
        self.head = nn.Linear(2 * channels, 1)

    def forward(self, x):
        # x: (B, L) -> (B, 1, L)
        h = self.body(x.unsqueeze(1))
        pooled = torch.cat([h.max(dim=2).values, h.mean(dim=2)], dim=1)
        return self.head(pooled).squeeze(1)


@torch.no_grad()
def score_windows(model, X, device, batch=4096, sigmoid=True):
    """Scores for a stack of ``(N, L)`` windows.  ``sigmoid=True`` for AUC /
    monitoring; ``sigmoid=False`` returns the raw model output, which is what
    we rank/cluster on (monotonic and tie-free at the top, unlike a saturating
    probability)."""
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(
            X[i : i + batch], dtype=torch.float32, device=device
        )
        s = model(xb)
        if sigmoid:
            s = torch.sigmoid(s)
        out.append(s.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def sliding_windows(y, L, stride):
    """``(n_windows, L)`` views of ``y`` and the centre index per window."""
    if len(y) < L:
        y = np.pad(y, (0, L - len(y)))
    views = np.lib.stride_tricks.sliding_window_view(y, L)[::stride]
    centers = np.arange(0, len(y) - L + 1, stride) + L // 2
    return views, centers


def dense_cnn_score(model, y, L, stride, device):
    """Score a full (already-standardized) segment array densely by sliding the
    CNN, then interpolate back onto the per-sample grid so it can be clustered
    exactly like an integrated statistic."""
    views, centers = sliding_windows(y, L, stride)
    s = score_windows(model, views.astype(np.float32), device, sigmoid=False)
    grid = np.arange(len(y))
    if len(centers) < 2:
        return np.full(len(y), s[0] if len(s) else 0.0)
    return np.interp(grid, centers, s, left=s[0], right=s[-1])
