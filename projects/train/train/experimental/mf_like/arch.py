"""Shallow networks from the literature, one class per paper."""

import torch
from torch import nn


class MNetShallow(nn.Module):
    """MNet-Shallow (Yan et al. 2022, PRD 105, 043006; arXiv:2104.03961):
    max_i (<w_i, x> + b_i) over ``num_templates`` rows of W (Eq. 29).

    Args:
        d_input: samples per input segment (n).
        num_templates: rows of W (k).
        bias: whether b is a trained parameter (zero at start, Eq. 28).
    """

    def __init__(
        self, d_input: int, num_templates: int, bias: bool = False
    ) -> None:
        super().__init__()
        self._d_input = d_input
        self._num_templates = num_templates
        self._bias = bias
        self._pool = nn.MaxPool1d(kernel_size=self._num_templates)
        self._hidden_layer = nn.Linear(
            in_features=self._d_input,
            out_features=self._num_templates,
            bias=self._bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n) segments -> (B,) max over templates of W x + b."""
        output_layer = self._pool(self._hidden_layer(x))
        return output_layer.squeeze(-1)
