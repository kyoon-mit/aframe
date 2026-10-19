import torch
import torch.nn as nn
from ml4gw.nn.ssm.s4d import S4D


class BothWaysS4D(nn.Module):
    """A residual S4D block that reads the sequence forwards and backwards.

    ml4gw's S4D only looks back in time; a second S4D on the reversed sequence
    lets every output sample see the whole window. Optional ``scale`` and
    ``shift`` of shape (batch, width, 1) adjust the normalized input; the
    generator uses them to tell each block the strength of the added static.
    """

    def __init__(self, width: int, d_state: int = 64, dt_min: float = 0.001, dt_max: float = 0.1):
        super().__init__()
        self.norm = nn.GroupNorm(1, width)
        self.forwards = S4D(width, d_state, dt_min=dt_min, dt_max=dt_max)
        self.backwards = S4D(width, d_state, dt_min=dt_min, dt_max=dt_max)

    def forward(self, x: torch.Tensor, scale: torch.Tensor | None = None, shift: torch.Tensor | None = None):
        h = self.norm(x)
        if scale is not None:
            h = h * (1 + scale) + shift
        return x + self.forwards(h) + self.backwards(h.flip(-1)).flip(-1)
