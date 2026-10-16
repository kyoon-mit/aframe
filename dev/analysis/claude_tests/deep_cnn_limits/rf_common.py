"""Receptive-field helpers for ResNetGridNet (ml4gw ResNet1D)."""

import contextlib
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/projects/train")

from ml4gw.nn.norm import GroupNorm1D  # noqa: E402
from train.experimental.long_conv.arch import ResNetGridNet  # noqa: E402

FS = 2048
N = 8192
RESNET_DIR = (
    "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/"
    "CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/checkpoints/"
)


def build(seed: int, dtype=torch.float64, state=None) -> ResNetGridNet:
    torch.manual_seed(seed)
    m = ResNetGridNet(num_ifos=2, layers=(3, 4, 6, 3), kernel_size=3, norm_groups=16)
    if state is not None:
        m.load_state_dict(state)
    return m.to(dtype).eval()


def load_state(path: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in ck["state_dict"].items() if k.startswith("model.")}
    return sd, ck.get("epoch")


def main_path(net) -> list:
    """(name, kernel, stride, padding, dilation) of every layer on the main path."""
    r = net.net
    out = [
        ("stem conv", r.conv1.kernel_size[0], r.conv1.stride[0], r.conv1.padding[0], r.conv1.dilation[0]),
        ("stem maxpool", r.maxpool.kernel_size, r.maxpool.stride, r.maxpool.padding, r.maxpool.dilation),
    ]
    for si, stage in enumerate(r.residual_layers):
        for bi, blk in enumerate(stage):
            for name in ("conv1", "conv2"):
                c = getattr(blk, name)
                out.append((f"stage{si + 1}.block{bi + 1}.{name}", c.kernel_size[0], c.stride[0], c.padding[0], c.dilation[0]))
    return out


def back_interval(layers, lo: int, hi: int):
    """Input interval seen by output indices [lo, hi] of a chain of layers."""
    for _, k, s, p, d in reversed(layers):
        lo, hi = lo * s - p, hi * s - p + (k - 1) * d
    return lo, hi


def block_union_check(net, idx: int):
    """Walk back block by block, taking the union of main and skip paths."""
    r = net.net
    lo, hi = idx, idx
    for stage in reversed(list(r.residual_layers)):
        for blk in reversed(list(stage)):
            c1, c2 = blk.conv1, blk.conv2
            main = back_interval(
                [("c1", c1.kernel_size[0], c1.stride[0], c1.padding[0], c1.dilation[0]),
                 ("c2", c2.kernel_size[0], c2.stride[0], c2.padding[0], c2.dilation[0])], lo, hi)
            if blk.downsample is not None:
                ds = blk.downsample[0]
                skip = back_interval([("ds", ds.kernel_size[0], ds.stride[0], ds.padding[0], ds.dilation[0])], lo, hi)
            else:
                skip = (lo, hi)
            assert main[0] <= skip[0] and skip[1] <= main[1], (main, skip)
            lo, hi = min(main[0], skip[0]), max(main[1], skip[1])
    stem = [("conv", r.conv1.kernel_size[0], r.conv1.stride[0], r.conv1.padding[0], r.conv1.dilation[0]),
            ("pool", r.maxpool.kernel_size, r.maxpool.stride, r.maxpool.padding, r.maxpool.dilation)]
    return back_interval(stem, lo, hi)


def features(net, x: torch.Tensor) -> torch.Tensor:
    """Output of the last residual stage, before the average pool."""
    r = net.net
    x = r.maxpool(r.relu(r.bn1(r.conv1(x))))
    for layer in r.residual_layers:
        x = layer(x)
    return x


def _gn_forward_frozen(self, x):
    """GroupNorm1D.forward with its mean and variance treated as constants."""
    keepdims = self.num_groups == self.num_channels
    xd = x.detach()
    mean = xd.mean(-1, keepdims=keepdims)
    sq_mean = (xd**2).mean(-1, keepdims=keepdims)
    if self.num_groups != self.num_channels:
        mean = torch.stack([mean, sq_mean], dim=1)
        mean = mean.reshape(-1, 2, self.num_groups, self.channels_per_group)
        mean = mean.mean(-1, keepdims=True)
        mean = mean.expand(-1, -1, -1, self.channels_per_group)
        mean = mean.reshape(-1, 2, self.num_channels, 1)
        mean, sq_mean = mean[:, 0], mean[:, 1]
    std = (sq_mean - mean**2 + self.eps) ** 0.5
    scale = self.weight / std
    shift = self.bias - scale * mean
    return shift + x * scale


@contextlib.contextmanager
def frozen_norm_stats():
    orig = GroupNorm1D.forward
    GroupNorm1D.forward = _gn_forward_frozen
    try:
        yield
    finally:
        GroupNorm1D.forward = orig


def whitened_noise(gen: torch.Generator, dtype=torch.float64, highpass: float = 20.0) -> torch.Tensor:
    """(1, 2, N) unit-variance white Gaussian noise with bins below ``highpass`` removed."""
    x = torch.randn(1, 2, N, generator=gen, dtype=torch.float64)
    X = torch.fft.rfft(x, dim=-1)
    f = torch.fft.rfftfreq(N, 1 / FS, dtype=torch.float64)
    X[..., f < highpass] = 0
    return torch.fft.irfft(X, n=N, dim=-1).to(dtype)


def grad_maps(net, x0: torch.Tensor, t_idx: int, n_feat: int, gen: torch.Generator, frozen: bool):
    """Squared input gradient, summed over detectors, of ``n_feat`` random active
    features of the last stage at time index ``t_idx``. Returns (n_feat, N)
    squared gradients, the chosen channels, and their activations."""
    ctx = frozen_norm_stats() if frozen else contextlib.nullcontext()
    with ctx:
        with torch.no_grad():
            f0 = features(net, x0)[0, :, t_idx]
        active = torch.nonzero(f0 > 0).flatten()
        pick = active[torch.randperm(len(active), generator=gen)[:n_feat]]
        x = x0.repeat(len(pick), 1, 1).clone().requires_grad_(True)
        f = features(net, x)
        sel = f[torch.arange(len(pick)), pick, t_idx]
        sel.sum().backward()
    g2 = x.grad.pow(2).sum(1)  # (n_feat, N)
    return g2.detach().cpu().numpy(), pick.numpy(), f0[pick].numpy(), len(active)


def centered_span(p: np.ndarray, c: int, q: float) -> int:
    """Width 2w+1 of the smallest interval [c-w, c+w] holding fraction q of p."""
    P = p.sum()
    cs = np.concatenate([[0.0], np.cumsum(p)])
    for w in range(len(p)):
        a, b = max(c - w, 0), min(c + w, len(p) - 1)
        if cs[b + 1] - cs[a] >= q * P:
            return 2 * w + 1
    return 2 * len(p) - 1


def shortest_span(p: np.ndarray, q: float) -> int:
    """Length of the shortest contiguous interval anywhere holding fraction q of p."""
    S = np.concatenate([[0.0], np.cumsum(p)])
    target = S[:-1] + q * S[-1]
    b = np.searchsorted(S, target * (1 - 1e-12), side="left")
    ok = b <= len(p)
    return int((b[ok] - np.arange(len(p))[ok]).min())


def nonzero_interval(p: np.ndarray):
    nz = np.flatnonzero(p > 0)
    if len(nz) == 0:
        return None, None, 0
    return int(nz[0]), int(nz[-1]), int(nz[-1] - nz[0] + 1)
