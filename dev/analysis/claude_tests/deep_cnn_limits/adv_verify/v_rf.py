"""Adversarial check of the ResNet receptive-field result.

1. Layer list read from the built ml4gw ResNet1D; r/j recursion; interval
   pull-back for last-stage indices 0, 128, 255.
2. Autograd support of one last-stage feature, GroupNorm statistics frozen
   (detached) and live, at initialization.
3. Trained ck_244: ERF (centered 50/90/99% widths) with statistics frozen
   and live, fewer seeds than the original run.
"""
import sys
import numpy as np
import torch
from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.nn.norm import GroupNorm1DGetter, GroupNorm1D

torch.set_num_threads(8)
L = 8192
CKPT = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/"
        "CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/checkpoints/ck_244.ckpt")

FROZEN = {"on": False}
_orig = GroupNorm1D.forward


def gn_forward(self, x):
    keep = self.num_groups == self.num_channels
    mean = x.mean(-1, keepdims=keep)
    sq = (x ** 2).mean(-1, keepdims=keep)
    if not keep:
        m = torch.stack([mean, sq], 1).reshape(-1, 2, self.num_groups, self.channels_per_group)
        m = m.mean(-1, keepdims=True).expand(-1, -1, -1, self.channels_per_group).reshape(-1, 2, self.num_channels, 1)
        mean, sq = m[:, 0], m[:, 1]
    if FROZEN["on"]:
        mean, sq = mean.detach(), sq.detach()
    std = (sq - mean ** 2 + self.eps) ** 0.5
    scale = self.weight / std
    return self.bias - scale * mean + x * scale


GroupNorm1D.forward = gn_forward


def build():
    return ResNet1D(2, layers=[3, 4, 6, 3], classes=235, kernel_size=3,
                    norm_layer=GroupNorm1DGetter(groups=16)).double().eval()


def features(net, x):
    x = net.maxpool(net.relu(net.bn1(net.conv1(x))))
    for layer in net.residual_layers:
        x = layer(x)
    return x


def kspd(m):
    g = lambda v: v[0] if isinstance(v, tuple) else v
    return g(m.kernel_size), g(m.stride), g(m.padding), g(getattr(m, "dilation", 1))


def main_path(net):
    out = [("stem", net.conv1), ("maxpool", net.maxpool)]
    for li, layer in enumerate(net.residual_layers):
        for bi, blk in enumerate(layer):
            out += [(f"s{li+1}b{bi}c1", blk.conv1), (f"s{li+1}b{bi}c2", blk.conv2)]
            if blk.downsample is not None:
                ds = [m for m in blk.downsample.modules() if isinstance(m, torch.nn.Conv1d)][0]
                print(f"  downsample s{li+1}b{bi}: k,s,p,d = {kspd(ds)}")
    return out


def pullback(layers, a, b):
    for _, m in reversed(layers):
        k, s, p, d = kspd(m)
        a, b = a * s - p, b * s - p + (k - 1) * d
    return a, b


def noise(seed, B=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, 2, L, generator=g, dtype=torch.float64)
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(L, 1 / 2048)
    X[..., f < 20] = 0
    return torch.fft.irfft(X, n=L)


def centered_widths(p, fracs=(0.5, 0.9, 0.99), c=4096):
    p = p / p.sum()
    out = []
    for q in fracs:
        for h in range(0, L):
            if p[max(0, c - h):min(L, c + h + 1)].sum() >= q:
                out.append(2 * h + 1)
                break
    return out


def erf(net, seeds, per=8, frozen=True):
    FROZEN["on"] = frozen
    maps, nact = [], []
    for sd in seeds:
        x0 = noise(1000 + sd)
        with torch.no_grad():
            f0 = features(net, x0)[0, :, 128]
        act = torch.nonzero(f0 > 0).flatten()
        nact.append(len(act))
        if len(act) == 0:
            continue
        g = torch.Generator().manual_seed(sd)
        ch = act[torch.randperm(len(act), generator=g)[:per]]
        x = x0.repeat(len(ch), 1, 1).requires_grad_(True)
        f = features(net, x)
        f[torch.arange(len(ch)), ch, 128].sum().backward()
        gm = (x.grad ** 2).sum(1).numpy()
        for m in gm:
            maps.append(m / m.sum())
    FROZEN["on"] = False
    maps = np.array(maps)
    nz = [(int(np.flatnonzero(m > 0)[0]), int(np.flatnonzero(m > 0)[-1]), int((m > 0).sum())) for m in maps]
    inside = maps[:, 3647:4546].sum(1)
    return dict(n=len(maps), nact_mean=float(np.mean(nact)), widths=centered_widths(maps.mean(0)),
                nz_first_last_count=sorted(set(nz))[:5], n_full=int(sum(1 for a in nz if a[2] == L)),
                inside_mean=float(inside.mean()), inside_min=float(inside.min()))


def main():
    net = build()
    print("main path (name, k, s, p, d):")
    layers = main_path(net)
    ends = {"stem", "maxpool"} | {f"s{li+1}b{len(l) - 1}c2" for li, l in enumerate(net.residual_layers)}
    r, j = 1, 1
    for name, m in layers:
        k, s, p, d = kspd(m)
        r += (k - 1) * d * j
        j *= s
        if name in ends:
            print(f"  after {name}: r={r} j={j}  (k,s,p,d of last = {k},{s},{p},{d})")
    print(f"RF = {r} samples = {r / 2048:.4f} s = {100 * r / L:.2f}% ; jump {j}; layers {len(layers)}")
    with torch.no_grad():
        print("last-stage shape:", tuple(features(net, noise(0)).shape))
    for i in (0, 128, 255):
        a, b = pullback(layers, i, i)
        print(f"index {i}: [{a}, {b}] clipped [{max(a, 0)}, {min(b, L - 1)}] -> {min(b, L - 1) - max(a, 0) + 1} samples")
    # autograd support at init
    for frozen in (True, False):
        FROZEN["on"] = frozen
        x = noise(5).requires_grad_(True)
        f = features(net, x)
        c = int(torch.argmax(f[0, :, 128]))
        f[0, c, 128].backward()
        nzi = np.flatnonzero((x.grad ** 2).sum(1)[0].numpy() > 0)
        print(f"init, GN {'frozen' if frozen else 'live'}: grad nonzero on {len(nzi)} samples, [{nzi[0]}, {nzi[-1]}]")
    FROZEN["on"] = False
    # init ERF
    print("init ERF frozen:", erf(net, range(10), frozen=True))
    print("init ERF live:", erf(net, range(10), frozen=False))
    # trained
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k.split("model.net.", 1)[1]: v for k, v in sd.items() if k.startswith("model.net.")}
    net2 = build()
    missing = net2.load_state_dict({k: v.double() for k, v in sd.items()}, strict=True)
    print("loaded ck_244:", missing)
    print("ck_244 ERF frozen:", erf(net2, range(30), frozen=True))
    print("ck_244 ERF live:", erf(net2, range(30), frozen=False))


if __name__ == "__main__":
    main()
