"""Receptive field of the aframe ResNet (ml4gw ResNet1D, layers [3, 4, 6, 3],
kernel 3) at 2048 Hz, by layer arithmetic and by the support of an input
gradient. For the gradient, GroupNorm and ReLU become identities and max
pooling becomes average pooling, so the support is the structural one."""
import torch
from torch import nn
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.nn.resnet.resnet_1d import ResNet1D

torch.set_num_threads(2)
FS, L = 2048, 8192


def arithmetic(layers=(3, 4, 6, 3), k=3):
    r, j = 1, 1
    r, j = r + (7 - 1) * j, j * 2          # stem conv k7 s2
    r, j = r + (3 - 1) * j, j * 2          # maxpool k3 s2
    for i, n in enumerate(layers):
        for b in range(n):
            s = 2 if (i > 0 and b == 0) else 1
            r, j = r + (k - 1) * j, j * s  # conv1 (strided in first block)
            r = r + (k - 1) * j            # conv2
    return r, j


def strip(m):
    for name, c in list(m.named_children()):
        if isinstance(c, (nn.GroupNorm, nn.BatchNorm1d, nn.ReLU)) or c.__class__.__name__.startswith("GroupNorm"):
            setattr(m, name, nn.Identity())
        elif isinstance(c, nn.MaxPool1d):
            setattr(m, name, nn.AvgPool1d(3, 2, padding=1))
        else:
            strip(c)


net = ResNet1D(2, layers=[3, 4, 6, 3], classes=235, kernel_size=3, norm_layer=GroupNorm1DGetter(groups=16)).double()
n_gn = sum(1 for m in net.modules() if m.__class__.__name__.startswith("GroupNorm") or isinstance(m, nn.GroupNorm))
strip(net)
x = torch.randn(1, 2, L, dtype=torch.float64, requires_grad=True)
y = net.relu(net.maxpool(net.relu(net.bn1(net.conv1(x)))))
for layer in net.residual_layers:
    y = layer(y)
c = y.shape[-1] // 2
y[0, :, c].sum().backward()
g = x.grad.abs().sum(1)[0]
nz = torch.nonzero(g > 1e-12 * g.max()).flatten()
span = int(nz.max() - nz.min() + 1)
r, j = arithmetic()
print(f"GroupNorm layers in the original net: {n_gn}")
print(f"final feature length {y.shape[-1]} (stride {L // y.shape[-1]})")
print(f"arithmetic receptive field {r} samples = {r / FS:.4f} s, jump {j}")
print(f"gradient support of the centre output: {span} samples = {span / FS:.4f} s "
      f"(samples {int(nz.min())}..{int(nz.max())})")
