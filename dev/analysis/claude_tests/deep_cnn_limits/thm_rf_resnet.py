"""Receptive field, total stride and feature length of aframe's ResNet1D
(ml4gw, layers [3,4,6,3], kernel 3, GroupNorm 16), as used by ResNetGridNet.

Two independent checks:
1. the analytic recursion r += (k-1)*jump, jump *= stride over every conv/pool;
2. autograd: gradient of one feature at the last residual layer (before
   AdaptiveAvgPool1d) with respect to the input; its nonzero support is the
   receptive field. GroupNorm is replaced by identity for this check, since
   its statistics span the whole time axis and would make every input sample
   reach every output (that global coupling is reported separately).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch
torch.set_num_threads(4)
from ml4gw.nn.resnet.resnet_1d import ResNet1D
from ml4gw.nn.norm import GroupNorm1DGetter

torch.manual_seed(0)

# 1. analytic recursion
layers = [3, 4, 6, 3]
r, jump = 1, 1
def add(k, s):
    global r, jump
    r += (k - 1) * jump
    jump *= s
add(7, 2)   # stem conv
add(3, 2)   # maxpool
for i, nb in enumerate(layers):
    for b in range(nb):
        s = 2 if (i > 0 and b == 0) else 1
        add(3, s)   # conv1 of the block (strided in the first block of stages 2-4)
        add(3, 1)   # conv2
print(f"analytic receptive field {r} samples, total stride {jump}")

# 2. autograd
class Id(torch.nn.Module):
    def __init__(self, c): super().__init__()
    def forward(self, x): return x
net = ResNet1D(2, layers=layers, classes=235, kernel_size=3, norm_layer=lambda c: Id(c)).double()
net.eval()
N = 8192
x = torch.randn(1, 2, N, dtype=torch.float64, requires_grad=True)
h = net.relu(net.conv1(x)); h = net.maxpool(h)
for layer in net.residual_layers:
    h = layer(h)
print("feature map before pooling:", tuple(h.shape))
P = h.shape[-1]
p = P // 2
g, = torch.autograd.grad(h[0, :, p].sum(), x, retain_graph=True)
nz = (g[0].abs().sum(0) > 0).nonzero().flatten()
print(f"autograd: feature {p} of {P} depends on input samples {nz.min().item()}..{nz.max().item()} "
      f"(span {nz.max().item() - nz.min().item() + 1})")
# stride check: support of feature p+1 shifted by how much
g2, = torch.autograd.grad(h[0, :, p + 1].sum(), x)
nz2 = (g2[0].abs().sum(0) > 0).nonzero().flatten()
print(f"autograd: feature {p+1} support starts at {nz2.min().item()} (shift {nz2.min().item() - nz.min().item()})")

# GroupNorm version: global coupling
net2 = ResNet1D(2, layers=layers, classes=235, kernel_size=3, norm_layer=GroupNorm1DGetter(groups=16)).double()
net2.eval()
h = net2.relu(net2.bn1(net2.conv1(x))); h = net2.maxpool(h)
for layer in net2.residual_layers:
    h = layer(h)
g3, = torch.autograd.grad(h[0, :, p].sum(), x)
frac = (g3[0].abs().sum(0) > 0).double().mean().item()
print(f"with GroupNorm: fraction of input samples reaching feature {p}: {frac:.3f}")
