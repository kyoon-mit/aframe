"""Architecture checks: parameter counts, how many channel dropouts act in one forward
pass and what follows them, and the group norm's invariance to a per-group scale and
offset. CPU only, short random inputs."""
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/projects/train")
from train.experimental.oracle.arch import ComplexS4Regressor, S4Regressor  # noqa: E402
from ml4gw.nn.norm import GroupNorm1D  # noqa: E402

ARCH = dict(num_ifos=2, num_params=1, d_model=64, d_state=32, n_layers=4, prenorm=True,
            num_groups=32, dt_min=1e-4, dt_max=10.0)
torch.manual_seed(0)
x = torch.randn(2, 2, 1024)
for name, m in [("real max", S4Regressor(**ARCH, dropout=0.2, pool="max")),
                ("real mean", S4Regressor(**ARCH, dropout=0.2, pool="mean")),
                ("complex energy max", ComplexS4Regressor(**ARCH, dropout=0.2, readout="energy_max"))]:
    n_par = sum(p.numel() for p in m.parameters())
    calls = []
    hooks = [mod.register_forward_hook(lambda mod, i, o, n=n: calls.append(n))
             for n, mod in m.named_modules() if isinstance(mod, nn.Dropout1d) and mod.p > 0]
    m.train()
    with torch.no_grad():
        m(x)
    for h in hooks:
        h.remove()
    used = set()
    for n, p in m.named_parameters():
        pass
    print(f"{name}: {n_par} parameters; channel dropouts with p > 0 called in one forward: {len(calls)}")
    print("   ", calls)
    # parameters that receive a gradient (used in the forward pass)
    m.zero_grad()
    m(x).sum().backward()
    n_used = sum(p.numel() for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"    parameters used in the forward pass: {n_used}")

gn = GroupNorm1D(64, 32)
with torch.no_grad():
    gn.weight.normal_()
    gn.bias.normal_()
z = torch.randn(4, 64, 8192)
a = torch.rand(4, 32, 1, 1) * 2 + 0.5
b = torch.randn(4, 32, 1, 1)
z2 = (z.view(4, 32, 2, 8192) * a + b).view(4, 64, 8192)
with torch.no_grad():
    d = (gn(z2) - gn(z)).abs().max().item()
    gn.eval()
    d_eval = (gn(z) - gn.train()(z)).abs().max().item()
print(f"GroupNorm1D: max |GN(a x + b) - GN(x)| for a per-group scale a in [0.5, 2.5] and offset b: {d:.2e}")
print(f"GroupNorm1D: max |eval - train| output difference: {d_eval:.1e}")
