"""Numerical checks for the dropout-average derivation: Dropout1d behaviour, moments of the
dropout factor, and averages of GELU, square and GLU under dropout. CPU only."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import integrate, stats

torch.manual_seed(0)
x = torch.ones(1000, 64, 100)
d = nn.Dropout1d(0.2)
d.train()
y = d(x)
vals = torch.unique(y)
const_t = bool((y == y[..., :1]).all())
print("Dropout1d(0.2) train: values", vals.tolist(), "constant along time", const_t,
      "zeroed channels %.4f" % (y[..., 0] == 0).float().mean().item(), "mean %.4f" % y.mean().item())
d.eval()
print("eval identity", bool((d(x) == x).all()))
d0 = nn.Dropout1d(0.0); d0.train()
xr = torch.randn(8, 64, 100)
print("p=0 train identity", bool((d0(xr) == xr).all()))

rng = np.random.default_rng(1)
p = 0.2
xi = (rng.random(10_000_000) >= p) / (1 - p)
print("xi mean %.5f mean sq %.5f var %.5f" % (xi.mean(), (xi ** 2).mean(), xi.var()))

# exact averages over the two values of xi
g = lambda v: F.gelu(torch.tensor(v, dtype=torch.float64)).item()
for v in (1.0, -1.0):
    avg = (1 - p) * g(v / (1 - p)) + p * g(0.0)
    print("GELU v=%+.0f: average over xi %.4f, dropout off %.4f" % (v, avg, g(v)))
# GLU a*sigmoid(b) when a and b are both linear in one dropped value u: a = u, b = u
sig = lambda z: 1 / (1 + np.exp(-z))
for u in (1.0, -1.0):
    avg = (1 - p) * (u / (1 - p)) * sig(u / (1 - p)) + p * 0.0
    print("GLU(u, u) u=%+.0f: average %.4f, off %.4f" % (u, avg, u * sig(u)))

# expected maximum of n standard Gaussians, exact integral
for n in (1000, 8192):
    f = lambda t: t * n * stats.norm.pdf(t) * stats.norm.cdf(t) ** (n - 1)
    val, err = integrate.quad(f, -10, 10, limit=200)
    print("e_%d = %.5f (quad error %.1e)" % (n, val, err))
# Student t, 15 degrees of freedom, beyond +3
print("P(t15 > 3) = %.5f" % stats.t.sf(3, 15))
