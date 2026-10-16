import numpy as np
# SNR power laws
def cdf(x, a, b, k):  # density ∝ x^-k on [a,b]
    e = 1 - k
    return (x**e - a**e) / (b**e - a**e)
def median(a, b, k):
    from scipy.optimize import brentq
    return brentq(lambda x: cdf(x, a, b, k) - 0.5, a, b)
for k in (3, 4):
    for a, b in ((4, 50), (8, 50)):
        print(f"k={k} [{a},{b}] P(<8)={cdf(8,a,b,k):.4f} P(<12)={cdf(12,a,b,k):.4f} median={median(a,b,k):.4f}")
print("unbounded medians: k3 from4", 4*np.sqrt(2), "k4 from4", 4*2**(1/3))
# light travel time H1-L1 (ml4gw vertices, float64)
H = np.array([-2.16141492636e06, -3.83469517889e06, +4.60035022664e06])
import re, pathlib
src = pathlib.Path('/home/kyoon/SSM-BNS/aframe/projects/train/.venv/lib/python3.11/site-packages/ml4gw/utils/interferometer.py').read_text()
blk = src.split('elif name == "L1"')[1]
m = re.search(r'self.vertex = torch.Tensor\(\s*\(([^)]*)\)', blk)
L = np.array([float(s) for s in m.group(1).split(',')])
d = np.linalg.norm(H - L); c = 299792458.0
print("L1 vertex", L, "baseline m", d, "light time ms", d / c * 1e3)
for fs in (2048, 1024):
    n = int(0.01 * fs); print(fs, "Hz window samples", n, "ms", n / fs * 1e3)
# S4D counts
def s4d_layer(H, N): return H + H*(N//2)*2 + 2*H*(N//2) + H + 2*H*H + 2*H
print("layer 256x256", s4d_layer(256,256), "x4", 4*s4d_layer(256,256))
tot = 4*s4d_layer(256,256) + (2*256+256) + 4*2*256 + (256*2+2)
print("total S4D 256x256x4", tot)
print("layer 256x32", s4d_layer(256,32), "x4", 4*s4d_layer(256,32))
print("free kernel stored total", 256*8192*2 + 514, "ratio", tot/(256*8192*2+514))
print("step ratio", 208/39, "conv weights", 512*256, "+bias", 512*256+512)
print("mixing ops", 2*256*512*8192)
# band init spacing
print("spacing", (512-20)/256)
# DINGO-BNS relative half-width
print("dingo", 0.005/2.2*100, 0.005/1.0*100, 0.005/1.1975*100)
# MBTA ratios 12 Oct
for a,b in ((9.5e-3,8.6e-4),(1.3e-2,1.5e-3),(1.6e-2,1.8e-3),(1.8e-2,2.9e-3)): print("mbta/bank", a/b)
# chirp-mass grid bins
g = np.arange(np.log(0.85), np.log(2.75)+1e-9, 0.005); print("bins", len(g))
