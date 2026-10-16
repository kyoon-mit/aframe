"""Independent check of the nonlinearity-section numbers (closed forms and quadrature)."""
import math
import numpy as np
from scipy import integrate, optimize, special

SQ2PI = math.sqrt(2 * math.pi)


def Phi(x):
    return 0.5 * special.erfc(-x / math.sqrt(2))


def phi(x):
    return np.exp(-0.5 * x * x) / SQ2PI


def gelu(x):
    return x * Phi(x)


def relu(x):
    return np.maximum(0.0, x)


def heav(x):
    return (x > 0).astype(float) if isinstance(x, np.ndarray) else float(x > 0)


def moments(f, sigma, a=0.0):
    """E f(a+n), E f(a+n)^2, E n f(a+n) for n~N(0,sigma^2) by quad on y=a+n."""
    def p(y):
        return math.exp(-0.5 * ((y - a) / sigma) ** 2) / (sigma * SQ2PI)
    L = 40 * sigma + abs(a)
    pts = [0.0] if -L < 0 < L else None
    m1 = integrate.quad(lambda y: f(y) * p(y), -L, L, points=pts, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    m2 = integrate.quad(lambda y: f(y) ** 2 * p(y), -L, L, points=pts, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    mn = integrate.quad(lambda y: (y - a) * f(y) * p(y), -L, L, points=pts, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    mh = integrate.quad(lambda y: ((y - a) ** 2 - sigma ** 2) * f(y) * p(y), -L, L, points=pts, limit=400, epsabs=1e-15, epsrel=1e-13)[0]
    return m1, m2, mn, mh


def r_num(f, sigma):
    m1, m2, mn, _ = moments(f, sigma)
    return mn / (sigma * math.sqrt(m2 - m1 * m1))


def r_gelu_closed(s):
    c = s * s / (1 + s * s)
    return (1 + (2 / math.pi) * (math.asin(c) - c + 2 * c / math.sqrt(1 + 2 * s * s))) ** -0.5


fg = lambda y: y * 0.5 * math.erfc(-y / math.sqrt(2))
fr = lambda y: max(0.0, y)
fh = lambda y: 1.0 if y > 0 else 0.0
fsq = lambda y: y * y
fab = lambda y: abs(y)

print("ReLU closed", 1 / math.sqrt(2 - 2 / math.pi), "num s=1", r_num(fr, 1.0), "s=3", r_num(fr, 3.0))
print("thr closed", math.sqrt(2 / math.pi), "num s=1", r_num(fh, 1.0), "s=0.4", r_num(fh, 0.4))
print("sign num", r_num(lambda y: 1.0 if y > 0 else -1.0, 1.3))
for s in [0.01, 0.25, 0.5, 1.0, 1.678, 2.0, 4.0, 10.0, 1000.0]:
    rc = r_gelu_closed(s)
    rn = r_num(fg, s) if s >= 0.25 and s <= 10 else float('nan')
    print(f"GELU s={s}: closed {rc:.9f} num {rn:.9f}")
print("small-sigma 1-2s^2/pi at 0.01:", 1 - 2 * 0.0001 / math.pi)
res = optimize.minimize_scalar(r_gelu_closed, bounds=(0.5, 5), method="bounded", options={"xatol": 1e-10})
print("min r_gelu", res.fun, "at", res.x)
# single minimum check
ss = np.geomspace(1e-3, 1e3, 200001)
rr = np.array([r_gelu_closed(x) for x in ss])
d = np.diff(rr)
print("sign changes of slope:", int(np.sum(np.diff(np.sign(d)) != 0)))
print("r_gelu range on [0.5,2]:", rr[(ss >= 0.5) & (ss <= 2)].min(), rr[(ss >= 0.5) & (ss <= 2)].max())

# gain g = E[n phi]/sigma^2
for s in [0.5, 1, 2]:
    for name, f in [("gelu", fg), ("relu", fr)]:
        m1, m2, mn, mh = moments(f, s)
        print(f"g {name} s={s}: {mn/s**2:.15f}")

# h at sigma 1
for name, f in [("gelu", fg), ("relu", fr), ("thr", fh)]:
    m1, m2, mn, mh = moments(f, 1.0)
    print(f"h {name}: {mh:.9f}")
print("3/(4 sqrt pi)", 3 / (4 * math.sqrt(math.pi)), "1/sqrt(2pi)", 1 / SQ2PI)

# r2 values
def r2(f, s):
    m1, m2, mn, mh = moments(f, s)
    return mh / (math.sqrt(2) * s * s * math.sqrt(m2 - m1 * m1))
for s in [0.5, 1, 2]:
    print(f"r2 gelu s={s}: {r2(fg, s):.6f} -> r2*0.5 = {0.5*r2(fg, s):.5f}")
print("r2 relu", r2(fr, 1.0), 0.5 * r2(fr, 1.0))
print("r2 abs", r2(fab, 1.0), 0.5 * r2(fab, 1.0))
print("r2 sq", r2(fsq, 1.0))

# exact SNR for N=600 pieces each a=8/sqrt(600), sigma=1, equal weights
N = 600
a = 8 / math.sqrt(N)
print("a", a)
for name, f in [("gelu", fg), ("relu", fr), ("thr", fh), ("sq", fsq)]:
    m1a = moments(f, 1.0, a)[0]
    m10, m20, _, _ = moments(f, 1.0, 0.0)
    snr = math.sqrt(N) * (m1a - m10) / math.sqrt(m20 - m10 ** 2)
    print(f"SNR600 {name}: {snr:.5f}")
# closed forms of m(a)
b = a / math.sqrt(2)
mg = a * Phi(b) + 1 * phi(b) / math.sqrt(2)
print("m_gelu closed", mg, "num", moments(fg, 1.0, a)[0])
print("m_relu closed", a * Phi(a) + phi(a), "num", moments(fr, 1.0, a)[0])
print("r*8 gelu, relu, thr:", 8 * r_gelu_closed(1.0), 8 / math.sqrt(2 - 2 / math.pi), 8 * math.sqrt(2 / math.pi))
print("-a^2/6:", -a * a / 6)

# threshold exact for K=8192 +/-a in-step, rho=8, sigma=1
K = 8192
aK = 8 / math.sqrt(K)
print("thr K=8192 exact in-step:", 2 * math.sqrt(K) * (Phi(aK) - 0.5))
# cosine 101.3 Hz, amplitude A with ||s||^2=64 approx, averaged over phases
fs = 2048.0
t = np.arange(K) / fs
vals = []
for th in np.linspace(0, 2 * np.pi, 4000, endpoint=False):
    s = np.cos(2 * np.pi * 101.3 * t + th)
    s *= 8 / np.linalg.norm(s)
    shift = np.sum(s * (Phi(s) - 0.5)) / np.linalg.norm(s)
    vals.append(shift / 0.5)
print("thr cosine exact (per-phase normalised ||s||=8), mean over 4000 phases:", np.mean(vals))
# x^3-3x
for s in [0.5, 1, 2]:
    print("E n(n^3-3n)", s, 3 * s ** 4 - 3 * s ** 2, moments(lambda y: y ** 3 - 3 * y, s)[2])
# m(a)-m(-a)-a
mx = 0
for s in [0.5, 1, 2]:
    for aa in [0.1, 1, 3]:
        for f in [fg, fr]:
            mx = max(mx, abs(moments(f, s, aa)[0] - moments(f, s, -aa)[0] - aa))
print("max |m(a)-m(-a)-a|", mx)
# percentages
print("r^2 gelu at 0.5..2:", r_gelu_closed(0.5) ** 2, r_gelu_closed(2) ** 2, "relu", 1 / (2 - 2 / math.pi), "thr", 2 / math.pi)
