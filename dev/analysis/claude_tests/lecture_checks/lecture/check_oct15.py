"""Independent checks for the 15 Oct lecture-style update (derivations oct15-*).

Leading-order (Newtonian) chirp throughout. Run: uv run -q --with numpy --with sympy python -I check_oct15.py
"""
import numpy as np
import sympy as sp

T_SUN = 4.925490947e-6  # G M_sun / c^3 in s


def tm(mc):  # G Mc / c^3 in s
    return T_SUN * mc


def phase_to_merger(tau, mc):  # |Phi(tau)|, radians accumulated from tau before merger to merger
    return 2 * (tau / (5 * tm(mc))) ** (5 / 8)


def f_of_tau(tau, mc):
    return (5 / tau) ** (3 / 8) * tm(mc) ** (-5 / 8) / (8 * np.pi)


def tau_of_f(f, mc):
    return 5 / 256 * tm(mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def fdot(f, mc):
    return 96 / 5 * np.pi ** (8 / 3) * tm(mc) ** (5 / 3) * f ** (11 / 3)


print("A) time a chirp stays within 1/8 cycle of a fixed-frequency wave, exact phase vs fdot^(-1/2)")
for mc in (0.85, 1.2, 2.75):
    for f0 in (50.0, 100.0, 300.0):
        tau0 = tau_of_f(f0, mc)
        Ts = fdot(f0, mc) ** -0.5
        s = np.linspace(-1.5 * Ts, 1.5 * Ts, 600001)
        tau = tau0 - s  # s > 0 is later in time
        ok = tau > 0
        s, tau = s[ok], tau[ok]
        # Phi(t) = -|Phi(tau)|, increasing toward merger; line tangent at s = 0
        dphi = -phase_to_merger(tau, mc) - (-phase_to_merger(tau0, mc) + 2 * np.pi * f0 * s)
        inside = np.abs(dphi) <= np.pi / 4
        i0 = np.argmin(np.abs(s))
        lo = i0
        while lo > 0 and inside[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(s) - 1 and inside[hi + 1]:
            hi += 1
        length = s[hi] - s[lo]
        win = slice(lo, hi + 1)
        kept = abs(np.exp(1j * dphi[win]).mean())
        # twice the interval, centred
        w2 = np.abs(s) <= Ts
        kept2 = abs(np.exp(1j * dphi[w2]).mean())
        print(f"  Mc {mc}, {f0:g} Hz: fdot {fdot(f0, mc):.4g} Hz/s, fdot^-1/2 {Ts:.4g} s, exact length {length:.4g} s "
              f"(ratio {length / Ts:.4f}); in-step sum kept inside {kept:.4f}, over twice the time {kept2:.4f}")
u = np.linspace(-0.5, 0.5, 200001)
print(f"  quadratic phase only: kept over |u|<=1/2: {abs(np.exp(1j * np.pi * u ** 2).mean()):.4f}")
u2 = np.linspace(-1, 1, 400001)
print(f"  quadratic phase only: kept over |u|<=1:   {abs(np.exp(1j * np.pi * u2 ** 2).mean()):.4f}")
for mc in (0.85, 1.2, 2.75):
    print(f"  Mc {mc}: frequency 4 s before merger {f_of_tau(4.0, mc):.2f} Hz")

print("B) steady phase drift of k cycles across a sum")
x = np.linspace(0, 1, 1000001)
for k in (0.25, 0.5, 1.0):
    num = abs(np.exp(2j * np.pi * k * x).mean())
    print(f"  k={k}: numeric {num:.5f}, |sin(pi k)/(pi k)| {abs(np.sin(np.pi * k) / (np.pi * k)):.5f}")
kk = np.linspace(1.0001, 2.0, 2000000)
side = np.abs(np.sin(np.pi * kk) / (np.pi * kk))
print(f"  largest side peak {side.max():.4f} at k={kk[side.argmax()]:.4f}")
print(f"  frequency error over 4 s that keeps 0.90: {0.25 / 4:.4f} Hz; that loses all: {1 / 4:.2f} Hz")
# discrete sum check: 600 pieces with phases 2 pi k (j + 1/2)/600
for k in (0.25, 0.5, 1.0):
    j = np.arange(600)
    print(f"  600 pieces, k={k}: {abs(np.exp(2j * np.pi * k * (j + 0.5) / 600).mean()):.5f}")

print("C) adding in step against adding powers (formulas of #semi-coherent)")
N, rho = 600, 8.0
print(f"  per-piece SNR {rho / np.sqrt(N):.4f}; in-step: signal {N * rho / np.sqrt(N):.1f} vs noise {np.sqrt(N):.2f}, ratio {rho:.1f}")
print(f"  power sum: rho^2/sqrt(2N) = {rho ** 2 / np.sqrt(2 * N):.3f}; (2N)^(1/4) = {(2 * N) ** 0.25:.3f}")
rng = np.random.default_rng(1)
trials = 20000
z = rho / np.sqrt(N) + rng.standard_normal((trials, N))
P = (z ** 2).sum(1)
z0 = rng.standard_normal((trials, N))
P0 = (z0 ** 2).sum(1)
print(f"  Monte Carlo: power sum stands out by {(P.mean() - P0.mean()) / P0.std():.3f} sd; in-step sum by "
      f"{(z.sum(1).mean() / np.sqrt(N)):.3f} sd")
print("  1% change: 11.90 and 24.78 rad =", f"{11.90 / (2 * np.pi):.2f} and {24.78 / (2 * np.pi):.2f} cycles")

print("D) leftover cycles after heterodyning; error defined by true = assumed * (1 + eps)")
masses = np.geomspace(0.85, 2.75, 4001)
for eps in (0.001, 0.01, 0.02, 0.05, 0.10):
    vals = []
    vals_truth = []
    for T in (3.5, 4.0):
        for sgn in (+1, -1):
            e = sgn * eps
            mref = masses / (1 + e)
            left = np.abs(phase_to_merger(T, masses) - phase_to_merger(T, mref)) / (2 * np.pi)
            formula = phase_to_merger(T, masses) / (2 * np.pi) * np.abs(1 - (1 + e) ** (5 / 8))
            assert np.allclose(left, formula, rtol=1e-10)
            vals.append(left)
            # alternative: assumed = true * (1 + e)
            mref2 = masses * (1 + e)
            vals_truth.append(np.abs(phase_to_merger(T, masses) - phase_to_merger(T, mref2)) / (2 * np.pi))
    v = np.concatenate(vals)
    v2 = np.concatenate(vals_truth)
    print(f"  {100 * eps:g}%: {v.min():.2f} to {v.max():.2f} cycles   (if assumed = true*(1+-eps): {v2.min():.2f} to {v2.max():.2f})")
# the leftover grows monotonically toward the window start
tau = np.linspace(1e-3, 4.0, 1000)
d = np.abs(phase_to_merger(tau, 1.2) - phase_to_merger(tau, 1.2 / 1.01))
print(f"  leftover is largest at the window start: {np.all(np.diff(d) > 0)}")
# leftover frequency ratio
for e in (0.01, -0.01, 0.1, -0.1):
    print(f"  eps {e:+}: leftover frequency / chirp frequency = {abs(1 - (1 + e) ** (5 / 8)):.5f} (5/8 eps = {5 / 8 * abs(e):.5f})")


def grid_leftover(lo, hi, n, T):
    grid = np.geomspace(lo, hi, n)
    m = masses[(masses >= lo) & (masses <= hi)]
    ref = grid[np.abs(np.log(m[:, None] / grid[None, :])).argmin(1)]
    left = np.abs(phase_to_merger(T, m) - phase_to_merger(T, ref)) / (2 * np.pi)
    err = np.abs(m / ref - 1)
    return err.max(), left.max(), np.median(left), grid


for lo, hi, n in ((0.85, 2.75, 10), (1.0, 2.5, 10), (1.0, 2.5, 100)):
    for T in (3.5, 4.0):
        e, mx, md, grid = grid_leftover(lo, hi, n, T)
        print(f"  {n} masses over {lo}-{hi}, window {T} s: neighbour ratio {grid[1] / grid[0]:.6f}, worst error {100 * e:.3f}%, "
              f"leftover worst {mx:.2f}, median {md:.2f} cycles")

print("E) grid masses of the 100-mass grid (1.0-2.5, log) within +-e of an estimate")
g = np.geomspace(1.0, 2.5, 100)
est = np.linspace(1.15, 2.2, 20001)
for e in (0.05, 0.10):
    c = ((g[None, :] >= est[:, None] * (1 - e)) & (g[None, :] <= est[:, None] * (1 + e))).sum(1)
    print(f"  +-{100 * e:g}%: {c.min()} to {c.max()} (estimates 1.15-2.2), expected {np.log((1 + e) / (1 - e)) / np.log(2.5) * 99:.2f}")
    print(f"     range check: 1.15*(1-e)={1.15 * (1 - e):.4f} >= 1.0, 2.2*(1+e)={2.2 * (1 + e):.4f} <= 2.5")

print("F) complex first-order equation -> damped driven oscillator, symbolic")
t = sp.symbols("t", real=True)
a, w = sp.symbols("a omega", positive=True)
p, q, u = sp.Function("p")(t), sp.Function("q")(t), sp.Function("u")(t)
dp = -a * p - w * q + u  # real part of x' = (-a + i w) x + u, x = p + i q
dq = w * p - a * q       # imaginary part
pdd = sp.diff(dp, t).subs({sp.Derivative(p, t): dp, sp.Derivative(q, t): dq})
expr = sp.simplify(pdd + 2 * a * dp + (a ** 2 + w ** 2) * p - (sp.diff(u, t) + a * u))
print(f"  p'' + 2a p' + (a^2 + w^2) p - (u' + a u) = {expr}")
xs = sp.symbols("x")
lhs = sp.expand(((-a + sp.I * w) * (p + sp.I * q) + u))
print(f"  Re/Im of (-a + i w)(p + i q) + u: {sp.simplify(sp.re(lhs))} | {sp.simplify(sp.im(lhs))}")
print(f"  free oscillation: roots of s^2 + 2a s + a^2 + w^2: {sp.solve(xs ** 2 + 2 * a * xs + a ** 2 + w ** 2, xs)}")

print("G) data")
print(f"  training background 100011 s = {100011 / 86400:.4f} days; validation 41841 s = {41841 / 86400:.4f} days; "
      f"logged train figure 100011-41841 = {(100011 - 41841) / 86400:.4f} days")
print(f"  windows per epoch 1564 * 32 = {1564 * 32}")
plain = [31.7, 30.9, 34.2, 34.1, 32.2]       # S4D 64x32x4, mean over time, epochs 44-48 (training log)
cplx = [96.9, 95.4, 96.8, 95.6, 96.3]        # complex last layer, energy and maximum, epochs 44-48 (training log)
print(f"  clean test, epochs 44-48: plain {np.mean(plain):.2f}%, complex energy-max {np.mean(cplx):.2f}% "
      f"(range {min(cplx)} to {max(cplx)})")

print("H) extra: total overlap of a steady wave with a chirp (quadratic phase), in units of fdot^(-1/2)")
for V in (0.5, 0.7071, 1.0, 2.0, 5.0, 20.0):
    v = np.linspace(-V, V, int(400000 * V) + 1)
    tot = abs(np.trapezoid(np.exp(1j * np.pi * v ** 2), v))
    print(f"  |integral over |v|<={V}| = {tot:.4f}")
Vs = np.linspace(0.05, 3.0, 2951)
tots = []
for V in Vs:
    v = np.linspace(-V, V, 20001)
    tots.append(abs(np.trapezoid(np.exp(1j * np.pi * v ** 2), v)))
tots = np.array(tots)
print(f"  largest {tots.max():.4f} at V={Vs[tots.argmax()]:.3f} (full interval {2 * Vs[tots.argmax()]:.3f} fdot^-1/2)")
mc50 = 1.2 * (f_of_tau(4.0, 1.2) / 50.0) ** (8 / 5)
print(f"  chirp mass whose window (4 s) starts at 50 Hz: {mc50:.3f} Msun; check f = {f_of_tau(4.0, mc50):.3f} Hz")
print(f"  1.2 Msun at 100 Hz: in-step fraction of 4 s = {fdot(100, 1.2) ** -0.5 / 4:.4f}")
print(f"  0.198% vs 1%: ratio {1 / 0.198:.2f}")
