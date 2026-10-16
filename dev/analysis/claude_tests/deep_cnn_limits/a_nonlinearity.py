"""(A) Weak signal through one nonlinearity: the factor r = corr(n, phi(n)).

For each phi and noise sigma:
  * E[phi'(n)] from E[n phi(n)] / sigma^2 (Stein), Var phi(n), r, and the
    second-order factor r2 = corr(phi(n), n^2 - sigma^2), all by adaptive
    quadrature against the Gaussian density (scipy.integrate.quad), compared
    with closed forms where they exist;
  * Monte Carlo of the in-step sum o = sum_k c_k phi(s_k + n_k), c_k = s_k, and
    of the plain sum (power sum) o = sum_k phi(s_k + n_k), on a constant-
    amplitude alternating signal s_k = +-a (sum_k s_k = sum_k s_k^3 = 0), with
    a = rho / sqrt(K); measured SNR = mean[o(n+s) - o(n)] / std[o(n)] with
    common noise draws; compared with the first-order prediction r * rho
    (in step) or r2 * rho^2 / sqrt(2K) (power sum), and with the exact
    value from the closed-form mean m(a) = E phi(a + n) (computed by quadrature).
Output: out/a_nonlinearity.txt
"""
import numpy as np
from scipy import integrate, special

SQ2PI = np.sqrt(2 * np.pi)

def Phi(x):
    return special.ndtr(x)

def gelu(x):
    return x * Phi(x)

PHIS = {
    "identity": lambda x: x,
    "ReLU": lambda x: np.maximum(x, 0.0),
    "sign": np.sign,
    "GELU": gelu,
    "step at 2 sigma, 1{z > 2 sigma}": None,      # filled per sigma
    "square": lambda x: x * x,
    "absolute value": np.abs,
    "even threshold, 1{|z| > 2 sigma}": None,     # filled per sigma
}

def gauss_expect(f, sigma, breaks=()):
    """E f(n), n ~ N(0, sigma^2), by quad on [-12 sigma, 12 sigma] split at breaks."""
    pts = sorted(set([-12 * sigma, 12 * sigma] + [b for b in breaks if abs(b) < 12 * sigma]))
    tot = 0.0
    for lo, hi in zip(pts[:-1], pts[1:]):
        v, err = integrate.quad(lambda x: f(x) * np.exp(-0.5 * (x / sigma) ** 2) / (sigma * SQ2PI),
                                lo, hi, epsabs=1e-14, epsrel=1e-13, limit=400)
        tot += v
    return tot

def moments(phi, sigma, breaks):
    m0 = gauss_expect(phi, sigma, breaks)
    m2 = gauss_expect(lambda x: phi(x) ** 2, sigma, breaks)
    e1 = gauss_expect(lambda x: x * phi(x), sigma, breaks)               # = sigma^2 E phi'
    e2 = gauss_expect(lambda x: (x * x - sigma ** 2) * phi(x), sigma, breaks)  # = sigma^4 E phi''
    var = m2 - m0 ** 2
    std = np.sqrt(max(var, 0.0))
    Ed1 = e1 / sigma ** 2
    r = sigma * Ed1 / std if std > 0 else np.nan
    r2 = e2 / (np.sqrt(2) * sigma ** 2 * std) if std > 0 else np.nan
    return dict(mean=m0, std=std, Ephi1=Ed1, Ephi2=e2 / sigma ** 4, r=r, r2=r2)

def phi_for(name, sigma):
    if name.startswith("step at 2"):
        return (lambda x: (x > 2 * sigma).astype(float) if isinstance(x, np.ndarray) else float(x > 2 * sigma)), (2 * sigma,)
    if name.startswith("even threshold"):
        return (lambda x: (np.abs(x) > 2 * sigma).astype(float) if isinstance(x, np.ndarray) else float(abs(x) > 2 * sigma)), (-2 * sigma, 2 * sigma)
    breaks = (0.0,) if name in ("ReLU", "sign", "absolute value") else ()
    return PHIS[name], breaks

def closed_form_r(name, sigma):
    if name == "identity":
        return 1.0
    if name == "ReLU":
        return 0.5 / np.sqrt(0.5 - 1 / (2 * np.pi))
    if name == "sign":
        return np.sqrt(2 / np.pi)
    if name.startswith("step at 2"):
        p = Phi(-2.0)
        return (np.exp(-2.0) / SQ2PI) / np.sqrt(p * (1 - p))
    if name in ("square", "absolute value") or name.startswith("even threshold"):
        return 0.0
    return np.nan

def closed_form_r2(name):
    if name == "square":
        return 1.0
    if name == "absolute value":
        return np.sqrt(2 / np.pi) / (np.sqrt(1 - 2 / np.pi) * np.sqrt(2))
    return np.nan

def main():
    rng = np.random.default_rng(20261015)
    lines = []
    P = lambda s="": (print(s), lines.append(s))
    P("(A) r = corr(n, phi(n)) = sigma E[phi'(n)] / std phi(n);  r2 = corr(phi(n), n^2 - sigma^2)")
    P("    quadrature (quad, 1e-13 relative) against closed form where one exists")
    P(f"{'phi':36s} {'sigma':>5s} {'E phi_prime':>12s} {'std phi':>10s} {'r (quad)':>10s} {'r (closed)':>10s} {'r2 (quad)':>10s} {'r2 closed':>10s}")
    table = {}
    for name in PHIS:
        sigmas = (0.5, 1.0, 2.0) if name == "GELU" else (1.0,)
        for sigma in sigmas:
            phi, br = phi_for(name, sigma)
            mo = moments(phi, sigma, br)
            rc = closed_form_r(name, sigma)
            r2c = closed_form_r2(name)
            table[(name, sigma)] = mo
            P(f"{name:36s} {sigma:5.2f} {mo['Ephi1']:12.6f} {mo['std']:10.6f} {mo['r']:10.6f} {rc:10.6f} {mo['r2']:10.6f} {r2c:10.6f}")
    # GELU std in closed form check: E GELU(n) = sigma^2 / sqrt(2 pi (1 + sigma^2))
    for sigma in (0.5, 1.0, 2.0):
        mo = table[("GELU", sigma)]
        P(f"GELU sigma={sigma}: E GELU(n) quad {mo['mean']:.10f}, closed sigma^2/sqrt(2 pi (1+sigma^2)) = {sigma**2/np.sqrt(2*np.pi*(1+sigma**2)):.10f}; E GELU' = {mo['Ephi1']:.10f} (closed 1/2)")
    P("")

    # Monte Carlo of in-step and power sums
    T = 20000          # noise draws
    chunk = 500
    for K, rho in ((8192, 8.0), (600, 8.0)):
        a = rho / np.sqrt(K)
        s = a * np.where(np.arange(K) % 2 == 0, 1.0, -1.0)       # alternating +-a
        P(f"Monte Carlo: K = {K} pieces, rho = {rho}, a = rho/sqrt(K) = {a:.4f}, sigma = 1, {T} noise draws")
        P(f"{'phi':36s} {'in-step: 1st-order r*rho':>24s} {'exact':>8s} {'MC':>16s}   {'power sum: 2nd-order':>20s} {'exact':>8s} {'MC':>16s}")
        for name in PHIS:
            sigma = 1.0
            phi, br = phi_for(name, sigma)
            mo = table[(name, sigma)]
            # exact means m(a), m(-a), m(0) by quadrature
            ma = gauss_expect(lambda x: phi(x + a), sigma, tuple(b - a for b in br))
            mma = gauss_expect(lambda x: phi(x - a), sigma, tuple(b + a for b in br))
            m0 = mo["mean"]
            exact_in = np.sqrt(K) * (ma - mma) / (2 * mo["std"])
            exact_pw = (K / 2) * ((ma - m0) + (mma - m0)) / (np.sqrt(K) * mo["std"])
            pred_in = mo["r"] * rho
            pred_pw = mo["r2"] * rho ** 2 / np.sqrt(2 * K)
            d_in, d_pw, o0_in, o0_pw = [], [], [], []
            for _ in range(T // chunk):
                n = rng.standard_normal((chunk, K))
                f0 = phi(n)
                f1 = phi(n + s)
                o0_in.append(f0 @ s / rho); d_in.append((f1 - f0) @ s / rho)   # weights c = s / rho (unit norm)
                o0_pw.append(f0.sum(1)); d_pw.append((f1 - f0).sum(1))
            d_in = np.concatenate(d_in); o0_in = np.concatenate(o0_in)
            d_pw = np.concatenate(d_pw); o0_pw = np.concatenate(o0_pw)
            def snr(d, o0):
                sd = o0.std(ddof=1)
                val = d.mean() / sd
                se = np.sqrt((d.std(ddof=1) / np.sqrt(len(d)) / sd) ** 2 + val ** 2 / (2 * len(d)))
                return val, se
            vi, si = snr(d_in, o0_in)
            vp, sp = snr(d_pw, o0_pw)
            P(f"{name:36s} {pred_in:24.4f} {exact_in:8.4f} {vi:9.4f} +- {si:.4f}   {pred_pw:20.4f} {exact_pw:8.4f} {vp:9.4f} +- {sp:.4f}")
        P(f"linear matched filter for reference: SNR {rho}; power sum of squares rho^2/sqrt(2K) = {rho**2/np.sqrt(2*K):.4f}")
        P("")
    open("out/a_nonlinearity.txt", "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
