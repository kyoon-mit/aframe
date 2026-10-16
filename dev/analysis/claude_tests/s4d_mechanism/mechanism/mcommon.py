"""Shared code for the mechanism tests: data, manipulations, in-step match, estimators."""
import json
import math
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
HARNESS = os.path.join(ROOT, "harness")
KERNELS = os.path.join(ROOT, "kernels")
sys.path.insert(0, HARNESS)
import harness  # noqa: E402

torch.set_num_threads(8)
PLAN = json.load(open(os.path.join(HERE, "plan.json")))
FS = 2048
L = 8192
SEG = 1024
NSEG = 8
RAMP = 64
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NETS = {
    "s4d_ck69": ("sig_only_reg_s4d_max_dropout0_64x32x4", "ck_69"),
    "s4d_ck81": ("sig_only_reg_s4d_max_dropout0_64x32x4", "ck_81"),
    "free_ck46": ("sig_only_reg_free_kernel", "ck_46"),
    "resnet_ck37": ("sig_only_reg_resnet", "ck_37"),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def load_clean(n=None):
    labels, paths = harness.load_set("clean")
    X = harness.read_rows(harness.DATA_DIR / "clean_X_f32.npy", 0, n)
    n = len(X)
    return X, labels["chirp_mass"][:n].astype(np.float64), labels["merger_sample"][:n].astype(np.int64), labels


# ---------------------------------------------------------------- tapers
def ramp(n):
    return 0.5 * (1 - np.cos(np.pi * (np.arange(n) + 0.5) / n))


def tukey(length, r):
    w = np.ones(length)
    w[:r] = ramp(r)
    w[length - r:] = ramp(r)[::-1]
    return w


TAPER = np.tile(tukey(SEG, RAMP), NSEG)          # (8192,)


# ---------------------------------------------------------------- signals
def analytic(x):
    """Analytic signal of real x (..., L) on a 2L zero-padded grid, cut to L (torch, complex)."""
    Ln = x.shape[-1]
    n = 2 * Ln
    w = torch.zeros(n, dtype=torch.float64, device=x.device)
    w[0] = 1.0
    w[1:Ln] = 2.0
    w[Ln] = 1.0
    return torch.fft.ifft(torch.fft.fft(x, n=n) * w, n=n)[..., :Ln]


def taper_only(x):
    return x * torch.as_tensor(TAPER, dtype=x.dtype, device=x.device)


def shuffle(x, perms):
    """x (B, 2, L) torch, perms (B, 8) int: output segment j = tapered input segment perms[b, j]."""
    B = x.shape[0]
    xt = taper_only(x).reshape(B, x.shape[1], NSEG, SEG)
    idx = torch.as_tensor(perms, device=x.device)[:, None, :, None].expand(-1, x.shape[1], -1, SEG)
    return torch.gather(xt, 2, idx).reshape(B, x.shape[1], L)


def phase(x, phis):
    """x (B, 2, L) torch, phis (B, 8): segment k = taper * Re(analytic * exp(i phi_k))."""
    xa = analytic(x.double())
    ph = torch.as_tensor(phis, dtype=torch.float64, device=x.device).repeat_interleave(SEG, dim=1)  # (B, L)
    y = (xa * torch.exp(1j * ph)[:, None, :]).real
    return taper_only(y).to(x.dtype)


@torch.no_grad()
def instep_match(y, x):
    """Per window: max over the two detectors of max over all lags
    |sum_t y(t) conj(x_a(t - lag))| / (||y|| ||x||). y, x (B, 2, L) torch."""
    y = y.double()
    x = x.double()
    n = 2 * L
    xa = analytic(x)
    r = torch.fft.ifft(torch.fft.fft(y, n=n) * torch.fft.fft(xa, n=n).conj(), n=n)
    m = r.abs().amax(-1) / (y.norm(dim=-1) * x.norm(dim=-1)).clamp_min(1e-30)
    return m.amax(-1), m                            # (B,), (B, 2)


# ---------------------------------------------------------------- template bank
class Bank:
    """Coherent template bank: 235 whitened templates on a 0.5% log chirp-mass grid."""

    def __init__(self, device=DEV):
        z = np.load(os.path.join(KERNELS, "templates_2048.npz"))
        T = torch.tensor(z["white"], dtype=torch.float64, device=device)          # (J, 2, L)
        self.mc = z["chirp_mass"].astype(np.float64)
        self.merge = int(z["merge_index"])
        self.norm2 = T.pow(2).sum(-1).float()                                      # (J, 2)
        self.TF = torch.fft.fft(analytic(T), n=2 * L).conj().to(torch.complex64)   # (J, 2, 2L)
        self.device = device
        self.J = T.shape[0]

    @torch.no_grad()
    def snr2(self, X, batch=16, want_lag=False):
        """(n, J) summed SNR^2 (each detector at its own best lag); optional (n, J) lag of the
        detector-summed |corr|^2 maximum (for T4)."""
        out, lags = [], []
        for i in range(0, len(X), batch):
            x = torch.as_tensor(np.asarray(X[i:i + batch]), device=self.device).float()
            XF = torch.fft.fft(x, n=2 * L)                                         # (b, 2, 2L)
            tot = 0
            comb = 0
            for d in range(2):
                r = torch.fft.ifft(XF[:, d, None, :] * self.TF[None, :, d, :], n=2 * L)   # (b, J, 2L)
                p = r.abs().pow(2) / self.norm2[None, :, d, None]
                tot = tot + p.amax(-1)
                if want_lag:
                    comb = comb + p
            out.append(tot.double().cpu())
            if want_lag:
                lags.append(comb.argmax(-1).cpu())
            del XF
        res = torch.cat(out).numpy()
        if want_lag:
            lag = torch.cat(lags).numpy()
            lag = np.where(lag >= L, lag - 2 * L, lag)          # circular index -> signed lag
            return res, lag
        return res

    def __call__(self, X):
        return self.mc[self.snr2(X).argmax(-1)]


def load_estimator(key):
    if key == "bank":
        return Bank()
    name, tag = NETS[key]
    return harness.load_net(name, tag, gpu_gb=3.0)


def within1(est, true):
    return np.abs(est - true) / true < 0.01


def boot_drop(ok_ref, ok_test, n_boot=2000, seed=0):
    """Paired bootstrap of the relative drop (ref - test) / ref of the within-1% fraction."""
    rng = np.random.default_rng(seed)
    n = len(ok_ref)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = ok_ref[idx].mean(1)
    t = ok_test[idx].mean(1)
    d = (r - t) / np.where(r > 0, r, np.nan)
    return [float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]


def boot_frac(ok, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ok), size=(n_boot, len(ok)))
    v = ok[idx].mean(1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
