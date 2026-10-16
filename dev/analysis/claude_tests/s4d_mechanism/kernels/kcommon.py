"""Shared helpers: chirp formulas, model loading, templates, kernels, matches."""
import math
import os

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CK = os.path.join(HERE, "ckpt")
CFG = "/home/kyoon/SSM-BNS/aframe/dev/local/train/claude_tests"
BANK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin0_q0.8.npz"
TSUN = 4.925490947e-6          # G Msun / c^3, s
MERGE = 7680                   # merger index of the 4 s window at 2048 Hz (bank convention, 3.75 s)

NETS = {
    "noisy_s4d_d64": dict(cfg="long_conv/s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96.yaml",
                          ck="noisy_s4d_d64_ck_266.ckpt", kind="long_conv"),
    "noisy_s4d_d512": dict(cfg="long_conv/s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96.yaml",
                           ck="noisy_s4d_d512_ck_288.ckpt", kind="long_conv"),
    "noisy_s4d_d512_band": dict(cfg="long_conv/s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96.yaml",
                                ck="noisy_s4d_d512band_ck_122.ckpt", kind="long_conv"),
    "noisy_free": dict(cfg="long_conv/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96.yaml",
                       ck="noisy_free_ck_248.ckpt", kind="long_conv"),
    "clean_s4d_layer1": dict(cfg="oracle/sig_only_reg_s4d_max_dropout0_64x32x4.yaml",
                             ck="clean_s4d_ck_69.ckpt", kind="oracle"),
    "clean_free": dict(cfg="oracle/sig_only_reg_free_kernel.yaml",
                       ck="clean_free_ck_46.ckpt", kind="oracle"),
}


# ---------------------------------------------------------------- chirp (Newtonian)
def tau_of_f(f, mc):
    """Time before merger at GW frequency f (s), leading order."""
    return 5 / 256 * (TSUN * mc) ** (-5 / 3) * (np.pi * f) ** (-8 / 3)


def f_of_tau(tau, mc):
    """GW frequency a time tau before merger (Hz), leading order."""
    return (256 * tau / 5) ** (-3 / 8) * (TSUN * mc) ** (-5 / 8) / np.pi


def fdot(f, mc):
    """df/dt at frequency f (Hz/s), leading order."""
    return 96 / 5 * np.pi ** (8 / 3) * (TSUN * mc) ** (5 / 3) * f ** (11 / 3)


def t_step(f, mc):
    """In-step time fdot^(-1/2) (s)."""
    return fdot(f, mc) ** -0.5


# ---------------------------------------------------------------- models
def load_cfg(name):
    with open(os.path.join(CFG, NETS[name]["cfg"])) as fh:
        return yaml.safe_load(fh)


def load_state(name):
    c = torch.load(os.path.join(CK, NETS[name]["ck"]), map_location="cpu", weights_only=False)
    sd = {k[len("model."):]: v for k, v in c["state_dict"].items() if k.startswith("model.")}
    return sd, c


def build(name):
    """Arch module from the yaml (class_path, init_args), state loaded, eval, float64."""
    import importlib
    cfg = load_cfg(name)
    arch = cfg["model"]["init_args"]["arch"]
    mod, cls = arch["class_path"].rsplit(".", 1)
    Cls = getattr(importlib.import_module(mod), cls)
    torch.manual_seed(0)
    m = Cls(**arch["init_args"])
    sd, c = load_state(name)
    missing, unexpected = m.load_state_dict(sd, strict=True), None
    return m.double().eval(), c, cfg


# ---------------------------------------------------------------- S4D kernels
@torch.no_grad()
def s4d_modes(kernel, fs):
    """Per channel and mode: discrete pole z, coefficient c (kernel = sum c z^l, before the
    real part for the real S4DKernel), frequency (Hz, from angle z), ring time 1/(a dt fs) (s),
    aliasing flag |b dt| > pi, and dt."""
    dt = torch.exp(kernel.log_dt.double())[:, None]
    a = torch.exp(kernel.log_A_real.double())
    b = kernel.A_imag.double()
    A = torch.complex(-a, b)
    dtA = A * dt
    z = torch.exp(dtA)
    C = torch.view_as_complex(kernel.C.double())
    c = 2 * C * (torch.exp(dtA) - 1.0) / A
    f = torch.angle(z) * fs / (2 * math.pi)
    ring = 1.0 / (a * dt * fs)
    alias = (b * dt).abs() > math.pi
    return dict(z=z.numpy(), c=c.numpy(), f=f.numpy(), ring=ring.numpy(), alias=alias.numpy(),
                dt=dt[:, 0].numpy(), a=a.numpy(), b=b.numpy())


def kernel_from_modes(z, c, L, chunk=8):
    """(H, L) complex kernel sum_n c_n z_n^l, built a few channels at a time."""
    H = z.shape[0]
    out = np.empty((H, L), np.complex128)
    lz = np.log(z.astype(np.complex128))
    l = np.arange(L)
    for s in range(0, H, chunk):
        e = np.exp(lz[s:s + chunk, :, None] * l)               # (h, n, L)
        out[s:s + chunk] = np.einsum("hn,hnl->hl", c[s:s + chunk], e)
    return out


def mode_energy(z, c, L):
    """Energy over L lags of each complex mode c z^l."""
    r2 = np.abs(z) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(np.abs(1 - r2) > 1e-14, (1 - r2 ** L) / (1 - r2), L)
    return np.abs(c) ** 2 * g


# ---------------------------------------------------------------- templates
def templates(rate=2048, ifo=0):
    """Whitened bank templates (J, 4 s) at ``rate`` (2048 or 1024 via the long_conv
    decimation), detector ``ifo`` (0 H1, 1 L1); chirp masses (J,)."""
    z = np.load(os.path.join(HERE, "templates_2048.npz"))
    x = torch.tensor(z["white"][:, ifo])
    if rate == 1024:
        L = x.shape[-1] // 2
        spec = torch.fft.rfft(x, n=2 * L)[..., : L // 2 + 1]
        x = torch.fft.irfft(spec, n=L) / math.sqrt(2)
    return x.numpy(), z["chirp_mass"]


# ---------------------------------------------------------------- matches
@torch.no_grad()
def peak_energy_ratio(k, x, jchunk=4, cchunk=256, device="cpu", analytic=False, fband=None, fs=None,
                      xnorm2=None):
    """max_t 2|(k * x)(t)|^2 / ||x||^2 for unit-norm kernels k (C, L) complex and real
    inputs x (J, L), output times 0..L-1 as in LongConvEnergyNet; (J, C) and argmax t.
    analytic=True (real kernels): max_t |(k * x_a)(t)|^2 / ||x||^2 with x_a the analytic
    signal of x on the 2L grid, the squared match maximized over the template's phase.
    fband=(lo, hi), fs: with analytic=True, keep only template frequencies lo <= f < hi.
    xnorm2: (J,) normalization in place of ||x||^2."""
    k = torch.as_tensor(k, dtype=torch.complex128, device=device)
    x = torch.as_tensor(x, dtype=torch.float64, device=device)
    J, L = x.shape
    C = k.shape[0]
    n = 2 * L
    kf = torch.fft.fft(k, n=n).to(torch.complex64)
    xf = torch.fft.fft(x, n=n)
    gain = 2.0
    if analytic:
        w = torch.zeros(n, dtype=torch.float64, device=device)
        w[0] = 1.0
        w[1:L] = 2.0
        w[L] = 1.0
        if fband is not None:
            fr = torch.fft.fftfreq(n, 1 / fs).to(device)
            w = w * ((fr >= fband[0]) & (fr < fband[1]))
        xf = xf * w
        gain = 1.0
    xf = xf.to(torch.complex64)
    rho2 = x.pow(2).sum(-1) if xnorm2 is None else torch.as_tensor(xnorm2, dtype=torch.float64, device=device)
    out = torch.empty(J, C, dtype=torch.float64, device=device)
    arg = torch.empty(J, C, dtype=torch.long, device=device)
    for j in range(0, J, jchunk):
        for c in range(0, C, cchunk):
            y = torch.fft.ifft(xf[j:j + jchunk, None] * kf[None, c:c + cchunk], n=n)[..., :L]
            e = gain * y.abs().pow(2)
            v, t = e.max(-1)
            out[j:j + jchunk, c:c + cchunk] = v.double() / rho2[j:j + jchunk, None]
            arg[j:j + jchunk, c:c + cchunk] = t
    return out.cpu().numpy(), arg.cpu().numpy()


def analytic_part(k_real):
    """Positive-frequency part of real kernels, times 2 (the analytic signal), unit norm."""
    L = k_real.shape[-1]
    n = 2 * L
    F = np.fft.fft(k_real, n=n, axis=-1)
    w = np.zeros(n)
    w[0] = 1.0
    w[1:n // 2] = 2.0
    w[n // 2] = 1.0
    ka = np.fft.ifft(F * w, axis=-1)[..., :L]
    return ka / np.linalg.norm(ka, axis=-1, keepdims=True)


def transfer(k, fs, nfft):
    """Transfer function H(f) on the full fft grid and group delay (s), for kernels (C, L)."""
    L = k.shape[-1]
    l = np.arange(L)
    H = np.fft.fft(k, n=nfft, axis=-1)
    Hl = np.fft.fft(k * l, n=nfft, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        gd = np.real(Hl / H) / fs
    f = np.fft.fftfreq(nfft, 1 / fs)
    return f, H, gd


def band_norm2(x, fs, lo, hi):
    """||x_band||^2 of real x (J, L) restricted to lo <= |f| < hi, on the 2L grid."""
    L = x.shape[-1]
    X = np.fft.fft(x, n=2 * L, axis=-1)
    fa = np.abs(np.fft.fftfreq(2 * L, 1 / fs))
    return (np.abs(X[:, (fa >= lo) & (fa < hi)]) ** 2).sum(-1) / (2 * L)
