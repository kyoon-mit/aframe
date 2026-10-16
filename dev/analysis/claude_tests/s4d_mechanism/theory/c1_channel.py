"""Section 1 checks: one S4D channel is a weighted bank of damped, driven
resonators, and its transfer function is a sum of Lorentzian resonances.

Run from /home/kyoon/SSM-BNS/aframe/projects/train with its .venv python.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from scipy import signal

from common import FS, RES

torch.set_num_threads(4)
from ml4gw.nn.ssm.s4d import S4D, S4DKernel  # noqa: E402

out = {}
rng = np.random.default_rng(1)

# ---- (a) ml4gw kernel = closed form sum of damped cosines, in physical units
torch.manual_seed(0)
H, N, L = 3, 8, 4096
ker = S4DKernel(H, N=N, dt_min=1e-3, dt_max=1e-1)
with torch.no_grad():
    ker.log_dt.copy_(torch.log(torch.tensor([0.004, 0.02, 0.05])))
    K_ml = ker(L).double().numpy()
    dt = torch.exp(ker.log_dt).double().numpy()
    C = torch.view_as_complex(ker.C).to(torch.complex128).numpy()
    A = (-torch.exp(ker.log_A_real.double()) + 1j * ker.A_imag.double()).numpy()
errs = []
for h in range(H):
    a_phys = -A[h].real * dt[h] * FS            # 1/s
    f_phys = A[h].imag * dt[h] * FS / (2 * np.pi)  # Hz
    Ct = C[h] * (np.exp(dt[h] * A[h]) - 1) / A[h]
    t = np.arange(L) / FS
    Kc = np.sum(2 * np.abs(Ct)[:, None] * np.exp(-a_phys[:, None] * t)
                * np.cos(2 * np.pi * f_phys[:, None] * t + np.angle(Ct)[:, None]), 0)
    errs.append(float(np.max(np.abs(Kc - K_ml[h])) / np.max(np.abs(K_ml[h]))))
out["a_kernel_closed_form_max_rel_err"] = max(errs)
print("(a) ml4gw kernel vs sum of damped cosines, max rel err", max(errs))

# ---- (b) layer output before GELU = causal convolution + D u, and the
# per-sample recurrence x_l = zbar x_{l-1} + Bbar u_l gives the same
layer = S4D(H, d_state=N)
with torch.no_grad():
    layer.kernel.load_state_dict(ker.state_dict())
u = rng.standard_normal((1, H, L))
ut = torch.tensor(u, dtype=torch.float32)
with torch.no_grad():
    k = layer.kernel(L=L)
    k_f = torch.fft.rfft(k, n=2 * L)
    u_f = torch.fft.rfft(ut, n=2 * L)
    y_layer = (torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L] + ut * layer.D.unsqueeze(-1)).double().numpy()[0]
    D = layer.D.double().numpy()
err_conv, err_rec = [], []
for h in range(H):
    y_conv = np.convolve(u[0, h], K_ml[h])[:L] + D[h] * u[0, h]
    err_conv.append(float(np.max(np.abs(y_conv - y_layer[h])) / np.max(np.abs(y_layer[h]))))
    zb = np.exp(dt[h] * A[h])
    Bb = (zb - 1) / A[h]
    x = np.zeros(N // 2, complex)
    y_rec = np.empty(L)
    for l in range(L):
        x = zb * x + Bb * u[0, h, l]
        y_rec[l] = 2 * np.real(np.sum(C[h] * x)) + D[h] * u[0, h, l]
    err_rec.append(float(np.max(np.abs(y_rec - y_layer[h])) / np.max(np.abs(y_layer[h]))))
out["b_layer_vs_convolution_max_rel_err_float32"] = max(err_conv)
out["b_layer_vs_recurrence_max_rel_err_float32"] = max(err_rec)
print("(b) layer vs causal convolution", max(err_conv), "vs recurrence", max(err_rec))

# ---- (c) each damped cosine is a driven damped oscillator read out as
# position and velocity: 2 Re(C e^{(-a+iw)t}) = 2 c_r q' + (2 c_r a - 2 c_i w) q,
# q = e^{-at} sin(wt)/w the oscillator's response to a unit kick.
a, w = 3.0, 2 * np.pi * 40.0
Cm = 0.7 - 0.4j
t = np.arange(0, 2.0, 1 / 20000.0)
q = np.exp(-a * t) * np.sin(w * t) / w
qd = np.exp(-a * t) * (np.cos(w * t) - (a / w) * np.sin(w * t))
lhs = 2 * np.real(Cm * np.exp((-a + 1j * w) * t))
rhs = 2 * Cm.real * qd + (2 * Cm.real * a - 2 * Cm.imag * w) * q
out["c_oscillator_identity_max_abs_err"] = float(np.max(np.abs(lhs - rhs)))
# driven oscillator q'' + 2a q' + (a^2+w^2) q = u, read out y = 2c_r q' + (...) q
w0sq = a * a + w * w
sys_ss = signal.StateSpace([[0, 1], [-w0sq, -2 * a]], [[0], [1]],
                           [[2 * Cm.imag * -w + 2 * Cm.real * a, 2 * Cm.real]], [[0]])
fs_c = 20000.0
tt = np.arange(0, 1.0, 1 / fs_c)
uu = np.exp(-((tt - 0.3) / 0.05) ** 2) * np.cos(2 * np.pi * 37 * tt)
_, y_osc, _ = signal.lsim(sys_ss, uu, tt)
kern = 2 * np.real(Cm * np.exp((-a + 1j * w) * tt)) / fs_c
y_cnv = np.convolve(uu, kern)[:tt.size] - 0.5 * kern[0] * uu  # trapezoid rule
out["c_driven_oscillator_vs_convolution_rel_err"] = float(np.max(np.abs(y_osc - y_cnv)) / np.max(np.abs(y_osc)))
print("(c) oscillator identity", out["c_oscillator_identity_max_abs_err"],
      "driven oscillator vs convolution", out["c_driven_oscillator_vs_convolution_rel_err"])

# ---- (d) transfer function: sum of Lorentzians; FWHM = a/pi Hz = 1/(pi T)
fs_d = 4096.0
Ld = int(fs_d * 40)
td = np.arange(Ld) / fs_d
modes = [(2.0, 50.0, 1.0 + 0.5j), (5.0, 120.0, -0.3 + 1.2j), (0.5, 200.0, 0.8 - 0.2j)]
Kd = sum(2 * np.real(c * np.exp((-aa + 2j * np.pi * ff) * td)) for aa, ff, c in modes)
# Continuous Fourier integral approximated by the DFT times 1/fs_d; compare
# with the analytic sum of pole terms (trapezoid correction: subtract K(0)/2).
Hd = np.fft.rfft(Kd) / fs_d - Kd[0] / (2 * fs_d)
fd = np.fft.rfftfreq(Ld, 1 / fs_d)
om = 2 * np.pi * fd
Ha = sum(c / (aa + 1j * (om - 2 * np.pi * ff)) + np.conj(c) / (aa + 1j * (om + 2 * np.pi * ff))
         for aa, ff, c in modes)
band = (fd > 10) & (fd < 400)
out["d_transfer_vs_pole_sum_max_rel_err"] = float(np.max(np.abs(Hd[band] - Ha[band])) / np.max(np.abs(Ha[band])))
fwhm = []
for aa, ff, c in modes:
    fl = ff + np.linspace(-20, 20, 400001) * aa / (2 * np.pi)
    ol = 2 * np.pi * fl
    one = c / (aa + 1j * (ol - 2 * np.pi * ff)) + np.conj(c) / (aa + 1j * (ol + 2 * np.pi * ff))
    p = np.abs(one) ** 2
    i0 = np.argmax(p)
    above = np.flatnonzero(p >= p[i0] / 2)
    fw = fl[above[-1]] - fl[above[0]]
    fwhm.append({"a_per_s": aa, "f_Hz": ff, "fwhm_numeric_Hz": float(fw),
                 "a_over_pi_Hz": aa / np.pi, "grid_Hz": float(fl[1] - fl[0])})
out["d_fwhm"] = fwhm
print("(d) transfer vs pole sum", out["d_transfer_vs_pole_sum_max_rel_err"])
for r in fwhm:
    print("    FWHM", r)

json.dump(out, open(os.path.join(RES, "c1_channel.json"), "w"), indent=1)
print("saved")
