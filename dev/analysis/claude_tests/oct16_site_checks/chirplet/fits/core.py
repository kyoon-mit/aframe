"""Chirplet model, variable-projection objective and kernel helpers.

K(l) = sum_k a_k exp(-(l - t_k)^2 / (2 s_k^2)) exp(i (2 pi f_k x / fs + pi c_k (x / fs)^2)),
x = l - t_k in samples, l = 0..L-1. t_k, s_k in samples, f_k in Hz, c_k in Hz/s.
Objective f = ||K - G a||^2 with a the least-squares amplitudes; for unit-norm K,
f = 1 - match^2.
"""

import os

_THR = os.environ.get("CHIRPLET_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = _THR

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(int(_THR))

FS = 2048.0
L = 8064
MCS = [1.0, 1.2, 1.5, 2.0]
HERE = Path(__file__).resolve().parent
S4D = Path("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/s4dmath/fits")
KERNELS = S4D / "kernels.npz"
LOG_S_MIN = np.log(2.0)
LOG_S_MAX = np.log(4.0 * L)
F_MAX = FS / 2
DEV = torch.device("cpu")


def load_kernels():
    ker = np.load(KERNELS)
    return {Mc: ker[f"K_{Mc}"] for Mc in MCS}


def unpack(theta, M):
    """theta = [t (M), log s (M), f (M), c (M)] -> four arrays."""
    return theta[:M], theta[M:2 * M], theta[2 * M:3 * M], theta[3 * M:]


def atoms(t, logs, f, c, lag, dtype=torch.complex128):
    """(L, M) chirplet matrix and the offsets x = l - t."""
    x = lag[:, None] - t[None, :]
    s = torch.exp(logs)
    re = -0.5 * (x / s[None, :]) ** 2
    im = 2 * np.pi * f[None, :] * x / FS + np.pi * c[None, :] * (x / FS) ** 2
    G = torch.exp(torch.complex(re, im))
    return G.to(dtype), x


class Objective:
    """Variable projection with exact gradient (amplitudes held at their optimum)."""

    def __init__(self, K, M):
        self.K = torch.tensor(K, dtype=torch.complex128, device=DEV)
        self.lag = torch.arange(L, dtype=torch.float64, device=DEV)
        self.M = M
        self.nfev = 0
        self.n_jitter = 0

    def solve(self, G):
        """Least-squares amplitudes via column-scaled Cholesky + 2 refinement steps."""
        B = G.conj().T @ G
        d = 1.0 / torch.sqrt(torch.clamp(B.diagonal().real, min=1e-300))
        Bs = d[:, None] * B * d[None, :]
        eye = torch.eye(B.shape[0], dtype=B.dtype, device=DEV)
        jit_used = 0.0
        for jit in (0.0, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6):
            Lc, info = torch.linalg.cholesky_ex(Bs + jit * eye if jit else Bs)
            if int(info) == 0:
                jit_used = jit
                break
        if jit_used:
            self.n_jitter += 1
        b = (G.conj().T @ self.K) * d
        a = torch.cholesky_solve(b[:, None], Lc)[:, 0] * d
        r = self.K - G @ a
        for _ in range(2):
            db = (G.conj().T @ r) * d
            a = a + torch.cholesky_solve(db[:, None], Lc)[:, 0] * d
            r = self.K - G @ a
        return a, r

    def value_grad(self, theta):
        """f and df/dtheta for theta = [t, log s, f, c] (numpy, float64)."""
        self.nfev += 1
        M = self.M
        th = torch.tensor(theta, dtype=torch.float64, device=DEV)
        t, logs, f, c = unpack(th, M)
        G, x = atoms(t, logs, f, c, self.lag)
        a, r = self.solve(G)
        fval = float((r.abs() ** 2).sum())
        W = r.conj()[:, None] * G * a[None, :]
        xc = x.to(torch.complex128)
        m0 = W.sum(0)
        m1 = (W * xc).sum(0)
        m2 = (W * xc * xc).sum(0)
        s2 = torch.exp(2 * logs)
        g_t = -2 * (m1 / s2 - 1j * (2 * np.pi * f / FS) * m0 - 1j * (2 * np.pi * c / FS**2) * m1).real
        g_ls = -2 * (m2 / s2).real
        g_f = -2 * (1j * 2 * np.pi * m1 / FS).real
        g_c = -2 * (1j * np.pi * m2 / FS**2).real
        grad = torch.cat([g_t, g_ls, g_f, g_c]).cpu().numpy()
        return fval, grad

    def details(self, theta):
        """Amplitudes, match (float64), cancellation ratio, Gram condition number."""
        M = self.M
        th = torch.tensor(theta, dtype=torch.float64, device=DEV)
        t, logs, f, c = unpack(th, M)
        G, _ = atoms(t, logs, f, c, self.lag)
        a, r = self.solve(G)
        Kh = G @ a
        match = float(torch.abs(torch.vdot(Kh, self.K)) / (torch.linalg.norm(Kh) * torch.linalg.norm(self.K)))
        col = torch.linalg.norm(G, dim=0)
        cancel = float(((a.abs() * col) ** 2).sum() / torch.linalg.norm(Kh) ** 2)
        sv = torch.linalg.svdvals(G / col[None, :])
        cond = float(sv[0] / sv[-1]) if float(sv[-1]) > 0 else float("inf")
        return {"a": a.cpu().numpy(), "match": match, "cancel_ratio": cancel, "cond_colnorm_G": cond,
                "col_norm": col.cpu().numpy(), "Khat": Kh.cpu().numpy()}


def match_float32(theta, a, K, M):
    """Match when atoms, amplitudes and the sum are evaluated in complex64."""
    th = torch.tensor(theta, dtype=torch.float32)
    t, logs, f, c = unpack(th, M)
    lag = torch.arange(L, dtype=torch.float32)
    x = lag[:, None] - t[None, :]
    s = torch.exp(logs)
    re = -0.5 * (x / s[None, :]) ** 2
    im = 2 * np.float32(np.pi) * f[None, :] * x / np.float32(FS) + np.float32(np.pi) * c[None, :] * (x / np.float32(FS)) ** 2
    G = torch.exp(torch.complex(re, im))
    Kh = (G @ torch.tensor(a, dtype=torch.complex64)).numpy().astype(np.complex128)
    return float(np.abs(np.vdot(Kh, K)) / (np.linalg.norm(Kh) * np.linalg.norm(K)))


def kernel_of(theta, a, M, n=L):
    """Fitted kernel (float64) on lags 0..n-1."""
    th = torch.tensor(theta, dtype=torch.float64)
    t, logs, f, c = unpack(th, M)
    lag = torch.arange(n, dtype=torch.float64)
    G, _ = atoms(t, logs, f, c, lag)
    return (G @ torch.tensor(a, dtype=torch.complex128)).numpy()


def bounds_for(M):
    """L-BFGS-B bounds on [t, log s, f, c]."""
    return ([(0.0, L - 1.0)] * M + [(LOG_S_MIN, LOG_S_MAX)] * M
            + [(-F_MAX, F_MAX)] * M + [(None, None)] * M)
