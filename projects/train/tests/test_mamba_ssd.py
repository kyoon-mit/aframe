import torch

from train.experimental.mamba.arch import MambaDenoiseRegress
from train.experimental.mamba.ssd import Mamba2, ssd_scan


def recurrence(X, A, B, C):
    """state_t = exp(A_t) state_{t-1} + B_t x_t, y_t = C_t state_t, step by step,
    with B and C shared by the heads."""
    b, l, h, p = X.shape
    state = torch.zeros(b, h, p, B.shape[-1], dtype=X.dtype)
    ys = []
    for t in range(l):
        state = torch.exp(A[:, t])[..., None, None] * state + X[:, t, :, :, None] * B[:, t, None, None, :]
        ys.append((state * C[:, t, None, None, :]).sum(-1))
    return torch.stack(ys, 1)


def test_ssd_scan_matches_recurrence():
    torch.manual_seed(0)
    b, l, h, p, n = 2, 64, 3, 4, 5
    X = torch.randn(b, l, h, p, dtype=torch.float64)
    A = -torch.rand(b, l, h, dtype=torch.float64)
    B = torch.randn(b, l, n, dtype=torch.float64)
    C = torch.randn(b, l, n, dtype=torch.float64)
    expected = recurrence(X, A, B, C)
    for chunk in (1, 8, 16, 64):
        assert torch.allclose(ssd_scan(X, A, B, C, chunk), expected, atol=1e-10)


def test_mamba2_is_causal():
    torch.manual_seed(0)
    layer = Mamba2(16, d_state=8, headdim=8, chunk=8).double()
    u = torch.randn(2, 32, 16, dtype=torch.float64)
    v = u.clone()
    v[:, 20:] += torch.randn(2, 12, 16, dtype=torch.float64)
    assert torch.allclose(layer(u)[:, :20], layer(v)[:, :20], atol=1e-12)
    assert not torch.allclose(layer(u)[:, 20:], layer(v)[:, 20:])


def test_den_reg_shapes():
    net = MambaDenoiseRegress(num_ifos=2, d_output=4, denoiser_d_model=16, denoiser_d_state=8,
                              denoiser_n_layers=2, regressor_d_model=16, regressor_d_state=8,
                              regressor_n_layers=2, num_groups=4, headdim=8, chunk=16)
    x = torch.randn(3, 2, 128)
    denoised, y = net(x)
    assert denoised.shape == (3, 2, 128) and y.shape == (3, 4)
    (denoised.square().mean() + y.square().mean()).backward()
