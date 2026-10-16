"""Receptive field of one-layer long-convolution networks (free, time-stored, S4D)."""

import json
import math
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
from rf_common import FS, N, nonzero_interval, shortest_span, whitened_noise  # noqa: E402
from train.experimental.long_conv.arch import LongConvEnergyNet  # noqa: E402

torch.set_num_threads(4)
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/"
CK = "/CLAUDE-TESTS/{0}/checkpoints/{1}"
QS = (0.5, 0.9, 0.99)
SEEDS, N_FEAT = 20, 8

MODELS = {
    "free (spectrum)": (dict(kernel="free"), "free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96", "ck_248.ckpt"),
    "free (time-stored)": (dict(kernel="time"), "free_kernel_taps_energy_max_256ch_softmax_train_snr8_prob0.96", "ck_133.ckpt"),
    "S4D d_state 64 lin": (dict(kernel="s4d", d_state=64, s4d_init="lin"), "s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96", "ck_266.ckpt"),
    "S4D d_state 512 lin": (dict(kernel="s4d", d_state=512, s4d_init="lin"), "s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96", "ck_288.ckpt"),
    "S4D d_state 512 band": (dict(kernel="s4d", d_state=512, s4d_init="band", memory=4.0), "s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96", "ck_122.ckpt"),
}


def build(cfg, seed, state=None):
    torch.manual_seed(seed)
    m = LongConvEnergyNet(num_ifos=2, channels=256, decimate=2, dt_min=1e-4, dt_max=10.0, highpass=20.0, **cfg)
    if state is not None:
        m.load_state_dict(state)
    return m.double().eval()


def load(run, ck):
    c = torch.load(R + run + CK.format(run, ck), map_location="cpu", weights_only=False)
    return {k[len("model."):]: v for k, v in c["state_dict"].items() if k.startswith("model.")}, c["epoch"]


@torch.no_grad()
def kernels(m, chunk=8):
    """``m.kernels()``, with the S4D kernel built a few channels at a time."""
    if m.kernel_type != "s4d":
        return m.kernels()
    K = m.kernel
    out = []
    for a in range(0, K.log_dt.shape[0], chunk):
        s = slice(a, a + chunk)
        dt = torch.exp(K.log_dt[s])
        C = torch.view_as_complex(K.C[s])
        A = torch.complex(-torch.exp(K.log_A_real[s]), K.A_imag[s])
        dtA = A * dt.unsqueeze(-1)
        KK = dtA.unsqueeze(-1) * torch.arange(m.length)
        C = C * (torch.exp(dtA) - 1.0) / A
        out.append(2 * torch.einsum("hn, hnl -> hl", C, torch.exp(KK)))
    k = torch.cat(out)
    if m.mix_from:
        k = torch.view_as_complex(m.mix) @ k
    return k / k.abs().pow(2).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)


def energy(m, X, k, t):
    """Energy 2|y|^2 of detector 0 at 1024 Hz index ``t`` for kernels ``k`` (B, L),
    copying ``LongConvEnergyNet.peak_energy``."""
    L = m.length
    spectrum = torch.fft.rfft(X, n=L * m.decimate)[..., : m.f_keep]
    x = torch.fft.irfft(spectrum, n=L) / math.sqrt(m.decimate)
    n = 2 * L
    y = torch.fft.ifft(torch.fft.fft(x[:, 0], n=n) * torch.fft.fft(k, n=n), n=n)
    return 2 * y[:, t].abs().pow(2)


def one_sided_lag(e, q):
    """Smallest lag S (samples at 1024 Hz) with fraction q of kernel energy in lags [0, S)."""
    c = np.cumsum(e, -1) / e.sum(-1, keepdims=True)
    return (c < q).sum(-1) + 1


def check_replica(m):
    """The copied energy must match the module's own energy for all channels."""
    gen = torch.Generator().manual_seed(7)
    X = whitened_noise(gen)
    with torch.no_grad():
        L = m.length
        spectrum = torch.fft.rfft(X.float(), n=L * m.decimate)[..., : m.f_keep]
        x = torch.fft.irfft(spectrum, n=L) / math.sqrt(m.decimate)
        kf = m.float().kernels() if m.kernel_type != "s4d" or m.kernel.C.shape[1] <= 32 else None
        m.double()
        k = kernels(m)
        if kf is not None:
            y = torch.fft.ifft(torch.fft.fft(x, n=2 * L)[:, :, None] * torch.fft.fft(kf, n=2 * L), n=2 * L)
            ref = 2 * y[0, 0, :, L - 1].abs().pow(2).double()
            mine = energy(m, X.repeat(k.shape[0], 1, 1), k, L - 1)
            return float(((mine - ref).abs() / ref.abs().clamp_min(1e-9)).max())
    return None


def grad_maps(m, k, t, seed):
    gen = torch.Generator().manual_seed(3000 + seed)
    X0 = whitened_noise(torch.Generator().manual_seed(1000 + seed))
    chans = torch.randperm(k.shape[0], generator=gen)[:N_FEAT]
    X = X0.repeat(N_FEAT, 1, 1).clone().requires_grad_(True)
    energy(m, X, k[chans], t).sum().backward()
    return X.grad.pow(2).sum(1).numpy()


results = {}
for name, (cfg, run, ck) in MODELS.items():
    for tag in ("init", "trained"):
        if tag == "init":
            m, epoch = build(cfg, 0), None
        else:
            state, epoch = load(run, ck)
            m = build(cfg, 0, state)
        rel = check_replica(m) if tag == "trained" else None
        k = kernels(m)                                   # (256, 4096) complex, unit norm
        e = k.abs().pow(2).numpy()                       # kernel energy over lag (1024 Hz samples)
        lag = {q: one_sided_lag(e, q) for q in QS}       # per channel
        prof = e.mean(0)
        lag_prof = {q: int(one_sided_lag(prof[None], q)[0]) for q in QS}
        short_ch = {q: float(np.median([shortest_span(ei, q) for ei in e])) for q in QS}
        out = {"epoch": epoch, "replica_max_rel_err": rel}
        out["kernel_lag_median_over_channels_s"] = {q: float(np.median(lag[q])) / 1024 for q in QS}
        out["kernel_lag_quartiles_s"] = {q: [float(np.percentile(lag[q], 25)) / 1024, float(np.percentile(lag[q], 75)) / 1024] for q in QS}
        out["kernel_lag_channel_mean_profile_s"] = {q: lag_prof[q] / 1024 for q in QS}
        out["kernel_shortest_window_median_s"] = {q: short_ch[q] / 1024 for q in QS}
        for t_name, t in (("end", m.length - 1), ("center", m.length // 2)):
            g2 = np.concatenate([grad_maps(m, k, t, s) for s in range(SEEDS)])
            nzi = [nonzero_interval(g) for g in g2]
            P = g2.sum(1, keepdims=True)
            avg = (g2 / P).mean(0)
            out[f"grad_{t_name}"] = {
                "t_input_index": 2 * t,
                "maps": len(g2),
                "nonzero_span_min_max": [min(c for _, _, c in nzi), max(c for _, _, c in nzi)],
                "nonzero_lo_hi": [min(a for a, _, _ in nzi), max(b for _, b, _ in nzi)],
                "mass_after_output_time": float((g2[:, 2 * t + 2:].sum(1) / P[:, 0]).mean()),
                "avg_shortest_samples": {q: shortest_span(avg, q) for q in QS},
                "per_map_shortest_mean_samples": {q: float(np.mean([shortest_span(g, q) for g in g2])) for q in QS},
            }
        results[f"{name} / {tag}"] = out
        print(f"\n== {name} / {tag}" + (f" ({ck}, epoch {epoch})" if epoch is not None else "")
              + (f"; replica max rel err {rel:.2e}" if rel is not None else ""))
        print("  kernel energy over lag, one-sided from the output time, median channel [quartiles] (s): "
              + ", ".join(f"{int(q * 100)}% {out['kernel_lag_median_over_channels_s'][q]:.3f} "
                          f"[{out['kernel_lag_quartiles_s'][q][0]:.3f}, {out['kernel_lag_quartiles_s'][q][1]:.3f}]" for q in QS))
        print("  kernel energy, channel-averaged profile (s): "
              + ", ".join(f"{int(q * 100)}% {out['kernel_lag_channel_mean_profile_s'][q]:.3f}" for q in QS)
              + "; shortest window, median channel (s): "
              + ", ".join(f"{int(q * 100)}% {out['kernel_shortest_window_median_s'][q]:.3f}" for q in QS))
        for t_name in ("end", "center"):
            gq = out[f"grad_{t_name}"]
            print(f"  gradient, energy at {t_name} (input index {gq['t_input_index']}), {gq['maps']} maps: nonzero span "
                  f"{gq['nonzero_span_min_max']} lo/hi {gq['nonzero_lo_hi']}; mass after output time {gq['mass_after_output_time']:.2e}; "
                  "averaged map shortest window: "
                  + ", ".join(f"{int(q * 100)}% {gq['avg_shortest_samples'][q]} = {gq['avg_shortest_samples'][q] / FS:.3f} s" for q in QS))
        del m, k, e

with open("rf_longconv.json", "w") as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved rf_longconv.json")
