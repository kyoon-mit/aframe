"""S4D-Lin tone statistics at initialization, from ml4gw's S4DKernel (used by aframe).

Run on CPU with the aframe train venv:
  projects/train/.venv/bin/python s4d_lin_tone_stats.py
Writes s4d_lin_tone_stats.json next to this file.
"""

import json
import math
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ml4gw.nn.ssm.s4d import S4DKernel  # noqa: E402

torch.set_num_threads(1)

F_LOW = 20.0
MEMS = (1.0, 4.0)
FS = (1024.0, 2048.0)
D_STATES = (64, 512)
DT_RANGES = {
    "den-reg [1e-4,10]": (1e-4, 10.0),
    "S4D default [1e-3,1e-1]": (1e-3, 1e-1),
}
H_TOTAL = 2**18
CHUNK = 2**14
OUT = os.path.dirname(os.path.abspath(__file__))


def fold(theta):
    """Phase per sample folded to [0, pi]."""
    return abs(((theta + math.pi) % (2 * math.pi)) - math.pi)


def code_check(N, dt_min, dt_max, H=2**16, seed=0):
    """Init tensors of S4DKernel against S4D-Lin and log-uniform dt."""
    torch.manual_seed(seed)
    k = S4DKernel(d_model=H, N=N, dt_min=dt_min, dt_max=dt_max)
    M = N // 2
    a_im = k.A_imag.detach().double().numpy()
    a_re = -np.exp(k.log_A_real.detach().double().numpy())
    n = np.arange(M)
    u = (k.log_dt.detach().double().numpy() - math.log(dt_min)) / math.log(
        dt_max / dt_min
    )
    us = np.sort(u)
    i = np.arange(H)
    ks = max(np.max((i + 1) / H - us), np.max(us - i / H))
    return {
        "N": N,
        "dt_range": [dt_min, dt_max],
        "modes_per_channel": int(a_im.shape[1]),
        "C_shape": list(k.C.shape),
        "max_abs_ImA_minus_pi_n": float(np.abs(a_im - np.pi * n).max()),
        "max_abs_ReA_plus_0.5": float(np.abs(a_re + 0.5).max()),
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "u_var": float(u.var()),
        "u_var_uniform": 1 / 12,
        "ks_D_vs_uniform": float(ks),
        "ks_D_5pct_critical": 1.358 / math.sqrt(H),
    }


def discretization_check(L=48):
    """Recover the discrete pole from S4DKernel.forward with one-hot C."""
    cases = [(64, 0.01, 10), (64, 0.05, 31), (512, 0.002, 200),
             (512, 1e-4, 255), (512, 3e-3, 255), (512, 0.7, 3)]
    out = []
    for N, dt, m in cases:
        k = S4DKernel(d_model=1, N=N).double()
        with torch.no_grad():
            k.log_dt.fill_(math.log(dt))
            k.C.zero_()
            k.C[0, m, 0] = 1.0
            K = k(L=L)[0].numpy()
        # real kernel 2 Re(c z^l) obeys K[l+2] = 2 Re(z) K[l+1] - |z|^2 K[l]
        X = np.stack([K[1:-1], K[:-2]], 1)
        a, b = np.linalg.lstsq(X, K[2:], rcond=None)[0]
        r = math.sqrt(-b)
        ang = math.acos(max(-1.0, min(1.0, a / (2 * r))))
        theta = math.pi * m * dt
        out.append({
            "N": N, "dt": dt, "mode": m,
            "abs_z_fit": r, "abs_z_expected_exp(-dt/2)": math.exp(-dt / 2),
            "arg_z_fit": ang, "arg_z_expected_fold(pi*m*dt)": fold(theta),
            "aliased": theta >= math.pi,
        })
    return out


def plog(a, b, lo, hi):
    """P(a <= dt < b) for dt log-uniform on [lo, hi]."""
    a, b = max(a, lo), min(b, hi)
    return math.log(b / a) / math.log(hi / lo) if b > a else 0.0


def merge(intervals):
    """Union of [a, b) intervals as a sorted list of disjoint [a, b)."""
    merged = []
    for a, b in sorted((a, b) for a, b in intervals if b > a):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def union_plog(intervals, lo, hi):
    """P(dt in union of [a, b)) for dt log-uniform on [lo, hi]."""
    clipped = [(max(a, lo), min(b, hi)) for a, b in intervals]
    return sum(math.log(b / a) for a, b in merge(clipped)) / math.log(hi / lo)


def exact(fs, N, lo, hi):
    """Closed-form expectations over log-uniform dt, averaged over modes n."""
    M = N // 2
    inb, below, alias = [], [], []
    both = {T: [] for T in MEMS}
    for n in range(M):
        if n == 0:  # f = 0: always below 20 Hz
            inb.append(0.0), below.append(1.0), alias.append(0.0)
            for T in MEMS:
                both[T].append(0.0)
            continue
        a = 2 * F_LOW / (n * fs)  # f_n >= 20 Hz  <=> dt >= 40 / (n fs)
        b = 1.0 / n  # unaliased  <=> dt < 1 / n
        inb.append(plog(a, b, lo, hi))
        below.append(plog(lo, min(a, b), lo, hi))
        alias.append(plog(b, hi, lo, hi))
        for T in MEMS:
            both[T].append(plog(a, min(b, 2 / (T * fs)), lo, hi))
    res = {
        "in_band": float(np.mean(inb)),
        "below_20Hz": float(np.mean(below)),
        "aliased": float(np.mean(alias)),
    }
    for T in MEMS:
        res[f"mem>{T:g}s"] = plog(lo, 2 / (T * fs), lo, hi)
        res[f"in_band&mem>{T:g}s"] = float(np.mean(both[T]))
        res[f"tones_per_ch_in_band&mem>{T:g}s"] = float(np.sum(both[T]))
        ivs = [(2 * F_LOW / (n * fs), min(1.0 / n, 2 / (T * fs))) for n in range(1, M)]
        res[f"channels_any_in_band&mem>{T:g}s"] = union_plog(ivs, lo, hi)
        res[f"dt_window_any_in_band&mem>{T:g}s"] = merge(ivs)  # before clipping
    return res


def mc(fs, N, lo, hi, seed):
    """Monte Carlo over channels using S4DKernel's own init tensors."""
    torch.manual_seed(seed)
    keys = ["in_band", "below_20Hz", "aliased"]
    keys += [f"mem>{T:g}s" for T in MEMS] + [f"in_band&mem>{T:g}s" for T in MEMS]
    per_ch = {key: [] for key in keys}
    cnt = {T: [] for T in MEMS}
    for _ in range(H_TOTAL // CHUNK):
        k = S4DKernel(d_model=CHUNK, N=N, dt_min=lo, dt_max=hi)
        with torch.no_grad():
            dt = torch.exp(k.log_dt).double()
            A = torch.complex(-torch.exp(k.log_A_real.double()), k.A_imag.double())
            dtA = (A * dt.unsqueeze(-1)).numpy()  # as in S4DKernel.forward
        theta = dtA.imag  # phase per sample, unwrapped
        f = theta * fs / (2 * math.pi)
        tau = 1.0 / (-dtA.real * fs)  # amplitude e-fold time, s
        unal = theta < math.pi
        inband = unal & (f >= F_LOW)
        per_ch["in_band"].append(inband.mean(1))
        per_ch["below_20Hz"].append((unal & (f < F_LOW)).mean(1))
        per_ch["aliased"].append((~unal).mean(1))
        for T in MEMS:
            mem = tau > T
            per_ch[f"mem>{T:g}s"].append(mem.mean(1))
            both = inband & mem
            per_ch[f"in_band&mem>{T:g}s"].append(both.mean(1))
            cnt[T].append(both.sum(1))
        del k
    res, se = {}, {}
    for key, v in per_ch.items():
        v = np.concatenate(v)
        res[key] = float(v.mean())
        se[key] = float(v.std(ddof=1) / math.sqrt(len(v)))
    for T in MEMS:
        c = np.concatenate(cnt[T])
        res[f"tones_per_ch_in_band&mem>{T:g}s"] = float(c.mean())
        res[f"channels_any_in_band&mem>{T:g}s"] = float((c > 0).mean())
        se[f"channels_any_in_band&mem>{T:g}s"] = float(
            (c > 0).std(ddof=1) / math.sqrt(len(c))
        )
    return res, se


def closed_form_summary():
    rows = []
    for fs in FS:
        for N in D_STATES:
            M = N // 2
            for T in MEMS:
                ok = M - 1 > F_LOW * T
                rows.append({
                    "fs": fs, "d_state": N, "tones": M, "T": T,
                    "f_max_with_tau>=T_Hz": (M - 1) / T,
                    "dt_window": [2 * F_LOW / ((M - 1) * fs), 2 / (T * fs)] if ok else None,
                    "tau_window_s": [T, (M - 1) / F_LOW] if ok else None,
                })
        for T in MEMS:
            rows.append({
                "fs": fs, "T": T, "min_tones_cover_to_nyquist": fs * T / 2,
                "d_state": fs * T, "tones_below_20Hz": F_LOW * T,
                "tones_in_band": T * (fs / 2 - F_LOW), "dt": 2 / (fs * T),
            })
    return rows


def main():
    t0 = time.time()
    out = {"code_check": [], "discretization_check": discretization_check(),
           "configs": [], "closed_form": closed_form_summary()}
    for N in D_STATES:
        for lo, hi in DT_RANGES.values():
            out["code_check"].append(code_check(N, lo, hi))
    seed = 0
    for fs in FS:
        for N in D_STATES:
            for name, (lo, hi) in DT_RANGES.items():
                seed += 1
                res, se = mc(fs, N, lo, hi, seed)
                ex = exact(fs, N, lo, hi)
                out["configs"].append({"fs": fs, "d_state": N, "tones": N // 2,
                                       "dt": name, "mc": res, "mc_se": se,
                                       "exact": ex, "channels": H_TOTAL})
    out["seconds"] = time.time() - t0
    with open(os.path.join(OUT, "s4d_lin_tone_stats.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    print("== code check (S4DKernel init) ==")
    for c in out["code_check"]:
        print(c)
    print("== discretization check (forward, one-hot C, 2nd-order fit) ==")
    for c in out["discretization_check"]:
        print(c)
    cols = ["in_band", "mem>1s", "mem>4s", "in_band&mem>1s", "in_band&mem>4s",
            "aliased", "below_20Hz", "channels_any_in_band&mem>1s",
            "channels_any_in_band&mem>4s"]
    print("== MC percent of tones (channels for channels_any) ==")
    print("fs d_state tones dt | " + " | ".join(cols)
          + " | tones/ch in&>1s | tones/ch in&>4s")
    worst = 0.0
    for c in out["configs"]:
        vals = [100 * c["mc"][k] for k in cols]
        worst = max(worst, max(abs(c["mc"][k] - c["exact"][k]) for k in cols))
        print(f"{c['fs']:.0f} {c['d_state']} {c['tones']} {c['dt']} | "
              + " | ".join(f"{v:.2f}" for v in vals)
              + f" | {c['mc']['tones_per_ch_in_band&mem>1s']:.3f}"
              + f" | {c['mc']['tones_per_ch_in_band&mem>4s']:.3f}")
    print("== exact (closed form) ==")
    for c in out["configs"]:
        vals = [100 * c["exact"][k] for k in cols]
        print(f"{c['fs']:.0f} {c['d_state']} {c['tones']} {c['dt']} | "
              + " | ".join(f"{v:.2f}" for v in vals)
              + f" | {c['exact']['tones_per_ch_in_band&mem>1s']:.3f}"
              + f" | {c['exact']['tones_per_ch_in_band&mem>4s']:.3f}")
    print(f"max |MC - exact| over table cells: {100 * worst:.3f} pp")
    print("max MC SE:", max(max(c["mc_se"].values()) for c in out["configs"]))
    print("== closed form ==")
    for r in out["closed_form"]:
        print(r)
    print(f"seconds {out['seconds']:.1f}")


if __name__ == "__main__":
    main()
