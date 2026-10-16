"""Does Theorem 1 (first order sees only the time sum) hold for the real,
zero-padded aframe ResNet when the chirp's merger sits near the window end?

For every logit o_j: first-order response at rho = 8,
    FO_j = 8 * E_n[(o_j(n + eps s) - o_j(n - eps s)) / (2 eps)] / sd(o_j(n)),
averaged over noise and over the merger position (uniform over 1024 whole-sample
shifts, which also averages the stride-32 comb out). Circular theory: |FO| <= 0.002.

Signals (unit norm, split equally over both detectors):
  full_end : whitened BNS (theorem's out/sig_whitened_bns.npy), merger 3.5 to 4.0 s
  short_end: its last 1 s before the peak (+0.1 s after), 0.1 s taper at the start, merger 3.5 to 4.0 s
  short_mid: the same short piece, merger 2.0 to 2.5 s (no window edge within 0.5 s)
Also the odd and even parts at A = 8 for full_end.
usage: v_edge.py {trained|init} NDRAWS OUT.json
"""
import json
import sys
import time

import numpy as np
import torch
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.nn.resnet.resnet_1d import ResNet1D

torch.set_num_threads(8)
L = 8192
SIG = "/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/cnnproof/out/sig_whitened_bns.npy"
CKPT = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/"
        "CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/checkpoints/ck_244.ckpt")
EPS, A = 1e-3, 8.0


def build(which):
    torch.manual_seed(0)
    net = ResNet1D(2, layers=[3, 4, 6, 3], classes=235, kernel_size=3, norm_layer=GroupNorm1DGetter(groups=16))
    if which == "trained":
        sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
        net.load_state_dict({k.split("model.net.", 1)[1]: v for k, v in sd.items() if k.startswith("model.net.")})
    return net.double().eval()


def shifted(s, d):
    out = np.zeros(L)
    if d >= 0:
        out[d:] = s[:L - d]
    else:
        out[:L + d] = s[-d:]
    return out / np.linalg.norm(out)


def main():
    which, n_draws, out_file = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    net = build(which)
    s = np.load(SIG)
    pk = int(np.argmax(np.abs(s)))                      # 7680
    short = np.zeros(L)
    a, b = pk - 2048, pk + 205
    short[a:b] = s[a:b]
    ramp = int(0.1 * 2048)
    short[a:a + ramp] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(ramp) / ramp)
    short /= np.linalg.norm(short)
    rng = np.random.default_rng(12345)
    f = np.fft.rfftfreq(L, 1 / 2048)
    keys = ["full_end", "short_end", "short_mid"]
    acc = {k: [] for k in keys}
    o0s, odd, even = [], [], []
    t0 = time.time()
    B = 4
    for i0 in range(0, n_draws, B):
        rows, meta = [], []
        for i in range(i0, min(n_draws, i0 + B)):
            n = rng.standard_normal((2, L))
            N = np.fft.rfft(n)
            N[:, f < 20] = 0
            n = np.fft.irfft(N, n=L)
            d = int(rng.integers(-512, 512))
            sig = {"full_end": shifted(s, d), "short_end": shifted(short, d), "short_mid": shifted(short, d - 3072)}
            two = {k: np.stack([v, v]) / np.sqrt(2) for k, v in sig.items()}
            rows.append(n)
            for k in keys:
                rows += [n + EPS * two[k], n - EPS * two[k]]
            rows += [n + A * two["full_end"], n - A * two["full_end"]]
            meta.append(len(rows))
        with torch.no_grad():
            o = net(torch.as_tensor(np.stack(rows))).numpy()
        per = 1 + 2 * len(keys) + 2
        for q in range(len(meta)):
            base = q * per
            o0 = o[base]
            o0s.append(o0)
            for m, k in enumerate(keys):
                acc[k].append((o[base + 1 + 2 * m] - o[base + 2 + 2 * m]) / (2 * EPS))
            op, om = o[base + 1 + 2 * len(keys)], o[base + 2 + 2 * len(keys)]
            odd.append((op - om) / 2)
            even.append((op + om - 2 * o0) / 2)
        if (i0 // B) % 10 == 0:
            print(f"{min(n_draws, i0 + B)}/{n_draws} {time.time() - t0:.0f}s", flush=True)
    o0s = np.array(o0s)
    sd = o0s.std(0, ddof=1)
    n = len(o0s)
    res = {"which": which, "n": n, "sd_logit_median": float(np.median(sd))}
    jtrue = int(round((np.log(1.2188) - np.log(0.85)) / 0.005))
    for k in keys:
        g = np.array(acc[k])
        fo = A * g.mean(0) / sd
        se = A * g.std(0, ddof=1) / np.sqrt(n) / sd
        z = fo / se
        jm = int(np.argmax(np.abs(fo)))
        res[k] = dict(max_abs_FO=float(np.abs(fo).max()), at_logit=jm, se_there=float(se[jm]),
                      rms_FO=float(np.sqrt(np.mean(fo ** 2))), median_se=float(np.median(se)),
                      n_logits_abs_z_gt_3=int(np.sum(np.abs(z) > 3)), chi2_per_dof=float(np.mean(z ** 2)),
                      FO_true_logit=float(fo[jtrue]), se_true_logit=float(se[jtrue]))
    od, ev = np.array(odd), np.array(even)
    for name, arr in (("odd_A8_full_end", od), ("even_A8_full_end", ev)):
        m = arr.mean(0) / sd
        se = arr.std(0, ddof=1) / np.sqrt(n) / sd
        jm = int(np.argmax(np.abs(m)))
        res[name] = dict(max_abs=float(np.abs(m).max()), at_logit=jm, se_there=float(se[jm]),
                         rms=float(np.sqrt(np.mean(m ** 2))), true_logit=float(m[jtrue]), se_true=float(se[jtrue]),
                         n_abs_z_gt_3=int(np.sum(np.abs(m / se) > 3)))
    json.dump(res, open(out_file, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
