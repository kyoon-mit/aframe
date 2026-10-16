"""Second-order (even) response of the trained aframe ResNet: does it see only
the power spectrum, as Theorem 2 says for a time-shift-invariant network?

Signals (unit norm, split over both detectors), all with the same circular
power spectrum:
  fwd_end : whitened BNS, peak at 3.75 s + d (non-circular shift as in v_edge.py, data placement)
  fwd_mid : the same 4 s chirp rolled circularly so the peak sits at 1.75 s + d
  rev_mid : the chirp played backwards, rolled so its peak sits where fwd_mid's does
d uniform over 1024 whole-sample shifts. Even part (o(n+As) + o(n-As) - 2 o(n)) / 2
in units of sd(o(n)), for A = 2, 4, 8, for every logit.
Invariant network: fwd_end = fwd_mid at every order (circularly equivalent),
fwd_mid = rev_mid at second order.
usage: v_edge2.py NDRAWS OUT.json
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
AMPS = (2.0, 4.0, 8.0)
KEYS = ("fwd_end", "fwd_mid", "rev_mid")


def main():
    n_draws, out_file = int(sys.argv[1]), sys.argv[2]
    net = ResNet1D(2, layers=[3, 4, 6, 3], classes=235, kernel_size=3, norm_layer=GroupNorm1DGetter(groups=16))
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict({k.split("model.net.", 1)[1]: v for k, v in sd.items() if k.startswith("model.net.")})
    net = net.float().eval()
    s = np.load(SIG)
    pk = int(np.argmax(np.abs(s)))
    r = s[::-1].copy()
    rpk = int(np.argmax(np.abs(r)))
    mid = pk - 4096

    def place(d):
        out = {}
        e = np.zeros(L)
        if d >= 0:
            e[d:] = s[:L - d]
        else:
            e[:L + d] = s[-d:]
        out["fwd_end"] = e / np.linalg.norm(e)
        out["fwd_mid"] = np.roll(s, mid - pk + d)
        out["rev_mid"] = np.roll(r, mid - rpk + d)
        return out

    rng = np.random.default_rng(2024)
    f = np.fft.rfftfreq(L, 1 / 2048)
    o0s, ev = [], {(k, A): [] for k in KEYS for A in AMPS}
    t0 = time.time()
    B = 4
    for i0 in range(0, n_draws, B):
        rows = []
        nb = min(n_draws, i0 + B) - i0
        for _ in range(nb):
            n = rng.standard_normal((2, L))
            N = np.fft.rfft(n)
            N[:, f < 20] = 0
            n = np.fft.irfft(N, n=L)
            sig = place(int(rng.integers(-512, 512)))
            rows.append(n)
            for k in KEYS:
                two = np.stack([sig[k], sig[k]]) / np.sqrt(2)
                for A in AMPS:
                    rows += [n + A * two, n - A * two]
        with torch.no_grad():
            o = net(torch.as_tensor(np.stack(rows), dtype=torch.float32)).double().numpy()
        per = 1 + 2 * len(KEYS) * len(AMPS)
        for q in range(nb):
            b = q * per
            o0s.append(o[b])
            c = b + 1
            for k in KEYS:
                for A in AMPS:
                    ev[(k, A)].append((o[c] + o[c + 1] - 2 * o[b]) / 2)
                    c += 2
        if (i0 // B) % 10 == 0:
            print(f"{i0 + nb}/{n_draws} {time.time() - t0:.0f}s", flush=True)
    o0s = np.array(o0s)
    sdv = o0s.std(0, ddof=1)
    n = len(o0s)
    res = {"n": n}
    E = {key: np.array(v) / sdv for key, v in ev.items()}
    for A in AMPS:
        row = {}
        for k in KEYS:
            row[k] = dict(rms=float(np.sqrt(np.mean(E[(k, A)].mean(0) ** 2))),
                          median_se=float(np.median(E[(k, A)].std(0, ddof=1) / np.sqrt(n))))
        for a, b in (("fwd_end", "fwd_mid"), ("fwd_mid", "rev_mid")):
            dlt = E[(a, A)] - E[(b, A)]                            # paired over the same noise
            m = dlt.mean(0)
            se = dlt.std(0, ddof=1) / np.sqrt(n)
            row[f"{a}-{b}"] = dict(rms_diff=float(np.sqrt(np.mean(m ** 2))), max_abs_diff=float(np.abs(m).max()),
                                   median_se=float(np.median(se)), n_logits_abs_z_gt_3=int(np.sum(np.abs(m / se) > 3)),
                                   ratio_of_rms=float(np.sqrt(np.mean(E[(a, A)].mean(0) ** 2)) /
                                                      np.sqrt(np.mean(E[(b, A)].mean(0) ** 2))))
        res[str(A)] = row
    json.dump(res, open(out_file, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
