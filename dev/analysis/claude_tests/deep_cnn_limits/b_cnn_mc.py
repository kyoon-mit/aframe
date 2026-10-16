"""(B) Monte Carlo: random small CNNs that apply one function at every time and
then pool, on a weak toy chirp in white unit-variance noise (N = 8192).

For each network o(x) and unit-norm signal s_hat, with amplitude A (= SNR rho):
  first-order prediction  A * w . s_hat,  w = E[grad o(n)] (autograd, mean over draws)
  odd part   = E[o(n + A s) - o(n - A s)] / 2      (common noise draws)
  even part  = E[o(n + A s) + o(n - A s) - 2 o(n)] / 2
  all divided by sigma_o = std o(n), i.e. in SNR units.
For the GELU network also the second-order prediction (A^2 / 2) E[s^T Hess o(n) s],
with the Hessian term from a central second difference at eps = 1e-2 (smooth net).
Networks: conv(k=17) x3 with 16 channels, a 1x1 readout, receptive field 49
('valid' convolutions, so every output sees 49 real samples), then mean over
time; variants: GELU; max over time instead of mean; strides (1, 2, 2) (total
stride 4, receptive field 65); 'same' zero padding (edges).
Output: out/b_cnn_mc.json and out/b_cnn_mc.txt
"""
import os, sys, json, time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(int(os.environ.get("NTHREADS", "6")))
DT = torch.float64
N = 8192
AMPS = [1.0, 2.0, 4.0, 8.0, 16.0]


class LocalNet(nn.Module):
    def __init__(self, act="relu", pool="mean", strides=(1, 1, 1), padding="valid", C=16, k=17, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.convs = nn.ModuleList()
        cin = 1
        for s in strides:
            conv = nn.Conv1d(cin, C, k, stride=s, padding=(k // 2 if padding == "same" else 0), bias=True)
            with torch.no_grad():
                conv.weight.copy_(torch.randn(conv.weight.shape, generator=g) * np.sqrt(2.0 / (cin * k)))
                conv.bias.copy_((torch.rand(conv.bias.shape, generator=g) * 2 - 1) * 0.5)
            self.convs.append(conv)
            cin = C
        self.read = nn.Conv1d(C, 1, 1, bias=True)
        with torch.no_grad():
            self.read.weight.copy_(torch.randn(self.read.weight.shape, generator=g) / np.sqrt(C))
            self.read.bias.zero_()
        self.act = nn.ReLU() if act == "relu" else nn.GELU()
        self.pool = pool
        self.strides = strides
        self.k = k
        r, j = 1, 1
        for s in strides:
            r += (k - 1) * j
            j *= s
        self.R, self.S = r, j

    def forward(self, x):                      # x: (B, N)
        h = x[:, None, :]
        for conv in self.convs:
            h = self.act(conv(h))
        h = self.read(h)[:, 0, :]              # (B, P)
        return h.mean(-1) if self.pool == "mean" else h.amax(-1)


def run(net, s_hat, draws, batch, rng_seed, hess=False):
    torch.manual_seed(rng_seed)
    s = torch.tensor(s_hat, dtype=DT)
    o0_all, odd, even, grads, h2 = [], {a: [] for a in AMPS}, {a: [] for a in AMPS}, torch.zeros(N, dtype=DT), []
    wdots = []
    nb = draws // batch
    for b in range(nb):
        n = torch.randn(batch, N, dtype=DT)
        n.requires_grad_(True)
        o0 = net(n)
        gr, = torch.autograd.grad(o0.sum(), n)
        grads += gr.sum(0)
        wdots.append((gr @ s).detach().numpy())
        n = n.detach()
        with torch.no_grad():
            o0 = o0.detach()
            o0_all.append(o0.numpy())
            for a in AMPS:
                op = net(n + a * s)
                om = net(n - a * s)
                odd[a].append(((op - om) / 2).numpy())
                even[a].append(((op + om - 2 * o0) / 2).numpy())
            if hess:
                eps = 1e-2
                op = net(n + eps * s); om = net(n - eps * s)
                h2.append(((op + om - 2 * o0) / eps ** 2).numpy())
    o0 = np.concatenate(o0_all)
    sig = o0.std(ddof=1)
    w = (grads / (nb * batch)).numpy()
    wd = np.concatenate(wdots)
    res = dict(sigma_o=float(sig), mean_o=float(o0.mean()), draws=int(nb * batch),
               w_dot_s=float(wd.mean()), w_dot_s_se=float(wd.std(ddof=1) / np.sqrt(len(wd))))
    res["first_order_snr_per_unit_A"] = res["w_dot_s"] / sig
    res["first_order_snr_per_unit_A_se"] = res["w_dot_s_se"] / sig
    res["odd"], res["even"] = {}, {}
    for a in AMPS:
        d = np.concatenate(odd[a]); e = np.concatenate(even[a])
        res["odd"][a] = (float(d.mean() / sig), float(d.std(ddof=1) / np.sqrt(len(d)) / sig))
        res["even"][a] = (float(e.mean() / sig), float(e.std(ddof=1) / np.sqrt(len(e)) / sig))
    if hess:
        h = np.concatenate(h2)
        res["sHs"] = (float(h.mean()), float(h.std(ddof=1) / np.sqrt(len(h))))
    # structure of w
    R, S = net.R, net.S
    res["R"], res["S"] = R, S
    return res, w


def local_gradient(net, draws=200000, batch=20000, seed=11):
    """a = E[grad g(v)], v ~ N(0, I_R): the gradient of one output position
    of the conv stack (input exactly one receptive field long)."""
    torch.manual_seed(seed)
    acc = torch.zeros(net.R, dtype=DT)
    acc2 = torch.zeros(net.R, dtype=DT)
    for _ in range(draws // batch):
        v = torch.randn(batch, net.R, dtype=DT, requires_grad=True)
        h = v[:, None, :]
        for conv in net.convs:
            h = net.act(conv(h))
        g = net.read(h)[:, 0, :]
        assert g.shape[-1] == 1
        gr, = torch.autograd.grad(g.sum(), v)
        acc += gr.sum(0); acc2 += (gr ** 2).sum(0)
    m = acc / draws
    se = ((acc2 / draws - m ** 2) / draws).sqrt()
    return m.numpy(), se.numpy()


def w_from_a(a, S, R):
    """w_u = (1/P) sum_p a_{u - S p} for 'valid' convolutions (P output positions)."""
    P = (N - R) // S + 1
    w = np.zeros(N)
    for p in range(P):
        w[S * p:S * p + R] += a
    return w / P


def w_structure(w, R, S, padding):
    """Interior: away from both ends by 2R. Report the interior pattern's
    S-periodic mean and the residual relative to it, and the edge mass."""
    lo, hi = 2 * R, N - 2 * R
    inner = w[lo:hi]
    # S-periodic pattern of the interior
    idx = np.arange(lo, hi) % S
    pat = np.array([inner[idx == r].mean() for r in range(S)])
    resid = inner - pat[idx]
    return dict(interior_mean=float(inner.mean()), interior_pattern=[float(p) for p in pat],
                interior_resid_rms=float(np.sqrt(np.mean(resid ** 2))),
                interior_rms=float(np.sqrt(np.mean(inner ** 2))),
                head=[float(v) for v in w[:4]], tail=[float(v) for v in w[-4:]])


def main():
    out = {}
    kw_cache = {}
    lines = []
    P = lambda s="": (print(s, flush=True), lines.append(s))
    s_newt = np.load("out/sig_newtonian.npy")
    s_bns = np.load("out/sig_whitened_bns.npy")
    sys.path.insert(0, ".")
    from thm_signals import newtonian
    s_end, _ = newtonian(tc=4.0 + 0.0087 + 0.005)   # switched off right at the window end
    sigs = {"newtonian tc=3.75 s": s_newt, "newtonian ending at window end": s_end, "whitened BNS (IMRPhenomD)": s_bns}
    draws = int(os.environ.get("DRAWS", "2000"))
    batch = 50
    configs = [
        ("avg ReLU valid", dict(act="relu", pool="mean"), ["newtonian tc=3.75 s", "newtonian ending at window end", "whitened BNS (IMRPhenomD)"], False),
        ("avg GELU valid", dict(act="gelu", pool="mean"), ["newtonian tc=3.75 s"], True),
        ("max ReLU valid", dict(act="relu", pool="max"), ["newtonian tc=3.75 s"], False),
        ("avg ReLU stride 4", dict(act="relu", pool="mean", strides=(1, 2, 2)), ["newtonian tc=3.75 s"], False),
        ("avg ReLU same-padding", dict(act="relu", pool="mean", padding="same"), ["newtonian tc=3.75 s", "newtonian ending at window end"], False),
    ]
    t = np.arange(N) / 2048.0
    for f0 in (512.0, 500.0):
        tone = np.cos(2 * np.pi * f0 * t + 0.3)
        sigs[f"tone {f0:.0f} Hz"] = tone / np.linalg.norm(tone)
    configs[3] = ("avg ReLU stride 4", dict(act="relu", pool="mean", strides=(1, 2, 2)), ["newtonian tc=3.75 s", "tone 512 Hz", "tone 500 Hz"], False)
    for name, kw, siglist, hess in configs:
        net = LocalNet(seed=1, **kw).to(DT)
        for sname in siglist:
            t0 = time.time()
            res, w = run(net, sigs[sname], draws, batch, rng_seed=7, hess=hess)
            res["w_structure"] = w_structure(w, net.R, net.S, kw.get("padding", "valid"))
            if kw.get("pool", "mean") == "mean" and kw.get("padding", "valid") == "valid":
                if "a" not in kw_cache.setdefault(name, {}):
                    kw_cache[name]["a"] = local_gradient(net)
                a, a_se = kw_cache[name]["a"]
                w_ex = w_from_a(a, net.S, net.R)
                sh = sigs[sname]
                res["w_exact_dot_s"] = float(w_ex @ sh)
                res["first_order_exact_per_unit_A"] = float(w_ex @ sh) / res["sigma_o"]
                interior = slice(2 * net.R, N - 2 * net.R)
                res["first_order_interior_part"] = float(w_ex[interior] @ sh[interior]) / res["sigma_o"]
                res["sum_a"] = float(a.sum())
                res["w_autograd_vs_exact_rms"] = float(np.sqrt(np.mean((w - w_ex) ** 2)))
                res["w_exact_rms"] = float(np.sqrt(np.mean(w_ex ** 2)))
            key = f"{name} | {sname}"
            out[key] = res
            np.save(f"out/w_{name.replace(' ', '_')}_{sname.split()[0]}.npy", w)
            P(f"== {key}: R={res['R']} S={res['S']} draws={res['draws']} sigma_o={res['sigma_o']:.5g} ({time.time()-t0:.0f} s)")
            P(f"   first order per unit A: w.s/sigma_o = {res['first_order_snr_per_unit_A']:+.5f} +- {res['first_order_snr_per_unit_A_se']:.5f} (autograd, per-draw gradients)")
            if "w_exact_dot_s" in res:
                P(f"   first order per unit A from a = E[grad g] (2e5 draws): {res['first_order_exact_per_unit_A']:+.6f}; interior part {res['first_order_interior_part']:+.6f}; sum a = {res['sum_a']:.5f}; rms(w_autograd - w_from_a) / rms(w_from_a) = {res['w_autograd_vs_exact_rms']/res['w_exact_rms']:.3f}")
            ws = res["w_structure"]
            P(f"   w interior mean {ws['interior_mean']:.4e}, interior rms {ws['interior_rms']:.4e}, rms about S-periodic pattern {ws['interior_resid_rms']:.2e}; pattern {['%.3e' % p for p in ws['interior_pattern']]}")
            P(f"   {'A':>5s} {'pred 1st':>10s} {'odd/sigma':>20s} {'even/sigma':>20s}" + (" pred 2nd" if hess else ""))
            for a in AMPS:
                od, ods = res["odd"][a]; ev, evs = res["even"][a]
                line = f"   {a:5.1f} {a*res['first_order_snr_per_unit_A']:+10.5f} {od:+10.5f} +- {ods:.5f} {ev:+10.5f} +- {evs:.5f}"
                if hess:
                    line += f" {0.5*a*a*res['sHs'][0]/res['sigma_o']:+.5f} +- {0.5*a*a*res['sHs'][1]/res['sigma_o']:.5f}"
                P(line)
            # exponent of the even part from the smallest amplitudes with significant signal
            amps = np.array(AMPS); ev = np.array([res["even"][a][0] for a in AMPS]); evs = np.array([res["even"][a][1] for a in AMPS])
            ok = ev > 3 * evs
            if ok.sum() >= 2:
                sl = np.polyfit(np.log(amps[ok][:3]), np.log(ev[ok][:3]), 1)[0]
                res["even_exponent_low"] = float(sl)
                P(f"   even part: log-log slope over the first significant amplitudes {list(amps[ok][:3])}: {sl:.3f}")
            json.dump(out, open("out/b_cnn_mc.json", "w"), indent=1)
            open("out/b_cnn_mc.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
