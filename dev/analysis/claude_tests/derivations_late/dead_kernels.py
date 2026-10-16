"""Re-run of the 13 Oct kernel ablation (epoch 141), CPU: 125 chirp-like vs 131 other kernels."""
import h5py, numpy as np, torch, sys
torch.set_num_threads(8)
from train.experimental.long_conv.arch import LongConvEnergyNet
from train.experimental.template_bank.arch import TemplateBankNet
R = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests"
name = "free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96"
st = torch.load(f"{R}/{name}/test_ck_141/test.ckpt", map_location="cpu", weights_only=False)
net = LongConvEnergyNet(channels=256, kernel="free")
net.load_state_dict({k[6:]: v for k, v in st["state_dict"].items() if k.startswith("model.")}, strict=False)
net.eval()
bank = TemplateBankNet(f"{R}/template_bank/bank_mc0.5pct_spin3_q0.8.npz")
with torch.no_grad():
    T = torch.fft.ifft(bank.filters[0].conj(), n=bank.n); T = T / T.norm(dim=1, keepdim=True)
    n2 = 2 * bank.n; TF = torch.fft.fft(T, n=n2)
    h = torch.flip(net.kernels(), [1]).conj(); h = h / h.norm(dim=1, keepdim=True); H = torch.fft.fft(h, n=n2)
    best = torch.zeros(256)
    for i in range(0, T.shape[0], 32):
        m, a = torch.fft.ifft(H[:, None] * TF[None, i:i+32].conj(), dim=-1).abs().amax(-1).max(-1)
        best = torch.maximum(best, m)
chirp = best > 0.5
print("epoch", st["epoch"], "chirp-like", int(chirp.sum()), "others", int((~chirp).sum()), "max match", float(best.max()))
BENCH = (f"{R}/den_reg_sched_alpha_0.5_floor_1e-2_c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")
w = h5py.File(f"{BENCH}/snr4_uniform/windows.h5", "r"); n = 6400
X = w["X"][:n]; mc = w["params/chirp_mass"][:n]; snr = w["params/snr"][:n]; sig = np.isfinite(mc)
Z = []
with torch.no_grad():
    for a in range(0, n, 16):
        x = torch.tensor(X[a:a+16])
        Z.append(torch.log1p(net.peak_energy(x)))
Z = torch.cat(Z)
W, b = net.out.weight.detach(), net.out.bias.detach()
res = {}
for lab, m in (("all", torch.ones(256, dtype=torch.bool)), ("chirp-like", chirp), ("others", ~chirp)):
    z = Z[:, m]; z = (z - z.mean(-1, keepdim=True)) / z.std(-1, keepdim=True)
    logits = z @ W[:, m].T + b
    log_mc, _ = net.estimate(logits)
    est = log_mc.exp().numpy()
    err = np.abs(est[sig] - mc[sig]) / mc[sig]
    s = snr[sig]
    band = (s >= 8) & (s < 10)
    res[lab] = est[sig]
    print(f"{lab:11s} n={sig.sum()} within 1/2% all: {100*(err<0.01).mean():.1f}/{100*(err<0.02).mean():.1f}  SNR 8-10: {100*(err[band]<0.01).mean():.1f}/{100*(err[band]<0.02).mean():.1f}")
rng = np.random.default_rng(0); t = mc[sig]; e = res["others"]
sh = [np.mean(np.abs(rng.permutation(e) - t) / t < 0.01) for _ in range(200)]
print("others, predictions shuffled across events: within 1%%: %.2f%% (sd %.2f)" % (100*np.mean(sh), 100*np.std(sh)))
np.savez("/tmp/claude-1024/-home-kyoon-SSM-BNS/82effc75-bb98-44ad-af20-e41e39a7258d/scratchpad/late/dead_kernels_est.npz", truth=t, snr=snr[sig], **{k.replace("-","_"): v for k, v in res.items()})
