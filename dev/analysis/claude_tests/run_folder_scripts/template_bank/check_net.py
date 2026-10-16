"""Check TemplateBankNet against bank_check on benchmark windows."""
import sys
import h5py, numpy as np, torch
from train.experimental.template_bank.arch import TemplateBankNet
sys.path.insert(0, "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/bank_check")
from analyze import table
BENCH = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/den_reg_sched_alpha_0.5_floor_1e-2_"
         "c_rho_5_lambda_amp_3_powerlaw_m2_snr_d128_n6_mc_q_rep4_smooth20_wp0.96_ddp/mf_bench_ep630")
dev = torch.device("cuda")
net = TemplateBankNet("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz").to(dev)
bank = np.load("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz")
for pop in ["snr4_uniform", "snr4_powerlaw", "snr8_powerlaw"]:
    w = h5py.File(f"{BENCH}/{pop}/windows.h5"); n = w["X"].shape[0]
    X = w["X"][:]; mc = w["params/chirp_mass"][:]; snr = w["params/snr"][:]
    peaks, est = [], []
    noise_mean = []
    with torch.no_grad():
        for a in range(0, n, 32):
            x = torch.tensor(X[a:a+32], device=dev)
            logits, det = net(x)
            lm, _ = net.estimate(logits)
            est.append(lm.exp().cpu().numpy()); peaks.append(net.peak_snr2(x).cpu().numpy())
            bg = np.isnan(mc[a:a+32])
            if bg.any() and len(noise_mean) < 20:
                noise_mean.append(net.snr2_series(x[torch.tensor(bg, device=dev)]).mean(dim=(0, 2, 3)).cpu().numpy())
    peaks = np.concatenate(peaks); est = np.concatenate(est)
    print(pop, "noise mean rho^2 per detector (expect ~2):", np.mean(noise_mean, 0).round(3))
    ref = np.load(f"/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/bank_check/{pop}_spin_avg_1024.npz")
    # map model templates to bank_check rows
    pos = {r: i for i, r in enumerate(ref["rows"])}
    idx = np.array([pos.get(r, -1) for r in bank["rows"]])
    ok = idx >= 0
    ratio = peaks[:, ok] / ref["peak"][:, idx[ok]]
    print(pop, f"templates matched {ok.sum()}/{len(ok)}; model/bank_check peak ratio median {np.median(ratio):.4f} (expect 2), p1 {np.percentile(ratio,1):.4f} p99 {np.percentile(ratio,99):.4f}")
    sig = np.isfinite(mc)
    table(f"{pop} untrained net (beta .5)", est[sig], mc[sig], snr[sig])
