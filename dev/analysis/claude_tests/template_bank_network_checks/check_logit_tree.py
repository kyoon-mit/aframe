"""CPU check: for the MLP checkpoint, what step1_local_bank.py line 126 writes
to the "logit" tree vs the model's own detection output (forward / test_step)."""
import math

import torch

from train.experimental.template_bank.arch import TemplateBankNet

torch.set_num_threads(8)
torch.manual_seed(0)
CK = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/"
      "template_bank_703_spin3_1024hz_detection_mlp_train_snr4_prob0.5/CLAUDE-TESTS/"
      "template_bank_703_spin3_1024hz_detection_mlp_train_snr4_prob0.5/checkpoints/ck_15.ckpt")
net = TemplateBankNet(
    bank_file="/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/template_bank/bank_mc0.5pct_spin3_q0.8.npz",
    decimate=2, max_lag=0.3, coincidence=0.01, fduration=1.0, highpass=20.0,
    beta=0.5, n_bands=16, detection_hidden=32,
)
sd = torch.load(CK, map_location="cpu", weights_only=False)["state_dict"]
missing = net.load_state_dict({k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")})
print("load:", missing)
print("det_scale", net.det_scale.item(), "det_shift", net.det_shift.item(), "beta", net.log_beta.exp().item())
net.eval()

B, L, sr = 6, 8192, 2048.0
X = torch.randn(B, 2, L)
t = torch.arange(L) / sr
# rows 3-5: sine-Gaussian glitch in H1 only, growing amplitude
for i, amp in zip((3, 4, 5), (5.0, 20.0, 80.0)):
    X[i, 0] += amp * torch.exp(-((t - 3.7) ** 2) / (2 * 0.01**2)) * torch.sin(2 * math.pi * 120 * t)

with torch.no_grad():
    # step1_local_bank.py lines 125-126
    new, plain, peak, feats = net.newsnr2(X, features=True)
    logits, det_step1 = net.head(peak)
    # model's detection output (forward / test_step with detection_hidden > 0)
    _, det_model = net(X)
    det_test = net.detect(feats, new, plain)
    evidence = torch.logsumexp(logits, -1) - math.log(logits.shape[-1])

for i in range(B):
    print(f"row {i}: plain {plain[i]:9.1f} newsnr2 {new[i]:7.1f} | step1 'logit' {det_step1[i]:9.3f} "
          f"(evidence {evidence[i]:9.3f}) | model det {det_model[i]:8.3f} (test_step {det_test[i]:8.3f})")
print("step1 'logit' == evidence exactly:", torch.equal(det_step1, evidence))
print("model det == test_step det:", torch.allclose(det_model, det_test))
print("max |step1 - model|:", (det_step1 - det_model).abs().max().item())
print("rank step1:", det_step1.argsort().tolist(), " rank model:", det_model.argsort().tolist())
