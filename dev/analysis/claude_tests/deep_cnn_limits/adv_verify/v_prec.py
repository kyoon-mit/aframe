"""float32 vs float64 forward of the trained ResNet on the same inputs (even part at A = 2 and 8)."""
import numpy as np, torch
from ml4gw.nn.norm import GroupNorm1DGetter
from ml4gw.nn.resnet.resnet_1d import ResNet1D
torch.set_num_threads(8)
CKPT = ("/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/"
        "CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96/checkpoints/ck_244.ckpt")
net = ResNet1D(2, layers=[3, 4, 6, 3], classes=235, kernel_size=3, norm_layer=GroupNorm1DGetter(groups=16))
sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state_dict"]
net.load_state_dict({k.split("model.net.", 1)[1]: v for k, v in sd.items() if k.startswith("model.net.")})
net.eval()
s = np.load("../out/sig_whitened_bns.npy")
rng = np.random.default_rng(5)
rows = []
for i in range(8):
    n = rng.standard_normal((2, 8192))
    two = np.stack([s, s]) / np.sqrt(2)
    rows += [n, n + 2 * two, n - 2 * two, n + 8 * two, n - 8 * two]
X = np.stack(rows)
with torch.no_grad():
    o32 = net.float()(torch.as_tensor(X, dtype=torch.float32)).double().numpy()
    o64 = net.double()(torch.as_tensor(X, dtype=torch.float64)).numpy()
for o, lab in ((o32, "f32"), (o64, "f64")):
    o = o.reshape(8, 5, -1)
    e2 = (o[:, 1] + o[:, 2] - 2 * o[:, 0]) / 2
    e8 = (o[:, 3] + o[:, 4] - 2 * o[:, 0]) / 2
    print(lab, "even A=2 mean |.|", np.abs(e2).mean(), "even A=8 mean |.|", np.abs(e8).mean())
print("max |o32 - o64|", np.abs(o32 - o64).max(), "logit scale", np.abs(o64).mean())
