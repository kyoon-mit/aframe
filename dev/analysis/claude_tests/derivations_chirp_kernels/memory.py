"""Per-layer S4D memory 1/(dt*alpha) and tone frequency in trained den-reg checkpoints."""
import sys
import numpy as np
import torch

FS = 2048.0
for path in sys.argv[1:]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    print("==", path.split("claude_tests/")[-1], "epoch", ck["epoch"])
    prefixes = sorted({k.rsplit(".kernel.", 1)[0] for k in sd if k.endswith("kernel.log_dt")})
    all_mem = []
    for p in prefixes:
        dt = sd[p + ".kernel.log_dt"].double().exp()[:, None]
        alpha = sd[p + ".kernel.log_A_real"].double().exp()
        omega = sd[p + ".kernel.A_imag"].double()
        mem = (1.0 / (dt * alpha)).numpy().ravel()   # samples to fall by e
        phase_step = (dt * omega).abs().numpy().ravel()  # radians per sample
        above_nyq = np.mean(phase_step > np.pi)
        all_mem.append(mem)
        print(f"{p.split('model.model.')[-1]:28s} median memory {np.median(mem):7.2f} samples"
              f" = {1e3*np.median(mem)/FS:6.2f} ms; >8192: {np.mean(mem>8192):.3f};"
              f" phase step > pi: {above_nyq:.3f}")
    m = np.concatenate(all_mem)
    print(f"all layers: median {np.median(m):.2f} samples = {1e3*np.median(m)/FS:.2f} ms")
