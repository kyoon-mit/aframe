"""Validation within 1/2% at matched epochs: the long-conv arms vs the good den-reg run."""
import wandb
api = wandb.Api(); E = "kyoon-mit-massachusetts-institute-of-technology"
runs = {
    "den-reg good run": f"{E}/DENOISER-SCAN/keej8hie",
    "S4D d_state 64": f"{E}/CLAUDE-TESTS/s4d_energy_max_256ch_dstate64_softmax_train_snr8_prob0.96",
    "S4D d_state 512": f"{E}/CLAUDE-TESTS/s4d_energy_max_256ch_dstate512_softmax_train_snr8_prob0.96",
    "S4D d512 band init": f"{E}/CLAUDE-TESTS/s4d_energy_max_256ch_dstate512_bandinit_softmax_train_snr8_prob0.96",
    "free kernels": f"{E}/CLAUDE-TESTS/free_kernel_energy_max_256ch_softmax_train_snr8_prob0.96",
    "free kernels SNR 4": f"{E}/CLAUDE-TESTS/free_kernel_energy_max_256ch_softmax_train_snr4_prob0.96",
    "width 1024->256": f"{E}/CLAUDE-TESTS/s4d_energy_max_1024ch_dstate64_mixed_256_softmax_train_snr8_prob0.96",
    "depth 6 layers": f"{E}/CLAUDE-TESTS/s4d_regressor_6layer_dstate64_energy_max_softmax_train_snr8_prob0.96",
    "aframe ResNet1D": f"{E}/CLAUDE-TESTS/aframe_resnet1d_3463_softmax_train_snr8_prob0.96",
}
keys = ["epoch", "val/within_1pct_chirp_mass", "val/within_2pct_chirp_mass", "val/within_1pct_chirp_mass_snr8-12"]
hist = {}
for name, path in runs.items():
    try:
        h = api.run(path).history(keys=keys, pandas=True, samples=10000)
        if not len(h):  # runs without the SNR 8-12 metric
            h = api.run(path).history(keys=keys[:3], pandas=True, samples=10000)
            h[keys[3]] = float("nan")
        hist[name] = h
    except Exception as e:
        print(name, "not available:", e)
last = max(int(h["epoch"].max()) for n, h in hist.items() if n != "den-reg good run" and len(h)) if len(hist) > 1 else 0
print(f"latest epoch among the arms: {last}")
edges = [0, 5, 10, 20, 40, 60, 100, 140, 200, 300, 400, 620, 800]
print(f"{'epochs':>10s} " + " ".join(f"{n:>22s}" for n in hist))
for lo, hi in zip(edges[:-1], edges[1:]):
    if lo > last:
        break
    cells = []
    for n, h in hist.items():
        s = h[(h["epoch"] >= lo) & (h["epoch"] < hi)] if len(h) else h
        cells.append("--" if not len(s) else f"{100*s[keys[1]].mean():5.1f}/{100*s[keys[2]].mean():5.1f} ({100*s[keys[3]].mean():4.1f})".replace("( nan)", "      "))
    print(f"{lo:4d}-{hi-1:4d}  " + " ".join(f"{c:>22s}" for c in cells))
print("cells: val within 1% / 2% (within 1% at SNR 8-12), power law -3 from SNR 4")
