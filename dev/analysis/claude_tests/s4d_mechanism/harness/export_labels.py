"""Write the main labels of each saved set as plain .npy files next to the windows."""
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / "data"
for name in ("clean", "noisy"):
    z = np.load(D / f"{name}_labels.npz")
    for k in ("chirp_mass", "snr", "merger_sample"):
        np.save(D / f"{name}_{k}.npy", z[k])
        print(name, k, z[k].shape, z[k].dtype)
