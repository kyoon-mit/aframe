"""Section 7 checks: GELU's odd part is exactly y/2, so the part of each
channel's output that is linear in the input passes the activation with
gain 1/2 and can be added in step across channels by the next 1x1
convolution; and the resonator count per channel of the runs discussed."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
import torch
from scipy.special import erf

from common import RES

out = {}
y = np.linspace(-12, 12, 2_400_001)
gelu = lambda v: v * 0.5 * (1 + erf(v / np.sqrt(2)))  # noqa: E731
odd = 0.5 * (gelu(y) - gelu(-y))
out["gelu_odd_minus_half_y_max_abs"] = float(np.max(np.abs(odd - 0.5 * y)))
yt = torch.linspace(-12, 12, 240001, dtype=torch.float64)
g = torch.nn.functional.gelu(yt)
out["torch_gelu_odd_minus_half_y_max_abs"] = float((0.5 * (g - torch.nn.functional.gelu(-yt)) - 0.5 * yt).abs().max())
print(out)

ck = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt",
                  "sig_only_reg_s4d_max_dropout0_64x32x4__ck_69.ckpt")
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
shapes = {k: list(v.shape) for k, v in sd.items() if "s4_layers.0" in k or "encoder" in k}
out["clean_run_layer0_shapes"] = shapes
out["clean_run_resonators_per_channel"] = shapes["model.net.s4_layers.0.kernel.C"][1]
out["clean_run_channels"] = shapes["model.net.s4_layers.0.kernel.C"][0]
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(RES, "c7_layer.json"), "w"), indent=1)
