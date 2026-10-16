"""Per-epoch trace of one read-out without early stopping: within 1% on data/es split at SNR 8, and on
data/es8. Shows what the SNR >= 4 early-stopping criterion trades.

usage: diag_epochs.py TAG CONFIG N EPOCHS
"""
import json
import sys

import numpy as np
import torch

import readout as R
import readout_big as B


def main(tag, cfg_name, n, epochs):
    xtr, mctr = B.load_train(tag, n)
    xes, mces, snres, _ = R.features("es", tag)
    x8, mc8, _, _ = R.features("es8", tag)
    torch.manual_seed(0)
    model = R.build(R.CONFIGS[cfg_name], xtr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    xd = xtr.to(R.DEV)
    lt = torch.log(torch.as_tensor(mctr, dtype=torch.float32, device=R.DEV))
    g = R.GRID.to(R.DEV)
    hist = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xd), device=R.DEV)
        for i in range(0, len(xd), 512):
            idx = perm[i:i + 512]
            t = torch.softmax(-((lt[idx, None] - g[None]) ** 2) / (2 * R.WIDTH**2), -1)
            loss = -(t * torch.log_softmax(model(xd[idx]), -1)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        w = R.within(R.predict(model, xes), mces)
        w8 = R.within(R.predict(model, x8), mc8)
        row = {"epoch": ep, "es_all": float(w.mean()), "es_snr_lt8": float(w[snres < 8].mean()),
               "es_snr_ge8": float(w[snres >= 8].mean()), "es8": float(w8.mean())}
        hist.append(row)
        print(json.dumps(row), flush=True)
    json.dump(hist, open(R.R / f"diag_epochs__{tag}__{cfg_name}__n{n}.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
