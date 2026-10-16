"""Max-pool S4D (clean signal): chirp-mass error with dropout off (eval) and on
(train mode), on validation-pipeline and training-pipeline batches, by SNR."""

import sys

import numpy as np
import torch
import yaml
from lightning.pytorch.trainer.states import RunningStage, TrainerFn
from lightning.pytorch.utilities import move_data_to_device

from train.cli import AframeCLI

CFG = "/home/kyoon/SSM-BNS/aframe/dev/local/train/claude_tests/oracle/sig_only_reg_s4d_max_64x32x4.yaml"
CK = "/home/kyoon/SSM-BNS/aframe/dev/runs/claude_tests/sig_only_reg_s4d_max_64x32x4/CLAUDE-TESTS/sig_only_reg_s4d_max_64x32x4/checkpoints/"
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dev = torch.device("cuda")

cfg = yaml.safe_load(open(CFG))
cfg = {k: v for k, v in cfg.items() if k in ("seed_everything", "trainer", "model", "data")}
cli = AframeCLI(run=False, args=["--config", yaml.safe_dump(cfg), "--trainer.devices=1", "--trainer.strategy=auto",
                                 "--trainer.logger=false", "--trainer.callbacks=[]", "--data.max_num_workers=0",
                                 "--data.persistent_workers=false", "--data.valid_livetime=2000"])
model, dm, trainer = cli.model, cli.datamodule, cli.trainer
model.to(dev)
trainer.strategy.connect(model)
dm.trainer = trainer
trainer.state.fn = TrainerFn.FITTING
dm.setup("fit")


def batches(stage):
    trainer.state.stage = stage
    loader = dm.val_dataloader() if stage == RunningStage.VALIDATING else dm.train_dataloader()
    out = []
    for i, b in enumerate(loader):
        if i >= NB:
            break
        b = dm.on_before_batch_transfer(b, 0)
        b = move_data_to_device(b, dev)
        X, X_clean, y, params = dm.on_after_batch_transfer(b, 0)
        out.append((X_clean, params["chirp_mass"], params["snr"]))
    return out


def errors(ck, data, train_mode):
    sd = torch.load(CK + ck, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    model.to(dev)
    model.train() if train_mode else model.eval()
    rel, snr = [], []
    with torch.no_grad():
        for X_clean, mc, s in data:
            mean, var = model._split(model(X_clean))
            pred = mean[:, 0] * model.y_std[0] + model.y_mean[0]
            rel.append(((pred - mc) / mc).abs().cpu()); snr.append(s.cpu())
    return torch.cat(rel).numpy(), torch.cat(snr).numpy()


torch.manual_seed(0)
sets = {"validation pipeline": batches(RunningStage.VALIDATING), "training pipeline": batches(RunningStage.TRAINING)}
bins = [(4, 8), (8, 16), (16, 32), (32, 101)]
for ck in ("ck_05.ckpt", "last.ckpt"):
    for name, data in sets.items():
        for mode in (False, True):
            r, s = errors(ck, data, mode)
            parts = [f"{a}-{b}: {100*np.mean(r[(s>=a)&(s<b)]<0.01):5.1f}% mae {np.median(r[(s>=a)&(s<b)]):.3f}" for a, b in bins if ((s>=a)&(s<b)).any()]
            print(f"{ck:11s} {name:20s} dropout {'on ' if mode else 'off'} n={len(r)} within1% {100*np.mean(r<0.01):5.1f}  median|rel err| {np.median(r):.4f} || " + " | ".join(parts), flush=True)
