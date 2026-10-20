"""Training of the SLIC score network on whitened noise windows."""

import copy

import lightning.pytorch as pl
import torch

from train.experimental.slic.score_net import ScoreNet
from train.experimental.slic.sde import denoising_loss, ema_update


class ScoreTask(pl.LightningModule):
    """Denoising score matching on windows of shape (batch, 1, samples), as
    the scoregen_jax training step.

    AdamW at a constant learning rate, gradients clipped by global norm,
    and a weight average ``ema_net`` that takes in the weights from before
    each step. Validation uses ``ema_net``; the checkpoint holds both
    networks.
    """

    def __init__(
        self,
        net: ScoreNet,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.0,
        ema_rate: float = 0.999,
        max_grad_norm: float = 1.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = net
        self.ema_net = copy.deepcopy(net).requires_grad_(False)

    def training_step(self, windows, batch_idx):
        loss = denoising_loss(self.net, windows)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(windows),
        )
        return loss

    def validation_step(self, windows, batch_idx):
        loss = denoising_loss(self.ema_net, windows)
        self.log(
            "val/loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(windows),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.net.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        """optax clip_by_global_norm(max_grad_norm)."""
        grads = [p.grad for p in self.net.parameters() if p.grad is not None]
        norm = torch.linalg.vector_norm(
            torch.stack(torch._foreach_norm(grads))
        )
        scale = torch.clamp(self.hparams.max_grad_norm / norm, max=1.0)
        torch._foreach_mul_(grads, scale)
        self.log("train/grad_norm", norm, on_step=True, on_epoch=False)

    def on_before_optimizer_step(self, optimizer):
        ema_update(
            list(self.ema_net.parameters()),
            list(self.net.parameters()),
            self.hparams.ema_rate,
        )


def load_score_net(ckpt_path: str, map_location="cpu") -> ScoreNet:
    """The weight-averaged network of a ScoreTask checkpoint, in eval mode."""
    ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    net = ckpt["hyper_parameters"]["net"]
    prefix = "ema_net."
    state = {
        key[len(prefix) :]: value
        for key, value in ckpt["state_dict"].items()
        if key.startswith(prefix)
    }
    net.load_state_dict(state)
    return net.eval()
