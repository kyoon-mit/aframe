from types import SimpleNamespace

import lightning.pytorch.cli  # noqa: F401  (train.callbacks needs pl.cli loaded)
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from train.callbacks import ResumableEarlyStopping

M = "val/within_1pct_chirp_mass"


def _trainer(epoch, best=None, path=""):
    mc = ModelCheckpoint(monitor=M, mode="max")
    mc.best_model_score = None if best is None else torch.tensor(best)
    mc.best_model_path = path
    return SimpleNamespace(checkpoint_callbacks=[mc], current_epoch=epoch)


def test_resume_starts_from_checkpoint_best():
    es = ResumableEarlyStopping(monitor=M, mode="max", patience=80)
    es.on_train_start(_trainer(128, 0.996, "/x/checkpoints/ck_126.ckpt"), None)
    assert abs(float(es.best_score) - 0.996) < 1e-6
    assert es.wait_count == 1


def test_fresh_run_unchanged():
    es = ResumableEarlyStopping(monitor=M, mode="max", patience=80)
    es.on_train_start(_trainer(0), None)
    assert not torch.isfinite(es.best_score) and es.wait_count == 0


def test_saved_state_wins():
    es = ResumableEarlyStopping(monitor=M, mode="max", patience=80)
    es.best_score, es.wait_count = torch.tensor(0.5), 7
    es.on_train_start(_trainer(128, 0.996, "/x/ck_126.ckpt"), None)
    assert abs(float(es.best_score) - 0.5) < 1e-6 and es.wait_count == 7


def test_other_monitor_ignored():
    es = ResumableEarlyStopping(monitor="val/loss", mode="min", patience=80)
    es.on_train_start(_trainer(128, 0.996, "/x/ck_126.ckpt"), None)
    assert not torch.isfinite(es.best_score)


def test_cycle_end_after():
    f = ResumableEarlyStopping.cycle_end_after
    assert [f(e, 10, 40, 2) for e in (0, 49, 50, 129, 130, 151, 207, 289, 290)] == [49, 49, 129, 129, 289, 289, 289, 289, 609]


def _cycle_trainer(epoch):
    sch = SimpleNamespace(warmup_epochs=10, T_0=40, T_mult=2)
    return SimpleNamespace(current_epoch=epoch, lr_scheduler_configs=[SimpleNamespace(scheduler=sch)], should_stop=False)


def test_cycle_aware_defers_stop_until_cycle_end():
    es = ResumableEarlyStopping(monitor=M, mode="max", patience=80, cycle_aware=True)
    es.best_score, es.wait_count = torch.tensor(0.99), 80     # best at epoch 207
    es._trainer = _cycle_trainer(287)
    stop, _ = es._evaluate_stopping_criteria(torch.tensor(0.98))
    assert not stop                                           # cycle ends at 289
    es.wait_count = 81
    es._trainer = _cycle_trainer(289)
    stop, _ = es._evaluate_stopping_criteria(torch.tensor(0.98))
    assert stop


def test_plain_stop_without_cycle_aware():
    es = ResumableEarlyStopping(monitor=M, mode="max", patience=80)
    es.best_score, es.wait_count = torch.tensor(0.99), 80
    es._trainer = _cycle_trainer(287)
    stop, _ = es._evaluate_stopping_criteria(torch.tensor(0.98))
    assert stop
