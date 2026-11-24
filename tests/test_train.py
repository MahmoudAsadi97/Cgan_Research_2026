"""Training on a small synthetic archive: a seed fixes the result, and an interrupted run resumes exactly."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

import train as T
from data_prep import N_LABELS, decode_label

SMALL = {"epochs": 4, "val_every": 1, "val_max_per_label": 40}
VARIANTS = {
    "final": {**SMALL, "batch_size": 128, "hidden_size": 32},          # cWGAN, EMA, critic with hour-to-hour changes
    "abl_quantile": {**SMALL, "batch_size": 500},                      # reference cGAN, reshuffled, value transform
    "faithful": {**SMALL, "batch_size": 500},                          # reference cGAN as released
}


def config(name, archive, seed=0):
    return {**T.DEFAULTS, **T.CONFIGS[name], **VARIANTS[name], "config": name, "seed": seed, "data": str(archive)}


def weights(run_dir, ckpt="last"):
    state = torch.load(run_dir / f"{ckpt}.pt", map_location="cpu", weights_only=True)["model"]
    return torch.cat([v.flatten().double() for part in sorted(state) for v in state[part].values()])


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("name", list(VARIANTS))
def test_a_seed_fixes_the_training(name, archive, tmp_path):
    T.train(config(name, archive), tmp_path / "a")
    T.train(config(name, archive), tmp_path / "b")
    T.train(config(name, archive, seed=1), tmp_path / "c")
    assert torch.equal(weights(tmp_path / "a"), weights(tmp_path / "b"))
    assert not torch.equal(weights(tmp_path / "a"), weights(tmp_path / "c"))
    val = pd.read_csv(tmp_path / "a" / "val.csv")
    assert val.epoch.tolist() == [1, 2, 3, 4] and val.val_accuracy.between(0, 1).all()


@pytest.mark.parametrize("name", list(VARIANTS))
def test_an_interrupted_run_resumes_exactly(name, archive, tmp_path, monkeypatch):
    cfg = config(name, archive)
    T.train(cfg, tmp_path / "whole")

    score, calls = T.Validator.score, []

    def interrupted(self, wrapped):
        calls.append(1)
        if len(calls) == 3:                                            # during the validation of epoch 3
            raise KeyboardInterrupt
        return score(self, wrapped)

    monkeypatch.setattr(T.Validator, "score", interrupted)
    with pytest.raises(KeyboardInterrupt):
        T.train(cfg, tmp_path / "parts")
    monkeypatch.setattr(T.Validator, "score", score)
    assert not (tmp_path / "parts" / "last.pt").exists()
    summary = T.train(cfg, tmp_path / "parts", resume=True)

    for ckpt in ("last", "best"):
        assert torch.equal(weights(tmp_path / "whole", ckpt), weights(tmp_path / "parts", ckpt))
    whole = json.loads((tmp_path / "whole" / "train_summary.json").read_text())
    assert (summary["best_epoch"], summary["best_val_accuracy"]) == (whole["best_epoch"], whole["best_val_accuracy"])
    for table, columns in (("val.csv", ["epoch", "val_accuracy", "val_min_label"]), ("history.csv", ["epoch", "loss_G", "loss_D"])):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "whole" / table)[columns],
                                      pd.read_csv(tmp_path / "parts" / table)[columns])
    starts = json.loads((tmp_path / "parts" / "environment.json").read_text())
    assert [s["start_epoch"] for s in starts] == [0, 2] and starts[0]["device"] in ("cpu", "cuda")


def test_generated_days_repeat_and_respect_the_floor(archive, tmp_path):
    T.train(config("final", archive), tmp_path)
    wrapped, cfg = T.load_run(tmp_path, "best")
    labels = np.repeat(np.arange(N_LABELS), 20)
    days = wrapped.sample_kwh(labels, seed=5)
    assert days.shape == (len(labels), 24) and np.isfinite(days).all()
    assert np.array_equal(days, wrapped.sample_kwh(labels, seed=5))
    assert not np.array_equal(days, wrapped.sample_kwh(labels, seed=6))
    no_pv = decode_label(labels)[1] & 1 == 0
    assert cfg["nonneg_without_pv"] and days[no_pv].min() >= 0
