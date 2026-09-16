"""Numbered checkpoints survive subsequent rolling checkpoint saves."""

import json
from pathlib import Path

import pytest
import torch

from retroagi.core import load_checkpoint
from retroagi.stages.block_smb.train import train_and_evaluate_block_smb
from scripts.tests.test_block_smb_training import static_vision_factory, tiny_config


def test_epoch_five_snapshot_survives_later_training(tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        config = tiny_config(
            epochs=6,
            generated_scenarios=0,
            checkpoint_path=tmp_path / "policy.pth",
            save_checkpoints=True,
            retain_checkpoint_epochs=[5],
            log_path=tmp_path / "events.jsonl",
        )
        train_and_evaluate_block_smb(config, vision_factory=static_vision_factory)
        retained = load_checkpoint(tmp_path / "policy.epoch5.pth")
        latest = load_checkpoint(tmp_path / "policy.pth")
        assert retained["epoch"] == 5 and latest["epoch"] == 6
        assert retained["global_step"] < latest["global_step"]
        assert {"model", "optimizer", "torch_rng", "python_rng", "numpy_rng"} <= retained[
            "states"
        ].keys()
        assert retained["config"]["retain_checkpoint_epochs"] == [5]
        assert json.loads((tmp_path / "policy.epoch5.json").read_text())["epoch"] == 5
        assert list(tmp_path.glob("policy.epoch*.pth")) == [tmp_path / "policy.epoch5.pth"]
        events = [json.loads(line) for line in config.log_path.read_text().splitlines()]
        snapshots = [e for e in events if e["event"] == "checkpoint_retained"]
        assert len(snapshots) == 1 and snapshots[0]["epoch"] == 5
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize("epochs", [[0], [-1], [1.5], [True]])
def test_retained_epochs_reject_invalid_numbers(epochs):
    with pytest.raises(ValueError, match="retain_checkpoint_epochs"):
        tiny_config(retain_checkpoint_epochs=epochs)


def test_full_volume_recipe_runs_fifteen_epochs_and_preserves_epoch_five():
    recipe = json.loads(Path("scripts/configs/block_smb_full_volume_revision2.json").read_text())
    assert recipe["epochs"] == 15
    assert recipe["retain_checkpoint_epochs"] == [5]
    assert recipe["save_checkpoints"]
