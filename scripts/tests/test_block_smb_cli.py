"""The Block SMB command line: train one layer, or examine a saved agent."""

import dataclasses
from unittest.mock import patch

from retroagi.stages.block_smb import cli
from retroagi.stages.block_smb.layered_train import LayeredTrainConfig


def test_only_the_layer_commands_remain():
    subcommands = cli.build_parser()._subparsers._group_actions[0].choices
    assert set(subcommands) == {"train-layer", "exam-layer"}


def test_train_layer_passes_every_given_setting_to_the_trainer():
    seen = {}

    def train_layer(config):
        seen["config"] = config
        return {"best_validation_success": 0.5}

    with patch("retroagi.stages.block_smb.layered_train.train_layer", train_layer):
        result = cli.run(
            cli.build_parser().parse_args(
                [
                    "train-layer",
                    "--learner",
                    "skill",
                    "--init",
                    "runs/action/passed.pt",
                    "--families",
                    "flat_run",
                    "single_gap",
                    "--workers",
                    "3",
                ]
            )
        )
    config = seen["config"]
    assert result == {"best_validation_success": 0.5}
    assert config.learner == "skill"
    assert config.init == "runs/action/passed.pt"
    assert config.families == ("flat_run", "single_gap")
    assert config.workers == 3
    # Settings not given keep the trainer's defaults.
    defaults = LayeredTrainConfig()
    for setting in dataclasses.fields(LayeredTrainConfig):
        if setting.name not in ("learner", "init", "families", "workers"):
            assert getattr(config, setting.name) == getattr(defaults, setting.name), setting.name


def test_exam_layer_examines_the_whole_agent_by_default():
    seen = {}

    def examine_layer(checkpoint, learner, **settings):
        seen.update(checkpoint=checkpoint, learner=learner, **settings)
        return {"wins": 1.0}

    with patch("retroagi.stages.block_smb.layered_train.examine_layer", examine_layer):
        result = cli.run(cli.build_parser().parse_args(["exam-layer", "--checkpoint", "a.pt"]))
    assert result == {"wins": 1.0}
    assert seen["checkpoint"] == "a.pt"
    assert seen["learner"] is None
