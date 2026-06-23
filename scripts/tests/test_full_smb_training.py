"""Tests for direct Full SMB policy training and checkpointing."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from retroagi.core import load_checkpoint
from retroagi.stages.full_smb import (
    FULL_SMB_POLICY_CHECKPOINT_KIND,
    FULL_SMB_POLICY_MODEL_NAME,
    FullSMBObservationConfig,
    FullSMBStage,
    FullSMBTrainingConfig,
    evaluate_full_smb_policy,
    load_full_smb_policy_checkpoint,
    train_full_smb_policy,
)
from retroagi.stages.full_smb.transfer import transfer_block_smb_checkpoint_to_full_smb
from scripts.tests.test_full_smb_transfer import (
    TinyFullSMBEnv,
    write_block_policy_checkpoint,
    write_full_smb_vision_checkpoint,
)


def tiny_stage(vision):
    return FullSMBStage(
        env=TinyFullSMBEnv(),
        vision=vision,
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=(16, 20),
        ),
    )


class TestFullSMBTraining(unittest.TestCase):
    def test_train_resume_and_evaluate_full_smb_policy_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            block_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            transfer_path = tmp / "full_smb_transfer.pth"
            policy_path = tmp / "full_smb_policy.pth"
            resumed_path = tmp / "full_smb_policy_resumed.pth"
            write_block_policy_checkpoint(block_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)
            transfer_block_smb_checkpoint_to_full_smb(
                block_policy_path,
                output_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
            )

            config = FullSMBTrainingConfig(
                seed=7,
                epochs=1,
                episodes_per_epoch=1,
                max_steps_per_episode=2,
                evaluation_episodes=1,
                evaluation_max_steps=2,
                device="cpu",
                init_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                checkpoint_path=policy_path,
                save_checkpoints=True,
            )
            result = train_full_smb_policy(config, make_stage=tiny_stage)
            model, _optimizer, checkpoint = load_full_smb_policy_checkpoint(
                policy_path,
                device="cpu",
            )
            evaluation = evaluate_full_smb_policy(
                model,
                config=config,
                make_stage=tiny_stage,
            )

            self.assertTrue(policy_path.exists())
            self.assertEqual(result.checkpoint["model_name"], FULL_SMB_POLICY_MODEL_NAME)
            self.assertEqual(
                result.checkpoint["checkpoint_kind"],
                FULL_SMB_POLICY_CHECKPOINT_KIND,
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertGreater(checkpoint["global_step"], 0)
            self.assertGreaterEqual(evaluation.steps, 1)
            self.assertIn("optimizer", checkpoint["states"])

            resume_config = FullSMBTrainingConfig(
                seed=7,
                epochs=2,
                episodes_per_epoch=1,
                max_steps_per_episode=2,
                evaluation_episodes=1,
                evaluation_max_steps=2,
                device="cpu",
                resume_path=policy_path,
                full_smb_vision_checkpoint=full_vision_path,
                checkpoint_path=resumed_path,
                save_checkpoints=True,
            )
            resumed = train_full_smb_policy(resume_config, make_stage=tiny_stage)
            resumed_checkpoint = load_checkpoint(resumed_path)

        self.assertEqual(resumed.checkpoint["epoch"], 2)
        self.assertEqual(resumed_checkpoint["epoch"], 2)
        self.assertGreater(
            resumed_checkpoint["global_step"],
            checkpoint["global_step"],
        )

    def test_training_changes_policy_weights(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=11,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                episodes_per_epoch=1,
                max_steps_per_episode=2,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
            )
            before_path = _write_checkpoint_for_comparison(tmp / "before.pth", config)
            before, _optimizer, _checkpoint = load_full_smb_policy_checkpoint(
                before_path,
                device="cpu",
            )
            result = train_full_smb_policy(config, make_stage=tiny_stage)
            after_state = result.checkpoint["states"]["model"]

        changed = any(
            not torch.equal(value, after_state[name]) for name, value in before.state_dict().items()
        )
        self.assertTrue(changed)


def _write_checkpoint_for_comparison(path: Path, config: FullSMBTrainingConfig) -> Path:
    result = train_full_smb_policy(
        FullSMBTrainingConfig(
            **{
                **config.__dict__,
                "epochs": 0,
                "episodes_per_epoch": 0,
                "evaluation_episodes": 0,
                "evaluation_max_steps": 0,
                "checkpoint_path": path,
                "save_checkpoints": True,
            }
        ),
        make_stage=tiny_stage,
    )
    assert result.checkpoint_path == path
    return path


if __name__ == "__main__":
    unittest.main()
