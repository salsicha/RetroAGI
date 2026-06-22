"""Tests for transferring Block SMB checkpoints into Full SMB."""

import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from retroagi.core import build_checkpoint, save_checkpoint
from retroagi.stages.block_smb import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
    BLOCK_SMB_SPEC,
    BlockSMBTrainingConfig,
)
from retroagi.stages.block_smb.train import make_block_smb_model, save_block_smb_checkpoint
from retroagi.stages.full_smb import (
    FULL_SMB_SPEC,
    FullSMBObservationConfig,
    FullSMBStage,
    FullSMBVisionTransformer,
    build_full_smb_vit_checkpoint,
)
from retroagi.stages.full_smb.transfer import (
    FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    FULL_SMB_TRANSFER_MODEL_NAME,
    load_transferred_full_smb_policy,
    select_transferred_full_smb_action,
    transfer_block_smb_checkpoint_to_full_smb,
)
from retroagi.stages.full_smb.compare import (
    FullSMBPolicyComparisonConfig,
    compare_transferred_checkpoint_with_scratch,
    save_full_smb_policy_comparison,
)


class TinyFullSMBEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.closed = False
        self.seed = 0
        self.step_count = 0

    def reset(self, seed=None):
        self.seed = 0 if seed is None else int(seed)
        self.step_count = 0
        return self._observation(0), {
            "x_pos": self.seed,
            "y_pos": 96,
            "score": 0,
            "coins": 0,
            "lives": 3,
        }

    def step(self, action):
        button_vector = np.asarray(action, dtype=np.int8)
        action_value = int(np.flatnonzero(button_vector).sum())
        self.step_count += 1
        terminated = self.step_count >= 8
        return (
            self._observation(action_value),
            float(action_value + self.step_count),
            terminated,
            False,
            {
                "x_pos": self.seed + self.step_count,
                "y_pos": 96,
                "score": self.step_count * 10,
                "coins": self.step_count % 3,
                "lives": 3,
                "level_complete": terminated,
            },
        )

    def close(self):
        self.closed = True

    def _observation(self, action_value):
        base = np.arange(16 * 20, dtype=np.uint16).reshape(16, 20)
        return np.stack(
            (
                (base + self.seed + self.step_count + action_value) % 256,
                (base * 2 + self.seed + self.step_count) % 256,
                (base * 3 + self.seed + action_value) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)


def tiny_block_config(**overrides):
    values = dict(
        seed=3,
        epochs=1,
        episodes_per_epoch=1,
        rollout_steps=1,
        hidden_dim=8,
        controller_schedule="linear",
        fixed_scenarios=("level_1_flat.json",),
        evaluation_episodes=1,
        evaluation_max_steps=1,
        device="cpu",
    )
    values.update(overrides)
    return BlockSMBTrainingConfig(**values)


def write_full_smb_vision_checkpoint(path: Path) -> None:
    model = FullSMBVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
    checkpoint = build_full_smb_vit_checkpoint(
        model,
        epoch=1,
        metrics={"mean_iou": 1.0},
        config={
            "model": {
                "hidden_dim": 16,
                "depth": 1,
                "heads": 4,
                "patch_size": 16,
                "dropout": 0.0,
            }
        },
    )
    save_checkpoint(path, checkpoint)


def write_block_policy_checkpoint(path: Path):
    config = tiny_block_config()
    model = make_block_smb_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_(0.01 * (index + 1))
    save_block_smb_checkpoint(
        path,
        model,
        optimizer,
        epoch=2,
        global_step=5,
        config=config,
        metrics={"loss_total": 0.25},
    )
    return model, config


class TestFullSMBTransfer(unittest.TestCase):
    def test_block_policy_transfers_to_full_smb_checkpoint_and_stage_batch(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            output_path = tmp / "full_smb_transfer.pth"
            source_model, source_config = write_block_policy_checkpoint(
                source_policy_path
            )
            write_full_smb_vision_checkpoint(full_vision_path)

            result = transfer_block_smb_checkpoint_to_full_smb(
                source_policy_path,
                output_checkpoint=output_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
            )
            loaded = load_transferred_full_smb_policy(
                output_path,
                full_smb_vision_checkpoint=full_vision_path,
                device="cpu",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(result.checkpoint["stage"], FULL_SMB_SPEC.name)
            self.assertEqual(
                result.checkpoint["model_name"], FULL_SMB_TRANSFER_MODEL_NAME
            )
            self.assertEqual(
                result.checkpoint["checkpoint_kind"],
                FULL_SMB_TRANSFER_CHECKPOINT_KIND,
            )
            self.assertEqual(result.checkpoint["epoch"], 2)
            self.assertEqual(result.checkpoint["global_step"], 5)
            self.assertEqual(
                result.checkpoint["config"]["model"]["controller_schedule"],
                source_config.controller_schedule,
            )
            self.assertEqual(result.full_smb_vision_path, full_vision_path)
            self.assertEqual(loaded.output_path, output_path)

            source_state = source_model.state_dict()
            transferred_state = result.model.state_dict()
            for key, source_value in source_state.items():
                torch.testing.assert_close(transferred_state[key], source_value)

            stage = FullSMBStage(
                env=TinyFullSMBEnv(),
                vision=result.vision,
                observation_config=FullSMBObservationConfig(
                    frame_skip=1,
                    frame_stack=2,
                    resize_shape=(16, 20),
                ),
            )
            try:
                observation = stage.reset(seed=4)
                batch = stage.encode_observation(observation)
                selection = select_transferred_full_smb_action(
                    result.model,
                    batch,
                    deterministic=True,
                    device="cpu",
                )
            finally:
                stage.close()

            self.assertGreaterEqual(selection.action, 0)
            self.assertLess(selection.action, 6)
            self.assertEqual(selection.logits.shape, (1, 6))
            self.assertTrue(stage.env.closed)

    def test_transfer_rejects_non_block_policy_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "wrong_stage.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            model = make_block_smb_model(tiny_block_config())
            write_full_smb_vision_checkpoint(full_vision_path)
            checkpoint = build_checkpoint(
                stage="full_smb",
                model_name=BLOCK_SMB_MODEL_NAME,
                checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
                states={"model": model.state_dict()},
                specs={
                    "stage": {
                        "name": BLOCK_SMB_SPEC.name,
                        "seq_len_a": BLOCK_SMB_SPEC.seq_len_a,
                        "seq_len_b": BLOCK_SMB_SPEC.seq_len_b,
                        "seq_len_c": BLOCK_SMB_SPEC.seq_len_c,
                        "ratio_bc": BLOCK_SMB_SPEC.ratio_bc,
                        "vocab_size": BLOCK_SMB_SPEC.vocab_size,
                    }
                },
            )
            save_checkpoint(source_policy_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "source policy stage"):
                transfer_block_smb_checkpoint_to_full_smb(
                    source_policy_path,
                    full_smb_vision_checkpoint=full_vision_path,
                    block_vision_checkpoint=None,
                    device="cpu",
                )

    def test_compares_transferred_policy_with_scratch_baseline(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            source_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            transfer_path = tmp / "full_smb_transfer.pth"
            report_path = tmp / "comparison.json"
            write_block_policy_checkpoint(source_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)
            transfer_block_smb_checkpoint_to_full_smb(
                source_policy_path,
                output_checkpoint=transfer_path,
                full_smb_vision_checkpoint=full_vision_path,
                block_vision_checkpoint=None,
                device="cpu",
            )

            result = compare_transferred_checkpoint_with_scratch(
                transfer_path,
                make_stage=lambda vision: FullSMBStage(
                    env=TinyFullSMBEnv(),
                    vision=vision,
                    observation_config=FullSMBObservationConfig(
                        frame_skip=1,
                        frame_stack=2,
                        resize_shape=(16, 20),
                    ),
                ),
                full_smb_vision_checkpoint=full_vision_path,
                config=FullSMBPolicyComparisonConfig(
                    steps=4,
                    seed=9,
                    scratch_seed=99,
                    device="cpu",
                ),
            )
            save_full_smb_policy_comparison(report_path, result)
            saved = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.requested_steps, 4)
        self.assertEqual(result.evaluated_steps, 4)
        self.assertEqual(sum(result.transfer_action_histogram.values()), 4)
        self.assertEqual(sum(result.scratch_action_histogram.values()), 4)
        self.assertGreaterEqual(result.action_agreement, 0.0)
        self.assertLessEqual(result.action_agreement, 1.0)
        self.assertEqual(result.scratch_source, "scratch_initialization")
        self.assertEqual(saved["evaluated_steps"], 4)
        self.assertEqual(saved["scratch_seed"], 99)


if __name__ == "__main__":
    unittest.main()
