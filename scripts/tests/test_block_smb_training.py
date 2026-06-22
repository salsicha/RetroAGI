"""Tests for Block SMB trainer plumbing."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from retroagi.core import VisionOutput, VisionSpec, load_checkpoint
from retroagi.stages.block_smb import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
    BLOCK_SMB_SPEC,
    BlockSMBStage,
    BlockSMBTrainingConfig,
    SequentialBlockSMBVectorEnv,
    build_curriculum,
    restore_block_smb_checkpoint,
    train_and_evaluate_block_smb,
)
from retroagi.stages.block_smb.train import collect_trajectory, make_block_smb_model


class StaticBlockVision:
    spec = VisionSpec(
        name="static_block_trainer",
        semantic_classes=(
            "background",
            "mario",
            "platform",
            "coin",
            "goal",
            "enemy",
            "moving_platform",
        ),
        token_dim=4,
    )

    def encode(self, observation):
        logits = torch.full((1, self.spec.num_classes, 2, 16), -8.0)
        logits[:, 1, :, 1] = 8.0
        logits[:, 2, :, :] = torch.maximum(logits[:, 2, :, :], torch.tensor(1.0))
        return VisionOutput(
            position=torch.tensor([[0.1, 0.8]], dtype=torch.float32),
            semantic_logits=logits,
            semantic_ids=logits.argmax(dim=1),
            tokens=torch.zeros(1, 240, self.spec.token_dim),
            metadata={},
        )


def static_vision_factory():
    return StaticBlockVision()


def tiny_config(**overrides):
    values = dict(
        seed=7,
        epochs=1,
        episodes_per_epoch=1,
        rollout_steps=2,
        hidden_dim=8,
        evaluation_episodes=1,
        evaluation_max_steps=2,
        fixed_scenarios=("level_1_flat.json",),
        generated_scenarios=1,
        device="cpu",
    )
    values.update(overrides)
    return BlockSMBTrainingConfig(**values)


class TestBlockSMBTraining(unittest.TestCase):
    def test_curriculum_and_sequential_vector_env_are_deterministic(self):
        config = tiny_config(generated_scenarios=2)
        curriculum = build_curriculum(config)
        self.assertEqual(
            [name for name, _scenario in curriculum],
            ["level_1_flat.json", "generated_000", "generated_001"],
        )

        vector_env = SequentialBlockSMBVectorEnv(curriculum, num_envs=2)
        try:
            resets = vector_env.reset(seed=11)
            self.assertEqual(len(resets), 2)
            steps = vector_env.step([0, 1])
            self.assertEqual(len(steps), 2)
            for observation, reward, terminated, truncated, info in steps:
                self.assertEqual(observation.shape, (240, 256, 3))
                self.assertIsInstance(float(reward), float)
                self.assertIsInstance(terminated, bool)
                self.assertIsInstance(truncated, bool)
                self.assertIn("state_vec", info)
        finally:
            vector_env.close()

    def test_collect_trajectory_records_episode_masks(self):
        config = tiny_config(generated_scenarios=0)
        model = make_block_smb_model(config)
        scenario_name, scenario = build_curriculum(config)[0]
        stage = BlockSMBStage(scenario=scenario, vision=StaticBlockVision())
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                scenario_name,
                rollout_steps=2,
                seed=3,
                deterministic=True,
                device=torch.device("cpu"),
                record_frames=True,
            )
        finally:
            stage.env.close()

        self.assertGreaterEqual(len(trajectory.transitions), 1)
        self.assertEqual(len(trajectory.frames), len(trajectory.transitions) + 1)
        for step in trajectory.transitions:
            self.assertIn(step.episode_mask, (0.0, 1.0))
            self.assertEqual(step.batch.src_c.shape, (1, BLOCK_SMB_SPEC.seq_len_c))

    def test_train_evaluate_checkpoint_and_recording_smoke(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            checkpoint = tmp / "block_smb.pth"
            video_dir = tmp / "videos"
            config = tiny_config(
                checkpoint_path=checkpoint,
                save_checkpoints=True,
                video_dir=video_dir,
                record_videos=True,
            )

            result = train_and_evaluate_block_smb(
                config, vision_factory=static_vision_factory
            )

            saved = load_checkpoint(checkpoint)
            self.assertEqual(saved["stage"], BLOCK_SMB_SPEC.name)
            self.assertEqual(saved["model_name"], BLOCK_SMB_MODEL_NAME)
            self.assertEqual(saved["checkpoint_kind"], BLOCK_SMB_CHECKPOINT_KIND)
            self.assertEqual(saved["epoch"], 1)
            self.assertEqual(saved["global_step"], 1)
            evaluation = result["evaluation"]
            self.assertIn("level_1_flat.json", evaluation["fixed_scenarios"])
            self.assertFalse(evaluation["success_thresholds_met"])
            level_result = evaluation["fixed_scenarios"]["level_1_flat.json"]
            self.assertIn("threshold", level_result)
            self.assertIn("threshold_diagnostics", level_result)
            self.assertFalse(level_result["threshold_met"])
            self.assertTrue((video_dir / "level_1_flat.json_episode0.npz").exists())
            for key in (
                "loss_actor_pass1",
                "loss_actor_pass2",
                "loss_world_model",
                "loss_critic",
                "loss_total",
                "gradient_norm",
            ):
                self.assertTrue(torch.isfinite(torch.tensor(result["metrics"][key])).item())

            resumed_config = tiny_config(
                epochs=2,
                resume_path=checkpoint,
                checkpoint_path=checkpoint,
                save_checkpoints=True,
            )
            resumed = train_and_evaluate_block_smb(
                resumed_config, vision_factory=static_vision_factory
            )
            resumed_checkpoint = load_checkpoint(checkpoint)
            self.assertEqual(resumed_checkpoint["epoch"], 2)
            self.assertEqual(resumed_checkpoint["global_step"], 2)
            self.assertEqual(len(resumed["history"]), 1)

            model = make_block_smb_model(config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
            restored = restore_block_smb_checkpoint(checkpoint, model, optimizer)
            self.assertEqual(restored["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
