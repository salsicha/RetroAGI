"""CPU smoke tests for Full SMB training paths without local ROM content."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi.stages.full_smb import (
    FullSMBArtifactLayout,
    FullSMBObservationConfig,
    FullSMBPlayConfig,
    FullSMBStage,
    FullSMBTrainingConfig,
    evaluate_full_smb_policy,
    load_full_smb_policy_checkpoint,
    play_full_smb_policy,
    train_full_smb_policy,
)
from scripts.tests.test_full_smb_transfer import (
    TinyFullSMBEnv,
    write_full_smb_vision_checkpoint,
)


def _tiny_full_smb_stage(vision):
    return FullSMBStage(
        env=TinyFullSMBEnv(),
        vision=vision,
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=(16, 20),
        ),
    )


class TestFullSMBCPUSmoke(unittest.TestCase):
    def test_tiny_backend_trains_evaluates_records_and_plays_on_cpu(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            layout = FullSMBArtifactLayout(
                "ci_cpu_smoke",
                root=tmp / "artifacts" / "full_smb",
            )
            layout.ensure_directories()
            files = layout.files()
            vision_checkpoint = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(vision_checkpoint)

            train_config = FullSMBTrainingConfig(
                seed=23,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                updates_per_epoch=1,
                rollout_length=1,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                deterministic_actions=True,
                device="cpu",
                full_smb_vision_checkpoint=vision_checkpoint,
                checkpoint_path=files["policy_checkpoint"],
                save_checkpoints=True,
                output_summary=files["train_summary"],
                log_path=files["train_log"],
            )

            train_result = train_full_smb_policy(
                train_config,
                make_stage=_tiny_full_smb_stage,
            )
            model, _optimizer, checkpoint = load_full_smb_policy_checkpoint(
                files["policy_checkpoint"],
                device="cpu",
            )

            eval_config = FullSMBTrainingConfig(
                seed=23,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                evaluation_episodes=1,
                evaluation_max_steps=1,
                device="cpu",
                full_smb_vision_checkpoint=vision_checkpoint,
            )
            evaluation = evaluate_full_smb_policy(
                model,
                config=eval_config,
                make_stage=_tiny_full_smb_stage,
            )

            record_config = FullSMBTrainingConfig(
                seed=23,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                evaluation_episodes=1,
                evaluation_max_steps=1,
                device="cpu",
                full_smb_vision_checkpoint=vision_checkpoint,
                recording_dir=layout.recordings_dir,
                recording_path=files["recording_manifest"],
            )
            recording = evaluate_full_smb_policy(
                model,
                config=record_config,
                make_stage=_tiny_full_smb_stage,
            )

            play_config = FullSMBPlayConfig(
                max_steps=1,
                render=False,
                fps=0.0,
                interactive_controls=False,
                deterministic_policy=True,
                recording_prefix="play",
            )
            play_runtime_config = FullSMBTrainingConfig(
                seed=23,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                device="cpu",
                full_smb_vision_checkpoint=vision_checkpoint,
                recording_dir=layout.recordings_dir,
                recording_path=files["play_manifest"],
            )
            playback = play_full_smb_policy(
                model,
                config=play_runtime_config,
                play_config=play_config,
                make_stage=_tiny_full_smb_stage,
            )

            train_events = [
                json.loads(line)
                for line in files["train_log"].read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(train_result.checkpoint_path, files["policy_checkpoint"])
            self.assertTrue(files["policy_checkpoint"].exists())
            self.assertTrue(files["train_summary"].exists())
            self.assertTrue(files["train_log"].exists())
            self.assertEqual(train_result.rollouts[0].step_count, 1)
            self.assertEqual(train_result.checkpoint["config"]["device"], "cpu")
            self.assertTrue(
                train_result.checkpoint["config"]["backend"]["env_class"].endswith("TinyFullSMBEnv")
            )
            self.assertEqual(checkpoint["config"]["device"], "cpu")
            self.assertEqual(
                [event["event"] for event in train_events],
                ["run_started", "train_rollout", "run_finished"],
            )

            self.assertEqual(evaluation.episodes, 1)
            self.assertEqual(evaluation.steps, 1)
            self.assertEqual(recording.recording["artifact_count"], 1)
            self.assertTrue(Path(recording.recording["manifest_path"]).exists())
            self.assertTrue(Path(recording.recording["artifacts"][0]["path"]).exists())
            self.assertEqual(playback.steps, 1)
            self.assertEqual(playback.recording["artifact_count"], 1)
            self.assertTrue(Path(playback.recording["manifest_path"]).exists())


if __name__ == "__main__":
    unittest.main()
