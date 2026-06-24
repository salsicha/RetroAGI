"""Focused Full SMB policy checkpoint, resume, and playback contract tests."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

import retroagi.stages.full_smb.train as full_smb_train_module
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SMB_ACTIONS,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.full_smb import (
    FULL_SMB_POLICY_CHECKPOINT_KIND,
    FULL_SMB_POLICY_MODEL_NAME,
    FULL_SMB_SPEC,
    FullSMBObservationConfig,
    FullSMBPlayResult,
    FullSMBStage,
    FullSMBTrainingConfig,
    evaluate_full_smb_policy,
    load_full_smb_policy_checkpoint,
    train_full_smb_policy,
)
from retroagi.stages.full_smb.transfer import FULL_SMB_TRANSFER_CHECKPOINT_KIND
from scripts.tests.test_full_smb_transfer import (
    TinyFullSMBEnv,
    write_block_policy_checkpoint,
    write_full_smb_vision_checkpoint,
)


def _tiny_stage(vision):
    return FullSMBStage(
        env=TinyFullSMBEnv(),
        vision=vision,
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=(16, 20),
        ),
    )


def _tiny_train_config(full_vision_path: Path, **overrides):
    values = dict(
        seed=61,
        architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
        epochs=1,
        updates_per_epoch=1,
        rollout_length=1,
        evaluation_episodes=0,
        evaluation_max_steps=0,
        deterministic_actions=True,
        device="cpu",
        full_smb_vision_checkpoint=full_vision_path,
    )
    values.update(overrides)
    return FullSMBTrainingConfig(**values)


class TestFullSMBPolicyContracts(unittest.TestCase):
    def test_policy_loader_rejects_incompatible_checkpoint_identity_before_state_loading(self):
        invalid_cases = (
            (
                "stage",
                {"stage": BLOCK_SMB_SPEC.name},
                "checkpoint stage does not match Full SMB",
            ),
            (
                "model_name",
                {"model_name": "wrong_full_smb_model"},
                "checkpoint model does not match Full SMB policy",
            ),
            (
                "checkpoint_kind",
                {"checkpoint_kind": "wrong_checkpoint_kind"},
                "checkpoint kind does not match Full SMB policy",
            ),
        )
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            for name, identity_override, message in invalid_cases:
                checkpoint_path = tmp / f"{name}.pth"
                identity = {
                    "stage": FULL_SMB_SPEC.name,
                    "model_name": FULL_SMB_POLICY_MODEL_NAME,
                    "checkpoint_kind": FULL_SMB_POLICY_CHECKPOINT_KIND,
                }
                identity.update(identity_override)
                checkpoint = build_checkpoint(
                    states={"model": {"unexpected.weight": torch.zeros(1)}},
                    config={
                        "architecture_name": BASELINE_ARCHITECTURE_NAME,
                        "architecture_config": {
                            "hidden_dim": 8,
                            "controller_schedule": "linear",
                        },
                    },
                    **identity,
                )
                save_checkpoint(checkpoint_path, checkpoint)

                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, message):
                        load_full_smb_policy_checkpoint(checkpoint_path, device="cpu")

    def test_resume_rejects_checkpoint_missing_rng_state(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            policy_path = tmp / "policy.pth"
            missing_rng_path = tmp / "policy_missing_rng.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            train_full_smb_policy(
                _tiny_train_config(
                    full_vision_path,
                    checkpoint_path=policy_path,
                    save_checkpoints=True,
                ),
                make_stage=_tiny_stage,
            )
            checkpoint = load_checkpoint(policy_path)
            checkpoint["states"].pop("numpy_rng")
            save_checkpoint(missing_rng_path, checkpoint)

            with self.assertRaisesRegex(ValueError, "missing RNG state keys: numpy_rng"):
                train_full_smb_policy(
                    _tiny_train_config(
                        full_vision_path,
                        epochs=2,
                        resume_path=missing_rng_path,
                    ),
                    make_stage=_tiny_stage,
                )

    def test_block_policy_checkpoint_transfers_into_full_smb_training(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            block_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            write_block_policy_checkpoint(block_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)

            result = train_full_smb_policy(
                _tiny_train_config(
                    full_vision_path,
                    init_checkpoint=block_policy_path,
                ),
                make_stage=_tiny_stage,
            )

        source = result.checkpoint["config"]["training_source"]
        self.assertEqual(source["mode"], "init_checkpoint")
        self.assertEqual(source["init_checkpoint_source"], "block_smb_policy_checkpoint")
        self.assertEqual(source["checkpoint_path"], str(block_policy_path))
        self.assertEqual(source["checkpoint_stage"], BLOCK_SMB_SPEC.name)
        self.assertEqual(
            source["resolved_transfer_checkpoint_kind"],
            FULL_SMB_TRANSFER_CHECKPOINT_KIND,
        )
        self.assertEqual(result.checkpoint["checkpoint_kind"], FULL_SMB_POLICY_CHECKPOINT_KIND)

    def test_evaluation_reports_failing_fixed_task_threshold_diagnostics(self):
        class FailingBenchmarkEnv(TinyFullSMBEnv):
            def reset(self, seed=None):
                observation, info = super().reset(seed=seed)
                info.update(
                    {
                        "level": "Level1-1",
                        "position": (0.0, 96.0),
                        "x_pos": 0.0,
                    }
                )
                return observation, info

            def step(self, action):
                del action
                self.step_count += 1
                return (
                    self._observation(0),
                    -1.0,
                    True,
                    False,
                    {
                        "level": "Level1-1",
                        "position": (0.0, 96.0),
                        "x_pos": 0.0,
                        "y_pos": 96,
                        "score": 0,
                        "coins": 0,
                        "lives": 2,
                        "death": True,
                        "level_complete": False,
                    },
                )

        def benchmark_stage(vision):
            return FullSMBStage(
                env=FailingBenchmarkEnv(),
                vision=vision,
                observation_config=FullSMBObservationConfig(
                    frame_skip=1,
                    frame_stack=2,
                    resize_shape=(16, 20),
                ),
            )

        with TemporaryDirectory() as tmpdir:
            full_vision_path = Path(tmpdir) / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            model = full_smb_train_module.make_full_smb_policy_model(
                architecture_name=BASELINE_ARCHITECTURE_NAME,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
            )
            evaluation = evaluate_full_smb_policy(
                model,
                config=FullSMBTrainingConfig(
                    seed=67,
                    architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                    evaluation_episodes=3,
                    evaluation_max_steps=2400,
                    device="cpu",
                    full_smb_vision_checkpoint=full_vision_path,
                ),
                make_stage=benchmark_stage,
            )

        task = evaluation.fixed_task_results["benchmark_1_1_start"]
        diagnostics = task["threshold_diagnostics"]
        self.assertFalse(evaluation.success_thresholds_met)
        self.assertFalse(task["threshold_met"])
        self.assertFalse(diagnostics["meets_progress"])
        self.assertFalse(diagnostics["meets_completion"])
        self.assertFalse(diagnostics["meets_survival"])
        self.assertFalse(diagnostics["meets_score"])
        self.assertFalse(diagnostics["within_death_budget"])
        self.assertTrue(diagnostics["enough_episodes"])
        self.assertEqual(task["death_count"], 3.0)
        self.assertEqual(evaluation.tuning_metrics["threshold_pass_rate"], 0.0)

    def test_play_cli_loads_saved_policy_checkpoint_before_playback(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            policy_path = tmp / "policy.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            train_full_smb_policy(
                _tiny_train_config(
                    full_vision_path,
                    checkpoint_path=policy_path,
                    save_checkpoints=True,
                ),
                make_stage=_tiny_stage,
            )
            playback = FullSMBPlayResult(
                steps=1,
                resets=1,
                completed_episodes=0,
                total_return=0.0,
                episode_returns=(0.0,),
                actions=(int(SMB_ACTIONS[0]),),
                action_names=("NOOP",),
                deterministic_policy=True,
                sampling_temperature=1.0,
                render=False,
                fps=0.0,
            )

            with (
                patch.object(
                    full_smb_train_module,
                    "play_full_smb_policy",
                    return_value=playback,
                ) as play,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                exit_code = full_smb_train_module.main(
                    [
                        "play",
                        "--checkpoint",
                        str(policy_path),
                        "--full-smb-vision-checkpoint",
                        str(full_vision_path),
                        "--device",
                        "cpu",
                        "--steps",
                        "1",
                        "--no-render",
                        "--fps",
                        "0",
                    ]
                )

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["steps"], 1)
        model = play.call_args.args[0]
        config = play.call_args.kwargs["config"]
        play_config = play.call_args.kwargs["play_config"]
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(config.architecture_name, BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            config.architecture_config,
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        self.assertEqual(config.full_smb_vision_checkpoint, full_vision_path)
        self.assertEqual(play_config.max_steps, 1)
        self.assertFalse(play_config.render)


if __name__ == "__main__":
    unittest.main()
