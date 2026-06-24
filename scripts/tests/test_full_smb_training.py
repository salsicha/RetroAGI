"""Tests for direct Full SMB policy training and checkpointing."""

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

import retroagi.stages.full_smb.train as full_smb_train_module
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    StageBatch,
    WorldModelState,
    load_checkpoint,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.block_smb.train import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
)
from retroagi.stages.full_smb import (
    FULL_SMB_PERCEPTION_FINE_TUNE,
    FULL_SMB_PERCEPTION_FREEZE,
    FULL_SMB_PERCEPTION_REPLACE,
    FULL_SMB_POLICY_CHECKPOINT_KIND,
    FULL_SMB_POLICY_MODEL_NAME,
    FULL_SMB_SPEC,
    FullSMBObservationConfig,
    FullSMBRewardConfig,
    FullSMBStage,
    FullSMBTrainingConfig,
    evaluate_full_smb_policy,
    load_full_smb_policy_checkpoint,
    train_full_smb_policy,
)
from retroagi.stages.full_smb.transfer import (
    FULL_SMB_TRANSFER_CHECKPOINT_KIND,
    FULL_SMB_TRANSFER_MODEL_NAME,
    transfer_block_smb_checkpoint_to_full_smb,
)
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
    def test_scratch_training_uses_shared_architecture_factory_and_stage_batches(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=5,
                architecture_name=BASELINE_ARCHITECTURE_NAME,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                updates_per_epoch=1,
                rollout_length=2,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
            )

            with patch.object(
                full_smb_train_module,
                "make_full_smb_policy_model",
                wraps=full_smb_train_module.make_full_smb_policy_model,
            ) as factory:
                result = train_full_smb_policy(config, make_stage=tiny_stage)

        factory.assert_called_once_with(
            architecture_name=BASELINE_ARCHITECTURE_NAME,
            architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
        )
        source = result.checkpoint["config"]["training_source"]
        self.assertEqual(source["mode"], "scratch")
        self.assertIsNone(source["checkpoint_path"])
        self.assertTrue(source["uses_shared_architecture_factory"])
        self.assertEqual(source["architecture_name"], BASELINE_ARCHITECTURE_NAME)
        self.assertEqual(
            source["architecture_config"],
            {"hidden_dim": 8, "controller_schedule": "linear"},
        )
        self.assertEqual(
            result.checkpoint["metadata"]["training"]["source"]["mode"],
            "scratch",
        )
        contract = result.checkpoint["metadata"]["training"]["stage_batch_contract"]
        self.assertEqual(contract["src_a"]["sequence_length"], FULL_SMB_SPEC.seq_len_a)
        self.assertEqual(contract["src_b"]["sequence_length"], FULL_SMB_SPEC.seq_len_b)
        self.assertEqual(contract["src_c"]["feature_length"], FULL_SMB_SPEC.seq_len_c)

    def test_scratch_training_rejects_non_full_smb_stage_batch(self):
        class BadStage:
            def reset(self, seed=None):
                del seed
                return object()

            def encode_observation(self, _observation):
                return StageBatch(
                    src_a=torch.zeros((1, FULL_SMB_SPEC.seq_len_a + 1), dtype=torch.long),
                    target_a=None,
                    src_b=torch.zeros((1, FULL_SMB_SPEC.seq_len_b), dtype=torch.long),
                    target_b=None,
                    src_c=torch.zeros((1, FULL_SMB_SPEC.seq_len_c), dtype=torch.float32),
                    target_c=None,
                    metadata={},
                )

            def step(self, _action):
                return object(), 0.0, True, False, {}

            def close(self):
                return None

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=6,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                updates_per_epoch=1,
                rollout_length=1,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
            )

            with self.assertRaisesRegex(ValueError, "src_a"):
                train_full_smb_policy(config, make_stage=lambda _vision: BadStage())

    def test_block_smb_init_checkpoint_transfers_and_fine_tunes_directly(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            block_policy_path = tmp / "block_policy.pth"
            full_vision_path = tmp / "full_smb_vit.pth"
            _source_model, source_config = write_block_policy_checkpoint(block_policy_path)
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=8,
                epochs=1,
                updates_per_epoch=1,
                rollout_length=2,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                init_checkpoint=block_policy_path,
                full_smb_vision_checkpoint=full_vision_path,
            )
            result = train_full_smb_policy(config, make_stage=tiny_stage)

        checkpoint = result.checkpoint
        source = checkpoint["config"]["training_source"]
        self.assertEqual(checkpoint["model_name"], FULL_SMB_POLICY_MODEL_NAME)
        self.assertEqual(source["mode"], "init_checkpoint")
        self.assertEqual(source["init_checkpoint_source"], "block_smb_policy_checkpoint")
        self.assertEqual(source["checkpoint_path"], str(block_policy_path))
        self.assertEqual(source["checkpoint_stage"], BLOCK_SMB_SPEC.name)
        self.assertEqual(source["checkpoint_model_name"], BLOCK_SMB_MODEL_NAME)
        self.assertEqual(source["checkpoint_kind"], BLOCK_SMB_CHECKPOINT_KIND)
        self.assertEqual(source["resolved_transfer_stage"], FULL_SMB_SPEC.name)
        self.assertEqual(source["resolved_transfer_model_name"], FULL_SMB_TRANSFER_MODEL_NAME)
        self.assertEqual(
            source["resolved_transfer_checkpoint_kind"],
            FULL_SMB_TRANSFER_CHECKPOINT_KIND,
        )
        self.assertEqual(source["full_smb_vision_checkpoint"], str(full_vision_path))
        self.assertEqual(source["architecture_config"], source_config.architecture_config)
        self.assertEqual(
            checkpoint["metadata"]["training"]["source"],
            source,
        )

    def test_full_smb_rollout_boundary_classifies_terminal_signals(self):
        cases = (
            (
                {"full_smb_signals": {"death": True}},
                True,
                False,
                {"terminated", "death"},
            ),
            (
                {"full_smb_signals": {"timeout": True}},
                False,
                False,
                {"timeout"},
            ),
            (
                {"full_smb_signals": {"completion": True}},
                True,
                False,
                {"terminated", "level_completion"},
            ),
            (
                {"full_smb_signals": {"game_over": True}},
                True,
                False,
                {"terminated", "game_over"},
            ),
            (
                {"manual_reset": True},
                False,
                False,
                {"manual_reset"},
            ),
        )

        for info, terminated, truncated, expected_reasons in cases:
            with self.subTest(expected_reasons=expected_reasons):
                boundary = full_smb_train_module._full_smb_rollout_boundary(
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                self.assertTrue(boundary.reset_recurrent_state)
                self.assertEqual(boundary.episode_mask, 0.0)
                self.assertLessEqual(expected_reasons, set(boundary.reasons))

        continuing = full_smb_train_module._full_smb_rollout_boundary(
            terminated=False,
            truncated=False,
            info={"full_smb_signals": {"death": False, "timeout": False}},
        )
        self.assertFalse(continuing.reset_recurrent_state)
        self.assertEqual(continuing.episode_mask, 1.0)

    def test_manual_reset_boundary_drops_carried_recurrent_state(self):
        class ManualResetEnv(TinyFullSMBEnv):
            def step(self, action):
                observation, reward, terminated, truncated, info = super().step(action)
                info["level_complete"] = False
                if self.step_count == 2:
                    info["manual_reset"] = True
                    info["termination_reason"] = "manual_reset"
                return observation, reward, False, False, info

        seen_states: list[WorldModelState | None] = []

        def fake_policy(_model, _batch, *, device, world_model_state=None):
            del device
            seen_states.append(world_model_state)
            value = float(len(seen_states))
            logits = torch.zeros(
                (1, full_smb_train_module.FULL_SMB_ACTION_COUNT),
                dtype=torch.float32,
                requires_grad=True,
            )
            next_state = WorldModelState(
                hidden=torch.full((1, 1, 8), value),
                cell=torch.full((1, 1, 8), value),
            )
            return logits, next_state

        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=9,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                updates_per_epoch=1,
                rollout_length=3,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                deterministic_actions=True,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
            )
            with patch.object(
                full_smb_train_module,
                "_policy_action_logits_and_state",
                side_effect=fake_policy,
            ):
                result = train_full_smb_policy(
                    config,
                    make_stage=lambda vision: FullSMBStage(
                        env=ManualResetEnv(),
                        vision=vision,
                        observation_config=FullSMBObservationConfig(
                            frame_skip=1,
                            frame_stack=2,
                            resize_shape=(16, 20),
                        ),
                    ),
                )

        self.assertEqual(len(seen_states), 3)
        self.assertIsNone(seen_states[0])
        self.assertIsInstance(seen_states[1], WorldModelState)
        self.assertIsNone(seen_states[2])
        self.assertEqual(result.history["episode_return"], [6.0])
        self.assertEqual(result.checkpoint["metrics"]["mean_train_return"], 6.0)
        self.assertEqual(
            result.checkpoint["metadata"]["training"]["rollout"]["recurrent_state_policy"],
            "carry_until_full_smb_boundary",
        )
        self.assertEqual(result.history["boundary_manual_reset"], [2.0])
        self.assertEqual(result.history["recurrent_state_resets"], [2.0])

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
                updates_per_epoch=1,
                rollout_length=2,
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
                result.checkpoint["config"]["training_source"]["mode"],
                "init_checkpoint",
            )
            self.assertEqual(
                result.checkpoint["config"]["training_source"]["init_checkpoint_source"],
                "full_smb_transfer_checkpoint",
            )
            self.assertEqual(
                result.checkpoint["config"]["training_source"]["checkpoint_path"],
                str(transfer_path),
            )
            self.assertEqual(
                result.checkpoint["config"]["training_source"]["checkpoint_kind"],
                FULL_SMB_TRANSFER_CHECKPOINT_KIND,
            )
            self.assertEqual(
                result.checkpoint["checkpoint_kind"],
                FULL_SMB_POLICY_CHECKPOINT_KIND,
            )
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertGreater(checkpoint["global_step"], 0)
            self.assertGreaterEqual(evaluation.steps, 1)
            self.assertIn("optimizer", checkpoint["states"])
            self.assertNotIn("perception", checkpoint["states"])
            self.assertEqual(
                checkpoint["config"]["perception"]["mode"],
                FULL_SMB_PERCEPTION_FREEZE,
            )
            self.assertEqual(checkpoint["config"]["rollout"]["rollout_length"], 2)
            self.assertEqual(checkpoint["config"]["rollout"]["updates_per_epoch"], 1)
            self.assertEqual(
                checkpoint["config"]["loss_weights"]["policy"],
                1.0,
            )
            self.assertEqual(
                checkpoint["config"]["reward"]["terms"]["emulator_progress"],
                1.0,
            )
            self.assertEqual(
                checkpoint["config"]["perception"]["checkpoint_path"],
                str(full_vision_path),
            )
            self.assertTrue(checkpoint["metadata"]["perception"]["frozen"])
            self.assertFalse(checkpoint["metadata"]["perception"]["optimizer_updates_enabled"])

            resume_config = FullSMBTrainingConfig(
                seed=7,
                epochs=2,
                updates_per_epoch=1,
                rollout_length=2,
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

    def test_training_config_normalizes_full_smb_contract_sections(self):
        config = FullSMBTrainingConfig(
            epochs=2,
            episodes_per_epoch=99,
            updates_per_epoch=3,
            max_steps_per_episode=99,
            rollout_length=4,
            vector_env_count=2,
            reward_config={"emulator_progress": 0.5, "death": -7.0},
            policy_loss_weight=0.75,
            representation_weight=0.1,
            world_model_weight=0.2,
            reward_loss_weight=0.3,
            value_loss_weight=0.4,
            action_aux_weight=0.5,
            critic_loss_weight=0.6,
            deterministic=False,
            checkpoint_path="data/full_smb/policy.pth",
            resume_path="data/full_smb/resume.pth",
            output_summary="artifacts/full_smb/summary.json",
            log_path="artifacts/full_smb/train.jsonl",
            recording_dir="artifacts/full_smb/recordings",
            recording_path="artifacts/full_smb/recording.npz",
            tracking_backend="NONE",
            tracking_log_dir="artifacts/full_smb/tracking",
            tracking_project="retroagi-test",
            tracking_run_name="full-smoke",
            tracking_mode="offline",
        )

        self.assertEqual(config.updates_per_epoch, 3)
        self.assertEqual(config.episodes_per_epoch, 3)
        self.assertEqual(config.rollout_length, 4)
        self.assertEqual(config.max_steps_per_episode, 4)
        self.assertEqual(config.vector_env_count, 2)
        self.assertIsInstance(config.reward_config, FullSMBRewardConfig)
        self.assertEqual(config.reward_config.emulator_progress, 0.5)
        self.assertEqual(config.reward_config.death, -7.0)
        self.assertEqual(config.policy_loss_weight, 0.75)
        self.assertEqual(config.value_loss_weight, 0.4)
        self.assertFalse(config.deterministic)
        self.assertEqual(config.checkpoint_path, Path("data/full_smb/policy.pth"))
        self.assertEqual(config.resume_path, Path("data/full_smb/resume.pth"))
        self.assertEqual(config.output_summary, Path("artifacts/full_smb/summary.json"))
        self.assertEqual(config.log_path, Path("artifacts/full_smb/train.jsonl"))
        self.assertEqual(config.recording_dir, Path("artifacts/full_smb/recordings"))
        self.assertEqual(config.recording_path, Path("artifacts/full_smb/recording.npz"))
        self.assertEqual(config.tracking_backend, "none")
        self.assertEqual(config.tracking_log_dir, Path("artifacts/full_smb/tracking"))
        self.assertEqual(config.tracking_project, "retroagi-test")
        self.assertEqual(config.tracking_run_name, "full-smoke")
        self.assertEqual(config.tracking_mode, "offline")

        with self.assertRaisesRegex(ValueError, "vector_env_count"):
            FullSMBTrainingConfig(vector_env_count=0)
        with self.assertRaisesRegex(ValueError, "loss weights"):
            FullSMBTrainingConfig(value_loss_weight=-0.1)
        with self.assertRaisesRegex(TypeError, "reward_config"):
            FullSMBTrainingConfig(reward_config=object())
        with self.assertRaisesRegex(ValueError, "tracking_backend"):
            FullSMBTrainingConfig(tracking_backend="unknown")
        with self.assertRaisesRegex(ValueError, "tracking_project"):
            FullSMBTrainingConfig(tracking_project="")

    def test_expanded_training_config_is_logged_and_checkpointed(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            log_path = tmp / "train.jsonl"
            recording_dir = tmp / "recordings"
            write_full_smb_vision_checkpoint(full_vision_path)
            config = FullSMBTrainingConfig(
                seed=17,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=1,
                updates_per_epoch=1,
                rollout_length=2,
                vector_env_count=3,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
                reward_config=FullSMBRewardConfig(
                    emulator_progress=0.5,
                    frame_penalty=-0.01,
                ),
                policy_loss_weight=0.5,
                value_loss_weight=0.25,
                deterministic=True,
                log_path=log_path,
                recording_dir=recording_dir,
                tracking_backend="none",
                tracking_project="retroagi-unit",
                tracking_run_name="full-config",
            )
            result = train_full_smb_policy(config, make_stage=tiny_stage)

            events = [
                json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
            ]

        checkpoint = result.checkpoint
        self.assertEqual(checkpoint["config"]["updates_per_epoch"], 1)
        self.assertEqual(checkpoint["config"]["rollout_length"], 2)
        self.assertEqual(checkpoint["config"]["vector_env_count"], 3)
        self.assertEqual(checkpoint["config"]["rollout"]["active_vector_env_count"], 1)
        self.assertFalse(checkpoint["config"]["rollout"]["vectorized_training_enabled"])
        self.assertEqual(checkpoint["config"]["loss_weights"]["policy"], 0.5)
        self.assertEqual(checkpoint["config"]["loss_weights"]["value"], 0.25)
        self.assertEqual(
            checkpoint["config"]["reward"]["terms"]["emulator_progress"],
            0.5,
        )
        self.assertEqual(
            checkpoint["config"]["recording"]["recording_dir"],
            str(recording_dir),
        )
        self.assertEqual(checkpoint["config"]["tracking"]["backend"], "none")
        self.assertEqual(checkpoint["metadata"]["training"]["deterministic"], True)
        self.assertEqual(
            checkpoint["metadata"]["training"]["rollout"]["vector_env_count"],
            3,
        )
        self.assertEqual(
            [event["event"] for event in events],
            ["run_started", "train_rollout", "run_finished"],
        )
        self.assertEqual(events[0]["config"]["rollout_length"], 2)
        self.assertEqual(events[1]["metrics"]["steps"], 2.0)

    def test_train_cli_builds_expanded_full_smb_config(self):
        with (
            patch.object(
                full_smb_train_module,
                "train_full_smb_policy",
                return_value=SimpleNamespace(as_dict=lambda: {"ok": True}),
            ) as train,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            exit_code = full_smb_train_module.main(
                [
                    "train",
                    "--seed",
                    "23",
                    "--device",
                    "cpu",
                    "--epochs",
                    "4",
                    "--updates-per-epoch",
                    "5",
                    "--rollout-steps",
                    "6",
                    "--vector-env-count",
                    "2",
                    "--learning-rate",
                    "0.0007",
                    "--policy-loss-weight",
                    "0.8",
                    "--value-loss-weight",
                    "0.2",
                    "--reward-emulator-progress",
                    "0.4",
                    "--reward-death",
                    "-9",
                    "--nondeterministic",
                    "--deterministic-actions",
                    "--checkpoint",
                    "data/full_smb/policy.pth",
                    "--recording-dir",
                    "artifacts/full_smb/recordings",
                    "--log-path",
                    "artifacts/full_smb/train.jsonl",
                    "--tracking-backend",
                    "none",
                    "--tracking-project",
                    "retroagi-cli",
                    "--tracking-run-name",
                    "full-cli",
                ]
            )

        config = train.call_args.args[0]
        self.assertEqual(exit_code, 0)
        self.assertEqual(config.seed, 23)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.epochs, 4)
        self.assertEqual(config.updates_per_epoch, 5)
        self.assertEqual(config.rollout_length, 6)
        self.assertEqual(config.vector_env_count, 2)
        self.assertEqual(config.learning_rate, 0.0007)
        self.assertEqual(config.policy_loss_weight, 0.8)
        self.assertEqual(config.value_loss_weight, 0.2)
        self.assertEqual(config.reward_config.emulator_progress, 0.4)
        self.assertEqual(config.reward_config.death, -9.0)
        self.assertFalse(config.deterministic)
        self.assertTrue(config.deterministic_actions)
        self.assertEqual(config.checkpoint_path, Path("data/full_smb/policy.pth"))
        self.assertTrue(config.save_checkpoints)
        self.assertEqual(config.recording_dir, Path("artifacts/full_smb/recordings"))
        self.assertEqual(config.log_path, Path("artifacts/full_smb/train.jsonl"))
        self.assertEqual(config.tracking_backend, "none")
        self.assertEqual(config.tracking_project, "retroagi-cli")
        self.assertEqual(config.tracking_run_name, "full-cli")

    def test_perception_modes_resolve_and_record_trainable_state(self):
        with TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            full_vision_path = tmp / "full_smb_vit.pth"
            write_full_smb_vision_checkpoint(full_vision_path)

            frozen = FullSMBTrainingConfig(full_smb_vision_checkpoint=full_vision_path)
            fine_tune = FullSMBTrainingConfig(
                full_smb_vision_checkpoint=full_vision_path,
                perception_mode="fine-tune",
            )
            legacy_fine_tune = FullSMBTrainingConfig(
                full_smb_vision_checkpoint=full_vision_path,
                freeze_vision=False,
            )
            replacement = FullSMBTrainingConfig(
                full_smb_vision_checkpoint=None,
                perception_mode=FULL_SMB_PERCEPTION_REPLACE,
            )

            self.assertEqual(frozen.perception_mode, FULL_SMB_PERCEPTION_FREEZE)
            self.assertTrue(frozen.freeze_vision)
            self.assertEqual(fine_tune.perception_mode, FULL_SMB_PERCEPTION_FINE_TUNE)
            self.assertFalse(fine_tune.freeze_vision)
            self.assertEqual(
                legacy_fine_tune.perception_mode,
                FULL_SMB_PERCEPTION_FINE_TUNE,
            )
            self.assertEqual(replacement.perception_mode, FULL_SMB_PERCEPTION_REPLACE)
            self.assertFalse(replacement.freeze_vision)

            config = FullSMBTrainingConfig(
                seed=13,
                architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
                epochs=0,
                episodes_per_epoch=0,
                evaluation_episodes=0,
                evaluation_max_steps=0,
                device="cpu",
                full_smb_vision_checkpoint=full_vision_path,
                perception_mode=FULL_SMB_PERCEPTION_FINE_TUNE,
            )
            result = train_full_smb_policy(config, make_stage=tiny_stage)

        perception = result.checkpoint["config"]["perception"]
        self.assertEqual(perception["mode"], FULL_SMB_PERCEPTION_FINE_TUNE)
        self.assertEqual(perception["checkpoint_path"], str(full_vision_path))
        self.assertFalse(perception["frozen"])
        self.assertTrue(perception["trainable"])
        self.assertTrue(perception["optimizer_updates_enabled"])
        self.assertTrue(perception["state_saved"])
        self.assertIn("perception", result.checkpoint["states"])
        self.assertIn(
            "perception",
            {
                group.get("name")
                for group in result.checkpoint["states"]["optimizer"]["param_groups"]
            },
        )

    def test_perception_mode_rejects_ambiguous_checkpointless_freeze(self):
        with self.assertRaisesRegex(ValueError, "full_smb_vision_checkpoint"):
            FullSMBTrainingConfig(
                full_smb_vision_checkpoint=None,
                perception_mode=FULL_SMB_PERCEPTION_FREEZE,
            )
        with self.assertRaisesRegex(ValueError, "perception_mode"):
            FullSMBTrainingConfig(
                full_smb_vision_checkpoint=Path("vision.pth"),
                perception_mode="mystery",
            )


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
