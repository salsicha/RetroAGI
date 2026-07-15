"""Tests for Full SMB real-emulator imitation warm starts."""

import pickle
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

import retroagi.stages.full_smb.imitation as full_smb_imitation_module
from retroagi.stages.full_smb import (
    FullSMBObstacleWindowDurationSpec,
    FullSMBObservationConfig,
    FullSMBStage,
    collect_full_smb_imitation_dataset,
    collect_full_smb_obstacle_window_duration_dataset,
    full_smb_opening_imitation_script,
    merge_full_smb_imitation_datasets,
    train_full_smb_imitation_warm_start,
)
from retroagi.stages.full_smb.save_states import (
    FULL_SMB_SAVE_STATE_KIND,
    FULL_SMB_SAVE_STATE_SCHEMA_VERSION,
)
from retroagi.stages.full_smb.transfer import make_full_smb_policy_model
from scripts.tests.test_full_smb_adapter import StaticFullSMBVision
from scripts.tests.test_full_smb_transfer import TinyFullSMBEnv


class SweepFullSMBEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self):
        self.seed = 0
        self.step_count = 0
        self.x_pos = 0.0
        self.jump_hold = 0
        self.dead = False

    def reset(self, seed=None):
        self.seed = 0 if seed is None else int(seed)
        self.step_count = 0
        self.x_pos = float(self.seed % 13)
        self.jump_hold = 0
        self.dead = False
        return self._observation(), self._info()

    def step(self, action):
        buttons = np.asarray(action, dtype=np.int8)
        self.step_count += 1
        jumping = bool(buttons[self.buttons.index("A")])
        right = bool(buttons[self.buttons.index("RIGHT")])
        if jumping:
            self.jump_hold += 1
        elif self.jump_hold:
            if self.jump_hold == 4:
                self.x_pos += 20.0
            elif self.jump_hold < 4 or self.jump_hold > 8:
                self.dead = True
            self.jump_hold = 0
        self.x_pos += 1.0 if right else 0.25
        return self._observation(), 1.0, self.dead, False, self._info()

    def get_state(self):
        return {
            "seed": self.seed,
            "step_count": self.step_count,
            "x_pos": self.x_pos,
            "jump_hold": self.jump_hold,
            "dead": self.dead,
        }

    def set_state(self, state):
        self.seed = int(state["seed"])
        self.step_count = int(state["step_count"])
        self.x_pos = float(state["x_pos"])
        self.jump_hold = int(state["jump_hold"])
        self.dead = bool(state["dead"])

    def close(self):
        return None

    def _info(self):
        return {
            "x_pos": self.x_pos,
            "y_pos": 96,
            "score": int(self.step_count * 10),
            "coins": 0,
            "lives": 0 if self.dead else 3,
            "death": self.dead,
        }

    def _observation(self):
        value = int(self.x_pos + self.step_count) % 256
        return np.full((16, 20, 3), value, dtype=np.uint8)


class TestFullSMBImitationWarmStart(unittest.TestCase):
    def test_opening_imitation_script_mixes_right_and_right_jump(self):
        script = full_smb_opening_imitation_script(320)

        self.assertEqual(len(script), 320)
        self.assertEqual(script.count(1), 280)
        self.assertEqual(script.count(2), 40)

    def test_opening_imitation_script_downsamples_frame_skip_timing(self):
        script = full_smb_opening_imitation_script(80, decision_frame_skip=4)

        self.assertEqual(len(script), 80)
        self.assertEqual(script.count(1), 70)
        self.assertEqual(script.count(2), 10)
        self.assertEqual(script[40:45], (2, 2, 2, 2, 2))
        self.assertEqual(script[45], 1)

    def test_opening_imitation_script_preserves_jump_windows_for_wide_frame_skip(self):
        script = full_smb_opening_imitation_script(12, decision_frame_skip=64)

        self.assertEqual(len(script), 12)
        self.assertGreater(script.count(2), 0)

    def test_primitive_targets_supervise_jump_run_duration_and_release(self):
        targets = full_smb_imitation_module._full_smb_imitation_primitive_targets(
            torch.tensor([1, 2, 2, 2, 1, 2, 2, 1], dtype=torch.long)
        )

        self.assertEqual(
            targets["duration_mask"].tolist(),
            [0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        )
        self.assertEqual(
            targets["release_mask"].tolist(),
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
        )
        self.assertEqual(
            targets["release"].tolist(),
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        )

    def test_obstacle_window_sweep_adds_explicit_duration_label(self):
        spec = FullSMBObstacleWindowDurationSpec(
            name="unit_first_enemy",
            save_state_artifact="section_1_1_first_enemy_approach",
            obstacle_kind="enemy",
            candidate_hold_decisions=(2, 4, 8),
            settle_frames=4,
            minimum_progress_delta=1.0,
        )
        stage = FullSMBStage(
            env=SweepFullSMBEnv(),
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
        )
        try:
            stage.reset(seed=23)
            state = stage.save_emulator_state()
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                state_path = (
                    root
                    / "local/full_smb/states/curriculum/1_1_first_enemy_approach.state"
                )
                state_path.parent.mkdir(parents=True)
                with state_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "kind": FULL_SMB_SAVE_STATE_KIND,
                            "schema_version": FULL_SMB_SAVE_STATE_SCHEMA_VERSION,
                            "spec": {"name": spec.save_state_artifact},
                            "state": state,
                        },
                        handle,
                    )

                dataset = collect_full_smb_obstacle_window_duration_dataset(
                    stage,
                    repository_root=root,
                    decision_frame_skip=1,
                    specs=(spec,),
                    seed=23,
                )
        finally:
            stage.close()

        self.assertEqual(int(dataset["actions"].numel()), 1)
        self.assertEqual(dataset["actions"].tolist(), [2])
        self.assertEqual(dataset["primitive_duration_mask"].tolist(), [1.0])
        self.assertEqual(dataset["primitive_duration_bin"].tolist(), [3])
        self.assertEqual(dataset["metrics"]["windows_labeled"], 1.0)
        self.assertEqual(dataset["metrics"]["labels"][0]["hold_decisions"], 4)
        self.assertEqual(len(dataset["metrics"]["labels"][0]["trials"]), 3)

        opening = {
            "src_a": dataset["src_a"].clone(),
            "src_b": dataset["src_b"].clone(),
            "src_c": dataset["src_c"].clone(),
            "actions": torch.tensor([1], dtype=torch.long),
            "metrics": {"samples": 1.0, "max_progress": 0.0},
        }
        merged = merge_full_smb_imitation_datasets((opening, dataset))
        targets = full_smb_imitation_module._full_smb_imitation_primitive_targets(
            merged["actions"],
            dataset=merged,
        )

        self.assertEqual(targets["explicit_duration_mask"].tolist(), [0.0, 1.0])
        self.assertEqual(targets["duration_bin"].tolist()[1], 3)
        # The obstacle sample deliberately ships a zero release mask; implicit
        # jump-release labels must not override that explicit contract.
        self.assertEqual(targets["release_mask"].tolist(), [0.0, 0.0])
        self.assertEqual(targets["release"].tolist(), [0.0, 0.0])

    def test_primitive_targets_do_not_bridge_runs_across_merged_sources(self):
        def scripted(actions):
            count = len(actions)
            return {
                "src_a": torch.zeros((count, 4), dtype=torch.long),
                "src_b": torch.zeros((count, 4), dtype=torch.long),
                "src_c": torch.zeros((count, 4), dtype=torch.float32),
                "actions": torch.tensor(actions, dtype=torch.long),
                "metrics": {"samples": float(count), "max_progress": 0.0},
            }

        merged = merge_full_smb_imitation_datasets((scripted([2, 2]), scripted([2, 2])))
        targets = full_smb_imitation_module._full_smb_imitation_primitive_targets(
            merged["actions"],
            dataset=merged,
        )

        self.assertEqual(merged["sample_trajectory_ids"].tolist(), [0, 0, 1, 1])
        # Each source trajectory holds its own run of two jumps, not one run of four.
        self.assertEqual(targets["duration_mask"].tolist(), [1.0, 0.0, 1.0, 0.0])
        self.assertEqual(targets["duration_bin"].tolist()[0], 1)
        self.assertEqual(targets["duration_bin"].tolist()[2], 1)
        self.assertEqual(targets["release"].tolist(), [0.0, 1.0, 0.0, 1.0])
        self.assertEqual(targets["release_mask"].tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_obstacle_window_sweep_skips_spec_when_warmup_terminates(self):
        spec = FullSMBObstacleWindowDurationSpec(
            name="unit_warmup_death",
            save_state_artifact="section_1_1_first_enemy_approach",
            obstacle_kind="enemy",
            warmup_script=((2, 2), (1, 1)),
            candidate_hold_decisions=(2, 4),
            settle_frames=4,
            minimum_progress_delta=1.0,
        )
        stage = FullSMBStage(
            env=SweepFullSMBEnv(),
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
        )
        try:
            stage.reset(seed=23)
            state = stage.save_emulator_state()
            with TemporaryDirectory() as tmpdir:
                root = Path(tmpdir)
                state_path = (
                    root
                    / "local/full_smb/states/curriculum/1_1_first_enemy_approach.state"
                )
                state_path.parent.mkdir(parents=True)
                with state_path.open("wb") as handle:
                    pickle.dump(
                        {
                            "kind": FULL_SMB_SAVE_STATE_KIND,
                            "schema_version": FULL_SMB_SAVE_STATE_SCHEMA_VERSION,
                            "spec": {"name": spec.save_state_artifact},
                            "state": state,
                        },
                        handle,
                    )

                dataset = collect_full_smb_obstacle_window_duration_dataset(
                    stage,
                    repository_root=root,
                    decision_frame_skip=1,
                    specs=(spec,),
                    seed=23,
                )
        finally:
            stage.close()

        self.assertEqual(int(dataset["actions"].numel()), 0)
        self.assertEqual(dataset["metrics"]["windows_labeled"], 0.0)
        self.assertEqual(dataset["metrics"]["trial_count"], 0.0)
        self.assertEqual(
            dataset["metrics"]["skipped"],
            ({"name": "unit_warmup_death", "reason": "warmup_terminated"},),
        )

    def test_collect_and_train_imitation_warm_start_updates_policy_head(self):
        stage = FullSMBStage(
            env=TinyFullSMBEnv(),
            vision=StaticFullSMBVision(),
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
        )
        try:
            dataset = collect_full_smb_imitation_dataset(
                stage,
                full_smb_opening_imitation_script(12),
                seed=5,
            )
        finally:
            stage.close()

        model = make_full_smb_policy_model(
            architecture_config={"hidden_dim": 8, "controller_schedule": "linear"},
        )
        before = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
            if name.startswith("agent.fc_out_A") or name.startswith("agent.fc_controller_params")
        }
        metrics, _optimizer = train_full_smb_imitation_warm_start(
            model,
            dataset,
            device=torch.device("cpu"),
            epochs=1,
            batch_size=4,
            learning_rate=1e-3,
            seed=11,
        )

        changed = [
            name
            for name, value in before.items()
            if not torch.equal(value, model.state_dict()[name])
        ]
        self.assertTrue(changed)
        self.assertGreater(dataset["metrics"]["samples"], 0.0)
        self.assertGreaterEqual(metrics["mean_action_accuracy"], 0.0)
        self.assertIn("agent.fc_out_A", metrics["trainable_prefixes"])
        self.assertIn("final_primitive_loss", metrics)
        self.assertIn("duration_supervision_count", metrics)
        self.assertIn("release_supervision_count", metrics)
        self.assertIn("release_positive_count", metrics)


if __name__ == "__main__":
    unittest.main()
