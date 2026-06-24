"""Tests for Full SMB local save-state artifact recipes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from retroagi.core import SMBAction
from retroagi.stages.full_smb import (
    FULL_SMB_SAVE_STATE_CATEGORIES,
    FullSMBSaveStateArtifactSpec,
    FullSMBSaveStateStep,
    create_full_smb_save_state_artifact,
    full_smb_save_state_plan,
    full_smb_task_catalog,
    load_full_smb_save_state_payload,
    write_full_smb_save_state_plan,
)


class FakeSaveStateStage:
    def __init__(self):
        self.seed = 0
        self.step_count = 0
        self.actions: list[int] = []
        self.closed = False
        self.last_info = {}

    def reset(self, seed=None):
        self.seed = 0 if seed is None else int(seed)
        self.step_count = 0
        self.actions.clear()
        self.last_info = {"x_pos": self.seed % 100, "score": 0}
        return self._observation()

    def step(self, action):
        shared_action = SMBAction(action)
        self.actions.append(int(shared_action))
        self.step_count += 1
        self.last_info = {
            "x_pos": self.seed % 100 + self.step_count,
            "score": self.step_count * 10,
            "action": shared_action.name.lower(),
        }
        return self._observation(), 1.5, False, False, dict(self.last_info)

    def save_emulator_state(self):
        return {
            "seed": self.seed,
            "step_count": self.step_count,
            "actions": tuple(self.actions),
            "last_info": dict(self.last_info),
        }

    def close(self):
        self.closed = True

    def _observation(self):
        value = (self.seed + self.step_count) % 256
        return np.full((8, 8, 3), value, dtype=np.uint8)


class TestFullSMBSaveStates(unittest.TestCase):
    def test_plan_covers_task_catalog_paths_and_categories(self):
        plan = full_smb_save_state_plan()
        catalog = full_smb_task_catalog()

        self.assertEqual(plan.categories(), FULL_SMB_SAVE_STATE_CATEGORIES)
        plan_paths = {path for path in plan.paths}
        self.assertTrue(set(catalog.save_state_artifact_paths).issubset(plan_paths))
        self.assertTrue(plan.artifacts_for_task("curriculum_1_1_midpipe"))
        self.assertTrue(plan.artifacts_for_task("benchmark_1_1_start"))
        for artifact in plan.artifacts:
            manifest = artifact.to_manifest()
            self.assertTrue(manifest["local_only"])
            self.assertFalse(artifact.path.is_absolute())
            self.assertTrue(str(artifact.path).startswith("local/full_smb/states/"))

    def test_plan_manifest_is_json_serializable_and_commits_no_content(self):
        manifest = full_smb_save_state_plan().to_manifest()
        encoded = json.dumps(manifest, sort_keys=True)

        self.assertIn("retroagi.full_smb.save_state_plan", encoded)
        self.assertFalse(manifest["copyrighted_content_committed"])
        self.assertIn("death_retry", manifest["categories"])

    def test_artifact_paths_must_stay_under_local_full_smb_states(self):
        with self.assertRaisesRegex(ValueError, "local/full_smb/states"):
            FullSMBSaveStateArtifactSpec(
                name="bad",
                category="benchmark",
                path=Path("artifacts/full_smb/bad.state"),
                source_state="Level1-1",
                reset_seed=1,
                description="bad path",
            )

    def test_create_artifact_writes_local_payload(self):
        spec = FullSMBSaveStateArtifactSpec(
            name="unit_state",
            category="benchmark",
            path=Path("local/full_smb/states/benchmark/unit.state"),
            source_state="Level1-1",
            reset_seed=7,
            description="unit state",
            task_names=("benchmark_1_1_start",),
            action_script=(
                FullSMBSaveStateStep(SMBAction.RIGHT, 2),
                FullSMBSaveStateStep("jump", 1),
            ),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            result = create_full_smb_save_state_artifact(
                spec,
                repository_root=Path(tmpdir),
                stage_factory=lambda _spec: FakeSaveStateStage(),
            )
            payload_path = Path(tmpdir) / result.path
            payload = load_full_smb_save_state_payload(payload_path)

        self.assertEqual(result.name, "unit_state")
        self.assertEqual(result.steps_executed, 3)
        self.assertEqual(result.final_info["score"], 30)
        self.assertGreater(result.bytes_written, 0)
        self.assertEqual(payload["spec"]["name"], "unit_state")
        self.assertEqual(payload["state"]["actions"], (1, 1, 5))

    def test_create_artifact_refuses_existing_file_without_overwrite(self):
        spec = FullSMBSaveStateArtifactSpec(
            name="unit_state",
            category="benchmark",
            path=Path("local/full_smb/states/benchmark/unit.state"),
            source_state="Level1-1",
            reset_seed=7,
            description="unit state",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output = root / spec.path
            output.parent.mkdir(parents=True)
            output.write_bytes(b"existing")

            with self.assertRaises(FileExistsError):
                create_full_smb_save_state_artifact(
                    spec,
                    repository_root=root,
                    stage_factory=lambda _spec: FakeSaveStateStage(),
                )

    def test_write_plan_writes_json_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "plan.json"
            manifest = write_full_smb_save_state_plan(output)
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(loaded["kind"], "retroagi.full_smb.save_state_plan")
        self.assertEqual(loaded["artifacts"], manifest["artifacts"])


if __name__ == "__main__":
    unittest.main()
