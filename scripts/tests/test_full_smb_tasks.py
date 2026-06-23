"""Tests for Full SMB train/evaluation task-set definitions."""

import unittest
from pathlib import Path

from retroagi.stages.full_smb import (
    FULL_SMB_TASK_SET_NAMES,
    FullSMBTaskCatalog,
    FullSMBTaskSpec,
    FullSMBTaskStart,
    full_smb_task_catalog,
)


class TestFullSMBTaskCatalog(unittest.TestCase):
    def test_catalog_declares_smoke_benchmark_curriculum_and_heldout_sets(self):
        catalog = full_smb_task_catalog()
        manifest = catalog.to_manifest()

        self.assertIsInstance(catalog, FullSMBTaskCatalog)
        self.assertEqual(catalog.task_sets(), FULL_SMB_TASK_SET_NAMES)
        self.assertEqual(
            tuple(manifest["task_sets"].keys()),
            FULL_SMB_TASK_SET_NAMES,
        )
        self.assertTrue(catalog.tasks_for_set("smoke"))
        self.assertGreaterEqual(len(catalog.tasks_for_set("fixed_benchmark")), 3)
        self.assertGreaterEqual(len(catalog.tasks_for_set("curriculum")), 4)
        self.assertGreaterEqual(len(catalog.tasks_for_set("heldout_generalization")), 3)

    def test_task_starts_resolve_to_full_smb_env_config(self):
        catalog = full_smb_task_catalog()
        smoke = catalog.task("smoke_1_1_spawn")
        midpipe = catalog.task("curriculum_1_1_midpipe")

        self.assertEqual(smoke.start.mode, "level_start")
        self.assertEqual(smoke.start.to_env_config().state, "Level1-1")
        self.assertEqual(midpipe.start.mode, "save_state_artifact")
        self.assertEqual(midpipe.start.to_env_config().state, "Level1-1")
        self.assertEqual(
            midpipe.start.save_state_path,
            Path("local/full_smb/states/curriculum/1_1_midpipe.state"),
        )

    def test_curriculum_is_ordered_and_uses_train_split(self):
        catalog = full_smb_task_catalog()
        curriculum = catalog.curriculum

        self.assertEqual(
            [task.curriculum_stage for task in curriculum],
            sorted(task.curriculum_stage for task in curriculum),
        )
        self.assertTrue(all(task.split == "train" for task in curriculum))
        self.assertEqual(curriculum[0].name, "curriculum_1_1_opening")

    def test_heldout_tasks_are_not_train_split(self):
        catalog = full_smb_task_catalog()
        heldout = catalog.tasks_for_set("heldout_generalization")

        self.assertTrue(heldout)
        self.assertTrue(all(task.split == "heldout" for task in heldout))
        self.assertTrue(all("heldout" in task.tags for task in heldout))

    def test_save_state_artifacts_are_local_only_paths(self):
        catalog = full_smb_task_catalog()
        paths = catalog.save_state_artifact_paths

        self.assertTrue(paths)
        for path in paths:
            self.assertFalse(path.is_absolute())
            self.assertTrue(str(path).startswith("local/full_smb/states/"))

    def test_catalog_rejects_missing_task_sets(self):
        with self.assertRaisesRegex(ValueError, "missing task sets"):
            FullSMBTaskCatalog(
                tasks=(
                    FullSMBTaskSpec(
                        name="unit",
                        task_set="smoke",
                        split="eval",
                        start=FullSMBTaskStart(mode="level_start", state="Level1-1"),
                        reset_seed=0,
                        episodes=1,
                        max_steps=1,
                        curriculum_stage=1,
                        goal="unit",
                    ),
                )
            )


if __name__ == "__main__":
    unittest.main()
