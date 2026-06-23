"""Tests for project documentation coverage."""

import unittest
from pathlib import Path

OPERATIONS_DOC = Path("docs/operations.md")
REPRODUCIBILITY_DOC = Path("docs/reproducibility.md")
FULL_SMB_CONTENT_DOC = Path("docs/full-smb-content.md")
FULL_SMB_TASKS_DOC = Path("docs/full-smb-tasks.md")
FULL_SMB_SAVE_STATES_DOC = Path("docs/full-smb-save-states.md")
FULL_SMB_SUCCESS_THRESHOLDS_DOC = Path("docs/full-smb-success-thresholds.md")
README = Path("README.md")


class TestOperationsDocumentation(unittest.TestCase):
    def test_operations_reference_covers_stage_runtime_metrics_and_artifacts(self):
        text = OPERATIONS_DOC.read_text(encoding="utf-8")

        for section in (
            "# Operations Reference",
            "## Runtime Baseline",
            "## Multi-Game Operations",
            "## Synthetic 1D",
            "## Block SMB Perception",
            "## Block SMB Policy",
            "## Full SMB Vision",
            "## Full SMB Content Setup",
            "## Full SMB Task Sets",
            "## Full SMB Adapter And Transfer",
        ):
            self.assertIn(section, text)

        for term in ("Hardware", "Runtime", "Expected Metrics", "Artifact Locations"):
            self.assertIn(term, text)

        for metric in (
            "controller_mse",
            "mean_iou",
            "position_rmse",
            "success_thresholds_met",
            "action_agreement",
        ):
            self.assertIn(metric, text)

        for term in (
            "artifacts/multi_game/<game>/<architecture>/manifest.json",
            "retroagi experiment",
            "--game pong",
            "artifacts/multi_game/reports/baseline_cross_game.json",
            "game_key",
            "[full-smb-content.md](full-smb-content.md)",
            "SuperMarioBros-Nes",
            "local/full_smb/checksums/SuperMarioBros-Nes.sha256",
            "retroagi check-env --game smb --stage full",
            "artifacts/full_smb/env_check.json",
            "[full-smb-tasks.md](full-smb-tasks.md)",
            "[full-smb-save-states.md](full-smb-save-states.md)",
            "[full-smb-success-thresholds.md](full-smb-success-thresholds.md)",
            "heldout_generalization",
            "python -m retroagi.stages.full_smb.save_states create",
            "FIXED_FULL_SMB_SUCCESS_THRESHOLDS",
            "FullSMBRewardConfig",
        ):
            self.assertIn(term, text)

        for artifact in (
            "data/block_vit/block_vit.pth",
            "data/block_smb/policy.pth",
            "data/vit/full_smb_vit.pth",
            "data/full_smb/transferred_policy.pth",
            "artifacts/block_smb/latest/run_summary.json",
            "artifacts/full_smb/transfer_vs_scratch.json",
        ):
            self.assertIn(artifact, text)

    def test_readme_links_operations_reference(self):
        readme = README.read_text(encoding="utf-8")

        self.assertIn("[operations reference](docs/operations.md)", readme)

    def test_reproducibility_procedure_starts_from_clean_checkout(self):
        text = REPRODUCIBILITY_DOC.read_text(encoding="utf-8")

        for section in (
            "# Reproducibility Procedure",
            "## 1. Start From A Clean Checkout",
            "## 2. Create A Supported Environment",
            "## 3. Run The Baseline Test Suite",
            "## 4. Run The Baseline Architecture Promotion Fixture",
            "## 5. Run A Traceable Architecture Sweep",
            "## 6. Run A Traceable CPU Smoke Training",
            "## 11. Set Up Full SMB Local Content",
            "## 13. Preserve The Run",
        ):
            self.assertIn(section, text)

        for command in (
            "git clone https://github.com/salsicha/RetroAGI.git",
            "git status --short",
            "python -m unittest discover -s scripts/tests -v",
            "retroagi promote",
            "retroagi experiment",
            "retroagi report",
            "retroagi experiment \\\n  --game pong",
            "retroagi train --game smb --stage block",
            "retroagi diagnose-vision --game smb --stage block",
            "retroagi evaluate --game smb --stage full",
            "retroagi transfer --game smb --stage full",
            "retroagi compare --game smb --stage full",
            "python -m retro.import local/full_smb/roms",
            "retroagi check-env --game smb --stage full",
            "python -m retroagi.stages.full_smb.save_states plan",
            "python -m retroagi.stages.full_smb.save_states create",
            "from retroagi.stages.full_smb import full_smb_task_catalog",
            "FIXED_FULL_SMB_SUCCESS_THRESHOLDS",
        ):
            self.assertIn(command, text)

        for artifact in (
            "artifacts/repro/promotion_baseline_interface.json",
            "artifacts/repro/architecture_sweeps/baseline/manifest.json",
            "artifacts/repro/architecture_sweeps/report.json",
            "artifacts/repro/multi_game/pong/baseline/manifest.json",
            "artifacts/repro/multi_game/report.json",
            "artifacts/repro/block_smb_smoke/run_summary.json",
            "artifacts/repro/block_smb_smoke/events.jsonl",
            "data/block_smb/policy.pth",
            "data/vit/full_smb_vit.pth",
            "data/full_smb/transferred_policy.pth",
            "artifacts/full_smb/transfer_vs_scratch.json",
            "local/full_smb/checksums/SuperMarioBros-Nes.sha256",
            "artifacts/full_smb/env_check.json",
            "local/full_smb/states/save_state_plan.json",
            "local/full_smb/states/save_state_manifest.json",
        ):
            self.assertIn(artifact, text)

    def test_full_smb_content_setup_documents_local_only_rom_contract(self):
        text = FULL_SMB_CONTENT_DOC.read_text(encoding="utf-8")

        for term in (
            "# Full SMB Content Setup",
            "SuperMarioBros-Nes",
            "retro.make(game=\"SuperMarioBros-Nes\")",
            "python -m pip install -e '.[full-smb]'",
            "local/full_smb/roms/",
            "python -m retro.import local/full_smb/roms",
            "local/full_smb/checksums/SuperMarioBros-Nes.sha256",
            "shasum -a 256",
            "must not be committed",
            "RuntimeError",
            "retroagi check-env --game smb --stage full",
            "verifies backend import",
            "registration, ROM availability",
        ):
            self.assertIn(term, text)

    def test_full_smb_task_sets_document_train_eval_catalog(self):
        text = FULL_SMB_TASKS_DOC.read_text(encoding="utf-8")

        for term in (
            "# Full SMB Task Sets",
            "full_smb_task_catalog",
            "`smoke`",
            "`fixed_benchmark`",
            "`curriculum`",
            "`heldout_generalization`",
            "`Level1-1`",
            "`local/full_smb/states/`",
            "[full-smb-save-states.md](full-smb-save-states.md)",
            "`smoke_1_1_spawn`",
            "`benchmark_1_1_start`",
            "`curriculum_1_1_midpipe`",
            "`heldout_8_1_long`",
            "[full-smb-success-thresholds.md](full-smb-success-thresholds.md)",
            "progress, completion, survival, score/coins",
        ):
            self.assertIn(term, text)

    def test_full_smb_save_states_document_local_artifact_workflow(self):
        text = FULL_SMB_SAVE_STATES_DOC.read_text(encoding="utf-8")

        for term in (
            "# Full SMB Save-State Artifacts",
            "full_smb_save_state_plan",
            "`starting_position`",
            "`benchmark`",
            "`level_section`",
            "`death_retry`",
            "python -m retroagi.stages.full_smb.save_states plan",
            "python -m retroagi.stages.full_smb.save_states create",
            "local/full_smb/states/save_state_plan.json",
            "local/full_smb/states/save_state_manifest.json",
            "`section_1_1_midpipe`",
            "`death_retry_1_1_first_gap`",
            "must not be committed",
            "[full-smb-success-thresholds.md](full-smb-success-thresholds.md)",
        ):
            self.assertIn(term, text)

    def test_full_smb_success_thresholds_document_fixed_benchmark_protocol(self):
        text = FULL_SMB_SUCCESS_THRESHOLDS_DOC.read_text(encoding="utf-8")

        for term in (
            "# Full SMB Success Thresholds",
            "FIXED_FULL_SMB_SUCCESS_THRESHOLDS",
            "`benchmark_1_1_start`",
            "`benchmark_1_2_start`",
            "`benchmark_2_1_start`",
            "`3200`",
            "`0.667`",
            "`0.333`",
            "progress, completion, survival, score/coins",
            "evaluate_full_smb_success_threshold",
            "threshold_met",
            "Full SMB signal extraction",
        ):
            self.assertIn(term, text)

    def test_readme_links_reproducibility_procedure(self):
        readme = README.read_text(encoding="utf-8")

        self.assertIn("[reproducibility procedure](docs/reproducibility.md)", readme)


if __name__ == "__main__":
    unittest.main()
