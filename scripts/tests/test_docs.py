"""Tests for project documentation coverage."""

import unittest
from pathlib import Path

OPERATIONS_DOC = Path("docs/operations.md")
REPRODUCIBILITY_DOC = Path("docs/reproducibility.md")
FULL_SMB_CONTENT_DOC = Path("docs/full-smb-content.md")
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

    def test_readme_links_reproducibility_procedure(self):
        readme = README.read_text(encoding="utf-8")

        self.assertIn("[reproducibility procedure](docs/reproducibility.md)", readme)


if __name__ == "__main__":
    unittest.main()
