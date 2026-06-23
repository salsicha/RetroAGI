"""Tests for project documentation coverage."""

import unittest
from pathlib import Path

OPERATIONS_DOC = Path("docs/operations.md")
REPRODUCIBILITY_DOC = Path("docs/reproducibility.md")
README = Path("README.md")


class TestOperationsDocumentation(unittest.TestCase):
    def test_operations_reference_covers_stage_runtime_metrics_and_artifacts(self):
        text = OPERATIONS_DOC.read_text(encoding="utf-8")

        for section in (
            "# Operations Reference",
            "## Runtime Baseline",
            "## Synthetic 1D",
            "## Block SMB Perception",
            "## Block SMB Policy",
            "## Full SMB Vision",
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
            "## 5. Run A Traceable CPU Smoke Training",
            "## 11. Preserve The Run",
        ):
            self.assertIn(section, text)

        for command in (
            "git clone https://github.com/salsicha/RetroAGI.git",
            "git status --short",
            "python -m unittest discover -s scripts/tests -v",
            "retroagi promote",
            "retroagi train --stage block-smb",
            "retroagi diagnose-vision --stage block-smb",
            "retroagi evaluate --stage full-smb",
            "retroagi transfer --stage full-smb",
            "retroagi compare --stage full-smb",
        ):
            self.assertIn(command, text)

        for artifact in (
            "artifacts/repro/promotion_baseline_interface.json",
            "artifacts/repro/block_smb_smoke/run_summary.json",
            "artifacts/repro/block_smb_smoke/events.jsonl",
            "data/block_smb/policy.pth",
            "data/vit/full_smb_vit.pth",
            "data/full_smb/transferred_policy.pth",
            "artifacts/full_smb/transfer_vs_scratch.json",
        ):
            self.assertIn(artifact, text)

    def test_readme_links_reproducibility_procedure(self):
        readme = README.read_text(encoding="utf-8")

        self.assertIn("[reproducibility procedure](docs/reproducibility.md)", readme)


if __name__ == "__main__":
    unittest.main()
