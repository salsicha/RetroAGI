"""Tests for project documentation coverage."""

import unittest
from pathlib import Path

OPERATIONS_DOC = Path("docs/operations.md")
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


if __name__ == "__main__":
    unittest.main()
