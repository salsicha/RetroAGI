"""Tests for the GitHub Actions CI workflow contract."""

import unittest
from pathlib import Path

WORKFLOW = Path(".github/workflows/ci.yml")


class TestCIWorkflow(unittest.TestCase):
    def test_ci_workflow_covers_required_p6_checks(self):
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("name: CI", workflow)
        self.assertIn("format:", workflow)
        self.assertIn("lint:", workflow)
        self.assertIn("unit-tests:", workflow)
        self.assertIn("cpu-smoke-training:", workflow)
        self.assertIn("git diff --name-only --diff-filter=ACMRT", workflow)
        self.assertIn("python -m black --check ${PYTHON_FILES}", workflow)
        self.assertIn("python -m ruff check ${PYTHON_FILES}", workflow)
        self.assertIn("python -m unittest discover -s scripts/tests -v", workflow)
        self.assertIn("retroagi train --stage block-smb", workflow)
        self.assertIn("--device cpu", workflow)
        self.assertIn("--disable-checkpoint-transfer", workflow)
        self.assertIn('python-version: "3.14"', workflow)
        self.assertIn('python-version: ["3.12", "3.14"]', workflow)


if __name__ == "__main__":
    unittest.main()
