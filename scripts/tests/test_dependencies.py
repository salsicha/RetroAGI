"""Tests for install dependency boundaries."""

import tomllib
import unittest
from pathlib import Path


class TestDependencyBoundaries(unittest.TestCase):
    def test_stable_retro_is_full_smb_optional_extra(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        dependencies = pyproject["project"]["dependencies"]
        full_smb = pyproject["project"]["optional-dependencies"]["full-smb"]

        self.assertFalse(any("stable-retro" in dependency for dependency in dependencies))
        self.assertTrue(any("stable-retro==1.0.0" in dependency for dependency in full_smb))
        self.assertTrue(
            any(
                "github.com/Farama-Foundation/stable-retro.git" in dependency
                for dependency in full_smb
            )
        )


if __name__ == "__main__":
    unittest.main()
