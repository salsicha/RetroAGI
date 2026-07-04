"""Tests for Full SMB architecture benchmark comparisons."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi.core import BASELINE_ARCHITECTURE_NAME
from retroagi.stages.full_smb import benchmark_full_smb_policy_architectures
from scripts.tests.test_full_smb_transfer import write_block_policy_checkpoint


class TestFullSMBArchitectureBenchmark(unittest.TestCase):
    def test_benchmark_reports_throughput_action_and_recurrent_metrics(self):
        result = benchmark_full_smb_policy_architectures(
            architecture_config={"hidden_dim": 8},
            device="cpu",
            batch_size=1,
            iterations=1,
            warmup=0,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertIn(BASELINE_ARCHITECTURE_NAME, result["architectures"])
        self.assertEqual(tuple(result["architectures"]), (BASELINE_ARCHITECTURE_NAME,))
        for architecture in result["architectures"].values():
            self.assertGreater(architecture["train_step"]["steps_per_second"], 0.0)
            self.assertGreater(architecture["inference"]["steps_per_second"], 0.0)
            self.assertIn("mean_entropy", architecture["action_quality"])
            self.assertTrue(architecture["recurrent_state"]["state_available"])
        self.assertEqual(result["comparison"], {})

    def test_benchmark_reports_block_to_full_shared_state_migration(self):
        with TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "block_policy.pth"
            write_block_policy_checkpoint(checkpoint_path)

            result = benchmark_full_smb_policy_architectures(
                architecture_config={"hidden_dim": 8},
                block_policy_checkpoint=checkpoint_path,
                device="cpu",
                batch_size=1,
                iterations=1,
                warmup=0,
            )

        baseline_transfer = result["architectures"][BASELINE_ARCHITECTURE_NAME]["transfer"]
        self.assertTrue(baseline_transfer["loaded"])
        self.assertEqual(baseline_transfer["unexpected_keys"], [])
        self.assertGreater(baseline_transfer["shared_key_count"], 0)


if __name__ == "__main__":
    unittest.main()
