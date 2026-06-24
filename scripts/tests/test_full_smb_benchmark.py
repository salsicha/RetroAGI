"""Tests for Full SMB throughput benchmarking."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from retroagi.stages.full_smb import (
    FullSMBArtifactLayout,
    FullSMBObservationConfig,
    FullSMBStage,
    FullSMBThroughputBenchmarkConfig,
    recommended_full_smb_runtime_settings,
    run_full_smb_throughput_benchmark,
)
from scripts.tests.test_full_smb_transfer import TinyFullSMBEnv


def _tiny_stage(env):
    def factory(vision):
        return FullSMBStage(
            env=env,
            vision=vision,
            observation_config=FullSMBObservationConfig(
                frame_skip=1,
                frame_stack=2,
                resize_shape=(16, 20),
            ),
        )

    return factory


class TestFullSMBBenchmark(unittest.TestCase):
    def test_tiny_backend_throughput_benchmark_writes_report(self):
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "throughput_benchmark.json"
            env = TinyFullSMBEnv()
            result = run_full_smb_throughput_benchmark(
                FullSMBThroughputBenchmarkConfig(
                    steps=4,
                    warmup_steps=1,
                    seed=5,
                    frame_skip=1,
                    device="cpu",
                    output=output,
                ),
                stage_factory=_tiny_stage(env),
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(env.closed)
        self.assertEqual(result.selected_device, "cpu")
        self.assertEqual(result.requested_steps, 4)
        self.assertEqual(result.warmup_steps, 1)
        self.assertEqual(result.measured_steps, 4)
        self.assertEqual(result.emulator_frames, 4)
        self.assertGreater(result.steps_per_second, 0.0)
        self.assertGreater(result.emulator_frames_per_second, 0.0)
        self.assertEqual(result.average_emulator_frames_per_step, 1.0)
        self.assertFalse(result.render)
        self.assertFalse(result.encode_observations)
        self.assertIn("cpu", result.recommended_settings)
        self.assertEqual(payload["measured_steps"], 4)
        self.assertIn("emulator_frames_per_second", payload)
        self.assertIn("--device cpu", payload["recommended_settings"]["cpu"]["device_flag"])

    def test_recommended_settings_cover_cpu_cuda_and_mps(self):
        settings = recommended_full_smb_runtime_settings()

        self.assertEqual(set(settings), {"cpu", "cuda", "mps"})
        self.assertIn("stable-retro emulator stepping is CPU-bound", settings["cpu"]["notes"])
        self.assertIn("--device cuda", settings["cuda"]["device_flag"])
        self.assertIn("emulator still runs on CPU", settings["cuda"]["training"])
        self.assertIn("--device mps", settings["mps"]["device_flag"])
        self.assertIn("benchmark CPU and MPS", settings["mps"]["notes"])

    def test_artifact_layout_declares_throughput_benchmark_summary(self):
        layout = FullSMBArtifactLayout("baseline_seed0")

        self.assertEqual(
            layout.files()["throughput_benchmark"],
            Path("artifacts/full_smb/baseline_seed0/summaries/throughput_benchmark.json"),
        )

    def test_config_rejects_invalid_step_counts(self):
        with self.assertRaisesRegex(ValueError, "steps"):
            FullSMBThroughputBenchmarkConfig(steps=-1)
        with self.assertRaisesRegex(ValueError, "warmup_steps"):
            FullSMBThroughputBenchmarkConfig(warmup_steps=-1)
        with self.assertRaisesRegex(ValueError, "frame_skip"):
            FullSMBThroughputBenchmarkConfig(frame_skip=0)


if __name__ == "__main__":
    unittest.main()
