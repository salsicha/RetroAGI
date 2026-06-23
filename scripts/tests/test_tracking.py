"""Tests for optional experiment tracking integrations."""

import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from retroagi.core import (
    ExperimentTrackerConfig,
    NullExperimentTracker,
    flatten_numeric_metrics,
    make_experiment_tracker,
)


class FakeSummaryWriter:
    instances = []

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.text = []
        self.scalars = []
        self.closed = False
        FakeSummaryWriter.instances.append(self)

    def add_text(self, name, text, global_step):
        self.text.append((name, text, global_step))

    def add_scalar(self, name, value, global_step):
        self.scalars.append((name, value, global_step))

    def close(self):
        self.closed = True


class FakeWandbConfig(dict):
    def update(self, values, *, allow_val_change=False):
        self["allow_val_change"] = allow_val_change
        super().update(values)


class FakeWandbRun:
    def __init__(self):
        self.config = FakeWandbConfig()
        self.logs = []
        self.finished = False

    def log(self, values, *, step):
        self.logs.append((values, step))

    def finish(self):
        self.finished = True


class TestExperimentTracking(unittest.TestCase):
    def test_tracking_extra_declares_optional_backends(self):
        pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

        tracking = pyproject["project"]["optional-dependencies"]["tracking"]

        self.assertIn("tensorboard>=2.20,<3", tracking)
        self.assertIn("wandb>=0.22,<1", tracking)

    def test_null_tracker_is_default_noop(self):
        tracker = make_experiment_tracker(ExperimentTrackerConfig())

        self.assertIsInstance(tracker, NullExperimentTracker)
        tracker.log_config({"seed": 1})
        tracker.log_metrics({"loss": 1.0}, step=1)
        tracker.close()

    def test_flatten_numeric_metrics_skips_non_numeric_values(self):
        flat = flatten_numeric_metrics(
            {
                "loss": 1.5,
                "success": False,
                "nested": {"return": 2, "label": "skip"},
                "label": "skip",
            }
        )

        self.assertEqual(flat, {"loss": 1.5, "success": 0.0, "nested/return": 2.0})

    def test_tensorboard_tracker_logs_config_and_scalars(self):
        FakeSummaryWriter.instances.clear()
        tensorboard = SimpleNamespace(SummaryWriter=FakeSummaryWriter)
        with patch("retroagi.core.tracking.importlib.import_module", return_value=tensorboard):
            tracker = make_experiment_tracker(
                ExperimentTrackerConfig(
                    backend="tensorboard",
                    log_dir=Path("artifacts/test/tensorboard"),
                )
            )
            tracker.log_config({"seed": 7})
            tracker.log_metrics({"loss": 0.5, "nested": {"return": 2}}, step=3, prefix="train")
            tracker.close()

        writer = FakeSummaryWriter.instances[0]
        self.assertEqual(writer.log_dir, "artifacts/test/tensorboard")
        self.assertEqual(writer.text[0][0], "config")
        self.assertIn('"seed": 7', writer.text[0][1])
        self.assertIn(("train/loss", 0.5, 3), writer.scalars)
        self.assertIn(("train/nested/return", 2.0, 3), writer.scalars)
        self.assertTrue(writer.closed)

    def test_wandb_tracker_logs_config_and_scalars(self):
        run = FakeWandbRun()
        wandb = SimpleNamespace(init=lambda **_kwargs: run)
        with patch("retroagi.core.tracking.importlib.import_module", return_value=wandb):
            tracker = make_experiment_tracker(
                ExperimentTrackerConfig(
                    backend="wandb",
                    log_dir=Path("artifacts/test/wandb"),
                    project="retroagi-test",
                    run_name="unit",
                    mode="offline",
                )
            )
            tracker.log_config({"seed": 11})
            tracker.log_metrics({"loss": 0.25}, step=4, prefix="eval")
            tracker.close()

        self.assertEqual(run.config["seed"], 11)
        self.assertTrue(run.config["allow_val_change"])
        self.assertEqual(run.logs, [({"eval/loss": 0.25}, 4)])
        self.assertTrue(run.finished)

    def test_optional_tracking_backend_raises_clear_install_error(self):
        with patch(
            "retroagi.core.tracking.importlib.import_module",
            side_effect=ModuleNotFoundError("tensorboard"),
        ):
            with self.assertRaisesRegex(RuntimeError, "retroagi\\[tracking\\]"):
                make_experiment_tracker(ExperimentTrackerConfig(backend="tensorboard"))


if __name__ == "__main__":
    unittest.main()
