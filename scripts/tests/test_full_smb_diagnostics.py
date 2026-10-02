"""Tests for the Full SMB vision measurement behind `retroagi diagnose-vision --stage full`."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from retroagi.core.scene_vision import save_scene_vision_checkpoint
from retroagi.core.smb_pixel_types import TYPE_ID
from retroagi.core.smb_scene_labels import SceneLabels
from retroagi.stages.full_smb import (
    FULL_SMB_SPEC,
    FullVisionTransformer,
    diagnostics,
    run_full_smb_vision_diagnostic,
)
from retroagi.stages.full_smb.pixel_labels import LabelledFrame
from retroagi.stages.full_smb.vision_frames import LEVELS


def _frame(level: str) -> LabelledFrame:
    labels = np.zeros((240, 256), dtype=np.uint8)
    labels[200:, :] = TYPE_ID["ground"]
    labels[184:200, 40:52] = TYPE_ID["mario"]
    instances = np.full((240, 256), -1, dtype=np.int32)
    instances[184:200, 40:52] = 0
    return LabelledFrame(
        image=np.zeros((240, 256, 3), dtype=np.uint8),
        labels=labels,
        on_ground=True,
        enemy_rects=(),
        family=level,
        scene=SceneLabels(
            types=labels,
            instances=instances,
            categories={0: "mario"},
            kinds={},
            standing=True,
            facing_right=True,
        ),
    )


class FakeLabelledFrames:
    """Stands in for vision_frames.labelled_frames: two frames per play, one refusal."""

    def __init__(self):
        self.calls = []

    def __call__(self, level, *, frames, seed, every, refusals=None):
        self.calls.append({"level": level, "frames": frames, "seed": seed, "every": every})
        if refusals is not None:
            refusals["a drawing memory does not explain"] += 1
        yield _frame(level)
        yield _frame(level)


class TestFullSMBVisionDiagnostic(unittest.TestCase):
    def test_measures_only_the_test_levels_and_counts_refused_frames(self):
        fake = FakeLabelledFrames()
        model = FullVisionTransformer(dim=16, depth=1, heads=4)
        with patch.object(diagnostics, "labelled_frames", fake):
            result = run_full_smb_vision_diagnostic(
                model, seed=3, plays=2, every=5, frames=40, batch_size=3
            )

        self.assertEqual(
            [call["level"] for call in fake.calls],
            [level for level in LEVELS for _ in range(2)],
        )
        self.assertTrue(all(call["every"] == 5 and call["frames"] == 40 for call in fake.calls))
        self.assertEqual(result["levels"], list(LEVELS))
        self.assertEqual(result["plays"], 2 * len(LEVELS))
        self.assertEqual(result["frames"], 4 * len(LEVELS))
        self.assertEqual(
            result["refused_frames"], {"a drawing memory does not explain": 2 * len(LEVELS)}
        )
        for key in ("pixels_correct", "mario", "enemies", "surfaces", "gaps"):
            self.assertIn(key, result)
        self.assertEqual(result["mario"]["frames"], result["frames"])
        self.assertTrue(model.training, "the measurement must restore the model's mode")

    def test_play_seeds_depend_only_on_the_seed(self):
        model = FullVisionTransformer(dim=16, depth=1, heads=4)
        seeds = []
        for _ in range(2):
            fake = FakeLabelledFrames()
            with patch.object(diagnostics, "labelled_frames", fake):
                run_full_smb_vision_diagnostic(model, seed=11, plays=1, frames=8)
            seeds.append([call["seed"] for call in fake.calls])
        self.assertEqual(seeds[0], seeds[1])

    def test_rejects_zero_plays(self):
        with self.assertRaisesRegex(ValueError, "plays"):
            run_full_smb_vision_diagnostic(FullVisionTransformer(dim=16, depth=1, heads=4), plays=0)

    def test_command_loads_the_checkpoint_frozen_and_writes_the_report(self):
        source = FullVisionTransformer(dim=16, depth=1, heads=4)
        with TemporaryDirectory() as tmpdir:
            checkpoint = Path(tmpdir) / "full_vit_scene.pth"
            output = Path(tmpdir) / "reports" / "vision.json"
            save_scene_vision_checkpoint(checkpoint, source, stage=FULL_SMB_SPEC.name, metrics={})
            with (
                patch.object(diagnostics, "labelled_frames", FakeLabelledFrames()),
                patch("builtins.print"),
            ):
                exit_code = diagnostics.main(
                    [
                        "--vision-checkpoint",
                        str(checkpoint),
                        "--device",
                        "cpu",
                        "--plays",
                        "1",
                        "--output",
                        str(output),
                    ]
                )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["vision"], {"checkpoint_path": str(checkpoint), "frozen": True})
        self.assertEqual(payload["config"]["plays"], 1)
        self.assertEqual(payload["levels"], list(LEVELS))
        self.assertEqual(payload["frames"], 2 * len(LEVELS))
        self.assertIn("refused_frames", payload)
        self.assertTrue(torch.isfinite(torch.tensor(payload["pixels_correct"])))


if __name__ == "__main__":
    unittest.main()
