"""Qualify recovered CNN proposals against independent real collision labels.

These are collision-interface metrics, not a claim of human-annotated sprite
segmentation accuracy. A class outside the teacher vocabulary stays unsupported.
"""

import hashlib
import json
from pathlib import Path

import torch

from retroagi.core.smb_scene import canonical_vision
from retroagi.stages.full_smb.segmentation_teacher import (
    DEFAULT_CNN_CHECKPOINT,
    LegacyCNNSegmentationTeacher,
)
from scripts.smb_perception_training import ClipDataset, evaluate_perception


def audit_teacher(clips, output, *, device="cuda"):
    if not clips or any(c["label_source"] != "nes_collision_instrumentation" for c in clips):
        raise ValueError("Teacher audit requires independent real emulator collision labels")

    class CanonicalTeacher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.teacher = LegacyCNNSegmentationTeacher(device=device)

        def encode(self, frames):
            return canonical_vision(self.teacher.encode(frames), "legacy_cnn")

    teacher = CanonicalTeacher().eval()
    metrics = evaluate_perception(teacher, ClipDataset(clips), batch_size=2)
    metrics.update(
        checkpoint=str(DEFAULT_CNN_CHECKPOINT),
        sha256=hashlib.sha256(DEFAULT_CNN_CHECKPOINT.read_bytes()).hexdigest(),
        unsupported_classes=[3, 4, 6],
        label_target="collision_bodies",
        full_level_qualified=False,
    )
    metrics["qualified_classes"] = [
        c
        for c in (1, 2, 5)
        if metrics["iou"][str(c)] is not None
        and metrics["iou"][str(c)] >= 0.9
        and metrics["body_edge_p95"] <= 2
    ]
    Path(output).write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
