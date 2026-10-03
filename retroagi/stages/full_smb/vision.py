"""The Full SMB vision model: the shared scene vision transformer, Full weights.

The model is retroagi.core.vision.SceneVisionTransformer; Full SMB only names
its checkpoints and trains it on real emulator frames, whose exact labels are
read from game memory by pixel_labels.label_frame (a frame memory cannot fully
explain is refused, never used). Frames come from vision_frames: plays of
every level start, measured on other plays. Training lives in
scripts/vit/train_full_vit.py and measurement in
scripts/vision/evaluate_full_vision.py, both through retroagi.core.scene_vision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from retroagi.core.scene_vision import load_scene_vision_checkpoint
from retroagi.core.vision import SceneVisionTransformer

DEFAULT_FULL_VIT_CHECKPOINT = Path("data/full_vit/full_vit_scene.pth")
FULL_VIT_NAME = "full_smb_vit"


class FullVisionTransformer(SceneVisionTransformer):
    """The shared scene vision transformer under the Full SMB checkpoint name."""

    def __init__(self, **settings: Any):
        super().__init__(**{**settings, "name": FULL_VIT_NAME})


@dataclass(frozen=True)
class FullVITLoadResult:
    model: FullVisionTransformer
    checkpoint: dict[str, Any]
    path: Path
    frozen: bool


def set_full_vit_trainable(model: SceneVisionTransformer, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.train(trainable)


def load_full_vit_checkpoint(
    path: Optional[Path] = None,
    *,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> FullVITLoadResult:
    """Load the Full SMB vision transformer for policy training (frozen by default).

    Pass ``freeze=False`` only for explicit fine-tuning experiments.
    """
    checkpoint_path = Path(path) if path is not None else DEFAULT_FULL_VIT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Full SMB ViT checkpoint not found at {checkpoint_path}; train it with "
            "scripts/vit/train_full_vit.py or pass an explicit checkpoint path"
        )

    model, checkpoint = load_scene_vision_checkpoint(
        checkpoint_path, stage="full_smb", model_class=FullVisionTransformer, device=device
    )
    set_full_vit_trainable(model, trainable=not freeze)
    return FullVITLoadResult(
        model=model, checkpoint=checkpoint, path=checkpoint_path, frozen=freeze
    )
