"""The Full SMB vision model: the shared per-pixel vision transformer, Full weights.

The model is retroagi.core.vision.PixelVisionTransformer; Full SMB only names
its checkpoints and trains it on real emulator frames, whose exact labels are
read from game memory by pixel_labels.label_frame (a frame memory cannot fully
explain is refused, never used). Frames come from vision_frames; the test
levels (vision_frames.TEST_LEVELS) are never trained on. Training lives in
scripts/vit/train_full_vit.py and measurement in
scripts/vision/evaluate_full_vision.py, both through retroagi.core.pixel_vision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from retroagi.core.pixel_vision import load_pixel_vision_checkpoint
from retroagi.core.vision import PixelVisionTransformer

DEFAULT_FULL_VIT_CHECKPOINT = Path("data/full_vit/full_vit_pixel.pth")
FULL_VIT_NAME = "full_smb_vit"


class FullVisionTransformer(PixelVisionTransformer):
    """The shared per-pixel vision transformer under the Full SMB checkpoint name."""

    def __init__(self, **settings: Any):
        super().__init__(**{**settings, "name": FULL_VIT_NAME})


@dataclass(frozen=True)
class FullVITLoadResult:
    model: FullVisionTransformer
    checkpoint: dict[str, Any]
    path: Path
    frozen: bool


def set_full_vit_trainable(model: PixelVisionTransformer, trainable: bool) -> None:
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

    from .adapter import FULL_SMB_SPEC

    model, checkpoint = load_pixel_vision_checkpoint(
        checkpoint_path, stage=FULL_SMB_SPEC, model_class=FullVisionTransformer, device=device
    )
    set_full_vit_trainable(model, trainable=not freeze)
    return FullVITLoadResult(
        model=model, checkpoint=checkpoint, path=checkpoint_path, frozen=freeze
    )
