"""The Block SMB vision model: the shared per-pixel vision transformer, Block weights.

The model is retroagi.core.vision.PixelVisionTransformer; Block SMB only names
its checkpoints and trains it on the practice game's frames, whose exact
labels come from MarioScenarioEnv.render_labels() (the frame's own shapes
drawn with their types instead of their colours). Training lives in
scripts/vit/train_block_vit.py and measurement in
scripts/vision/evaluate_block_vision.py, both through retroagi.core.pixel_vision.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch

from retroagi.core.pixel_vision import load_pixel_vision_checkpoint
from retroagi.core.vision import PixelVisionTransformer

DEFAULT_BLOCK_VIT_CHECKPOINT = Path("data/block_vit/block_vit_pixel.pth")
BLOCK_VIT_NAME = "block_smb_vit"


class BlockVisionTransformer(PixelVisionTransformer):
    """The shared per-pixel vision transformer under the Block SMB checkpoint name."""

    def __init__(self, **settings: Any):
        super().__init__(**{**settings, "name": BLOCK_VIT_NAME})


@dataclass(frozen=True)
class BlockVITLoadResult:
    model: BlockVisionTransformer
    checkpoint: dict[str, Any]
    path: Path
    frozen: bool


def set_block_vit_trainable(model: PixelVisionTransformer, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    model.train(trainable)


def load_block_vit_checkpoint(
    path: Optional[Path] = None,
    *,
    device: str | torch.device = "cpu",
    freeze: bool = True,
) -> BlockVITLoadResult:
    """Load the Block vision transformer for policy training (frozen by default).

    Pass ``freeze=False`` only for explicit fine-tuning experiments.
    """
    checkpoint_path = Path(path) if path is not None else DEFAULT_BLOCK_VIT_CHECKPOINT
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Block ViT checkpoint not found at {checkpoint_path}; train it with "
            "scripts/vit/train_block_vit.py or pass an explicit checkpoint path"
        )

    from .adapter import BLOCK_SMB_SPEC

    model, checkpoint = load_pixel_vision_checkpoint(
        checkpoint_path, stage=BLOCK_SMB_SPEC, model_class=BlockVisionTransformer, device=device
    )
    set_block_vit_trainable(model, trainable=not freeze)
    return BlockVITLoadResult(
        model=model, checkpoint=checkpoint, path=checkpoint_path, frozen=freeze
    )
