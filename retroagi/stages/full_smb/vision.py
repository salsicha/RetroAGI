"""Adapter for the existing full-SMB DeepLab segmentation checkpoint."""

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from retroagi.core import VisionOutput, VisionSpec
from retroagi.core.vision import image_tensor


FULL_SMB_SEMANTIC_CLASSES = ("background", "floor", "box", "enemy", "brick", "mario")
DEFAULT_CHECKPOINT = Path(__file__).parents[3] / "scripts" / "segmentation" / "MarioSegmentationModel.pth"


class FullSMBSegmentationVision(nn.Module):
    """Expose the trained six-class DeepLab model through the shared vision API."""

    def __init__(
        self,
        checkpoint: Optional[Path] = DEFAULT_CHECKPOINT,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        from torchvision.models.segmentation import deeplabv3_resnet50
        from torchvision.models.segmentation.deeplabv3 import DeepLabHead

        self.spec = VisionSpec("full_smb_deeplab", FULL_SMB_SEMANTIC_CLASSES, token_dim=6)
        self.model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=True)
        self.model.classifier = DeepLabHead(2048, self.spec.num_classes)
        if checkpoint is not None:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
        if device is not None:
            self.to(device)
        self.model.eval()

    def forward(self, observation: Any) -> VisionOutput:
        device = next(self.model.parameters()).device
        image = image_tensor(observation, device=device)
        logits = self.model(image)["out"]
        semantic_ids = logits.argmax(dim=1)
        probabilities = logits.softmax(dim=1)

        mario = probabilities[:, self.spec.semantic_classes.index("mario")]
        weights = mario / mario.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        height, width = mario.shape[-2:]
        y = torch.linspace(0, 1, height, device=device)
        x = torch.linspace(0, 1, width, device=device)
        position = torch.stack(
            (
                (weights * x.view(1, 1, width)).sum(dim=(1, 2)),
                (weights * y.view(1, height, 1)).sum(dim=(1, 2)),
            ),
            dim=-1,
        )

        token_logits = F.adaptive_avg_pool2d(logits, (15, 16))
        tokens = token_logits.flatten(2).transpose(1, 2)
        return VisionOutput(
            position=position,
            semantic_logits=logits,
            semantic_ids=semantic_ids,
            tokens=tokens,
            metadata={"checkpoint_classes": self.spec.semantic_classes},
        )

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)
