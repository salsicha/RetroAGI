"""Maintained, offline-only wrapper for the recovered six-class CNN teacher."""

from pathlib import Path

import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

from retroagi.core.interfaces import VisionOutput, VisionSpec
from retroagi.core.vision import image_tensor

CNN_CLASSES = ("background", "floor", "brick", "box", "enemy", "mario")
DEFAULT_CNN_CHECKPOINT = Path("scripts/segmentation/MarioSegmentationModel.pth")


class LegacyCNNSegmentationTeacher(nn.Module):
    """Six-class label proposals; no claim of missing-class or accuracy coverage."""

    def __init__(self, checkpoint=DEFAULT_CNN_CHECKPOINT, *, device="cpu"):
        super().__init__()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        # Both weights arguments are explicit: loading a local teacher must
        # never silently download another pretrained model.
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            aux_loss=any(k.startswith("aux_classifier.") for k in state),
        )
        self.model.classifier = DeepLabHead(2048, len(CNN_CLASSES))
        self.model.load_state_dict(state, strict=True)
        self.spec = VisionSpec("legacy_full_smb_cnn_teacher", CNN_CLASSES, len(CNN_CLASSES))
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(self, observation):
        device = next(self.parameters()).device
        frame = image_tensor(observation).to(device)
        logits = self.model(frame)["out"]
        p = logits.softmax(1)
        mario = p[:, 5]
        h, w = mario.shape[-2:]
        x = torch.arange(w, device=device, dtype=p.dtype) / max(1, w - 1)
        y = torch.arange(h, device=device, dtype=p.dtype) / max(1, h - 1)
        mass = mario.sum((-2, -1)).clamp_min(1e-8)
        position = torch.stack(
            ((mario.sum(-2) * x).sum(-1) / mass, (mario.sum(-1) * y).sum(-1) / mass), -1
        )
        return VisionOutput(
            position,
            logits,
            logits.argmax(1),
            p.flatten(2).transpose(1, 2),
            metadata={
                "teacher_only": True,
                "accuracy_qualified": False,
                "unsupported_classes": ["coin", "goal", "moving_platform"],
            },
        )
