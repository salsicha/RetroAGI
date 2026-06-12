"""Vision Transformer and semantic supervision for block SMB."""

from typing import Any

import torch
import torch.nn.functional as F

from retroagi.core import PatchVisionTransformer, VisionOutput
from retroagi.core.vision import image_tensor


BLOCK_SEMANTIC_CLASSES = (
    "background",
    "mario",
    "platform",
    "coin",
    "goal",
    "enemy",
    "moving_platform",
)

BLOCK_CLASS_COLORS = {
    "background": ((107, 140, 255),),
    "mario": ((255, 0, 0), (255, 220, 0)),
    "platform": ((139, 69, 19),),
    "coin": ((255, 215, 0),),
    "goal": ((0, 255, 0),),
    "enemy": ((160, 32, 240), (100, 0, 160)),
    "moving_platform": ((80, 160, 40),),
}


class BlockVisionTransformer(PatchVisionTransformer):
    """Mid-level ViT that predicts block-SMB semantics and Mario position."""

    def __init__(
        self,
        dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        patch_size: int = 16,
        drop: float = 0.1,
    ):
        super().__init__(
            semantic_classes=BLOCK_SEMANTIC_CLASSES,
            image_size=(240, 256),
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            drop=drop,
            position_class="mario",
            name="block_smb_vit",
        )

    @torch.no_grad()
    def semantic_targets(self, observation: Any) -> torch.Tensor:
        """Build exact pixel labels from the simplified renderer's fixed palette."""
        image = (image_tensor(observation, device=self.pos_embed.device) * 255).round().to(torch.uint8)
        labels = torch.zeros(
            image.shape[0], image.shape[2], image.shape[3], dtype=torch.long, device=image.device
        )
        pixels = image.permute(0, 2, 3, 1)
        for class_id, class_name in enumerate(self.spec.semantic_classes):
            for color in BLOCK_CLASS_COLORS[class_name]:
                rgb = torch.tensor(color, dtype=torch.uint8, device=image.device)
                labels[(pixels == rgb).all(dim=-1)] = class_id
        return labels

    @torch.no_grad()
    def patch_targets(self, observation: Any) -> torch.Tensor:
        """Reduce exact pixel labels to the ViT patch grid with actor priority."""
        labels = self.semantic_targets(observation)
        one_hot = F.one_hot(labels, num_classes=self.spec.num_classes).permute(0, 3, 1, 2).float()
        present = F.adaptive_max_pool2d(one_hot, self.grid_size)
        priority = torch.tensor(
            (0, 7, 2, 5, 6, 5, 3), dtype=present.dtype, device=present.device
        ).view(1, -1, 1, 1)
        return (present * priority).argmax(dim=1)

    @torch.no_grad()
    def position_targets(self, observation: Any) -> torch.Tensor:
        """Return the normalized center of Mario from exact renderer labels."""
        mario_pixels = self.semantic_targets(observation) == self.spec.semantic_classes.index("mario")
        mass = mario_pixels.sum(dim=(1, 2)).clamp_min(1)
        height, width = mario_pixels.shape[-2:]
        y = torch.linspace(0, 1, height, device=mario_pixels.device)
        x = torch.linspace(0, 1, width, device=mario_pixels.device)
        return torch.stack(
            (
                (mario_pixels * x.view(1, 1, width)).sum(dim=(1, 2)) / mass,
                (mario_pixels * y.view(1, height, 1)).sum(dim=(1, 2)) / mass,
            ),
            dim=-1,
        )

    def training_loss(
        self,
        observation: Any,
        semantic_weight: float = 1.0,
        position_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute joint patch-segmentation and normalized Mario-position loss."""
        output = self.forward(observation)
        targets = self.patch_targets(observation)
        semantic_loss = F.cross_entropy(output.semantic_logits, targets)

        position_target = self.position_targets(observation)
        position_loss = F.mse_loss(output.position, position_target)
        total = semantic_weight * semantic_loss + position_weight * position_loss
        return total, {"semantic": semantic_loss, "position": position_loss}

    def encode(self, observation: Any) -> VisionOutput:
        return self.forward(observation)
