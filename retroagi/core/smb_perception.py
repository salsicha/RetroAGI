"""Dense canonical ViT output shared by Block and Full SMB perception modules."""

import torch
from torch import nn
from torch.nn import functional as F

from retroagi.core.interfaces import VisionOutput, VisionSpec
from retroagi.core.smb_geometry import SEMANTICS
from retroagi.core.vision import PatchVisionTransformer, image_tensor


class DenseSMBPerception(nn.Module):
    """One pixel decoder per patch; internal tokens never become policy state."""

    def __init__(
        self, *, dim=128, depth=3, heads=4, patch_size=16, class_weights=None, refinement_channels=0
    ):
        super().__init__()
        weights = (
            torch.ones(len(SEMANTICS))
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32)
        )
        if weights.shape != (len(SEMANTICS),) or not bool(
            torch.isfinite(weights).all() and (weights > 0).all()
        ):
            raise ValueError("Class weights must contain seven finite positive values")
        self.config = dict(
            dim=dim,
            depth=depth,
            heads=heads,
            patch_size=patch_size,
            class_weights=weights.tolist(),
            refinement_channels=refinement_channels,
        )
        # Weighted cross entropy learns q(class|pixel) proportional to w * p.
        # Restore p for every downstream consumer, including semantic tokens and
        # support estimates. Config persists the weights; legacy files imply ones.
        self.register_buffer(
            "log_class_weights", weights.log()[None, :, None, None], persistent=False
        )
        self.backbone = PatchVisionTransformer(
            semantic_classes=SEMANTICS,
            image_size=(240, 256),
            patch_size=patch_size,
            dim=dim,
            depth=depth,
            heads=heads,
            drop=0.0,
            position_class="mario",
            support_ground_classes=("platform",),
            support_platform_classes=("moving_platform",),
            name="canonical_smb_dense_vit",
        )
        self.pixel_head = nn.Linear(dim, patch_size * patch_size * len(SEMANTICS))
        self.refinement = (
            nn.Sequential(
                nn.Conv2d(3 + len(SEMANTICS), refinement_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(refinement_channels, refinement_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(refinement_channels, len(SEMANTICS), 1),
            )
            if refinement_channels
            else None
        )
        self.support_head = nn.Linear(dim, 3)
        self.spec = VisionSpec("canonical_smb_dense_vit", SEMANTICS, len(SEMANTICS))

    def encode(self, frames):
        return self.forward(frames)

    def forward(self, frames, *, calibrated=True):
        native = self.backbone(frames)
        batch, _, _ = native.tokens.shape
        patch = self.config["patch_size"]
        gh, gw = 240 // patch, 256 // patch
        logits = self.pixel_head(native.tokens).reshape(batch, gh, gw, patch, patch, len(SEMANTICS))
        logits = logits.permute(0, 5, 1, 3, 2, 4).reshape(batch, len(SEMANTICS), 240, 256)
        if self.refinement is not None:
            pixels = image_tensor(frames, device=logits.device)
            if pixels.shape[-2:] != logits.shape[-2:]:
                pixels = F.interpolate(
                    pixels, size=logits.shape[-2:], mode="bilinear", align_corners=False
                )
            # Retain the ViT's contextual prediction while recovering local
            # boundaries that were compressed into each patch token.
            logits = logits + self.refinement(torch.cat((pixels, logits), dim=1))
        if calibrated:
            logits = logits - self.log_class_weights
        p = logits.softmax(1)
        mario = (logits.argmax(1) == 1).float()
        mass = mario.sum((-2, -1)).clamp_min(1e-8)
        x = torch.linspace(0, 1, 256, device=p.device)
        y = torch.linspace(0, 1, 240, device=p.device)
        position = torch.stack(
            ((mario.sum(-2) * x).sum(-1) / mass, (mario.sum(-1) * y).sum(-1) / mass), -1
        )
        from retroagi.core.vision import infer_agent_support_logits

        support = infer_agent_support_logits(
            logits,
            semantic_classes=SEMANTICS,
            agent_class="mario",
            ground_classes=("platform",),
            support_classes=("air", "ground", "platform"),
            platform_classes=("moving_platform",),
            scan_depth=2,
        )
        return VisionOutput(
            position,
            logits,
            logits.argmax(1),
            F.adaptive_avg_pool2d(p, (15, 16)).flatten(2).transpose(1, 2),
            support_logits=support,
            support_ids=support.argmax(-1),
            metadata={"canonical_scene_schema": "smb_scene_v2", "dense_collision_estimate": True},
        )

    def save(self, path, *, metrics=None):
        torch.save(
            dict(
                kind="dense_smb_perception_v1",
                config=self.config,
                state=self.state_dict(),
                metrics=metrics or {},
            ),
            path,
        )

    @classmethod
    def load(cls, path, *, device="cpu", freeze=True):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["kind"] != "dense_smb_perception_v1":
            raise ValueError("Incompatible perception checkpoint")
        model = cls(**payload["config"])
        model.load_state_dict(payload["state"], strict=True)
        model.to(device).eval().requires_grad_(not freeze)
        return model
