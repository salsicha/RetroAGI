"""Dense canonical ViT output shared by Block and Full SMB perception modules."""

import torch
from torch import nn
from torch.nn import functional as F

from retroagi.core.interfaces import VisionOutput, VisionSpec
from retroagi.core.smb_geometry import SEMANTICS
from retroagi.core.vision import PatchVisionTransformer


class DenseSMBPerception(nn.Module):
    """One pixel decoder per patch; internal tokens never become policy state."""

    def __init__(self, *, dim=128, depth=3, heads=4, patch_size=16):
        super().__init__()
        self.config = dict(dim=dim, depth=depth, heads=heads, patch_size=patch_size)
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
        self.support_head = nn.Linear(dim, 3)
        self.spec = VisionSpec("canonical_smb_dense_vit", SEMANTICS, len(SEMANTICS))

    def encode(self, frames):
        return self.forward(frames)

    def forward(self, frames):
        native = self.backbone(frames)
        batch, _, _ = native.tokens.shape
        patch = self.config["patch_size"]
        gh, gw = 240 // patch, 256 // patch
        logits = self.pixel_head(native.tokens).reshape(batch, gh, gw, patch, patch, len(SEMANTICS))
        logits = logits.permute(0, 5, 1, 3, 2, 4).reshape(batch, len(SEMANTICS), 240, 256)
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
