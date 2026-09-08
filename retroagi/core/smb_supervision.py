"""Collision-label supervision and an explicitly oracle-only diagnostic encoder."""

import numpy as np
import pygame
import torch
from torch.nn import functional as F

from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_scene import SCENE_SCHEMA


def collision_labels(scene):
    """Visible physical bodies, not sprite outlines or unobservable patrol bounds."""
    labels = np.zeros((240, 256), dtype=np.uint8)
    viewport = pygame.Rect(0, 0, 256, 240)

    def paint(rect, cls):
        rect = pygame.Rect(rect).clip(viewport)
        labels[rect.top : rect.bottom, rect.left : rect.right] = cls

    for p in scene.platforms:
        paint(p["rect"], 6 if p.get("moving") else 2)
    for c in scene.coins:
        if not c["collected"]:
            paint(c["rect"], 3)
    # The fallback viewport target is an objective, never a visible flag label.
    if getattr(scene, "visible_goal", None) is not None:
        paint(scene.visible_goal, 4)
    for e in scene.enemies:
        if not e["dead"]:
            paint((e["x"], e["y"], e["w"], e["h"]), 5)
    m = scene.mario
    paint((m["x"], m["y"], m["w"], m["h"]), 1)
    return labels


def labels_to_vision(labels, *, device="cpu"):
    labels = torch.as_tensor(labels, device=device).long()
    if labels.ndim == 2:
        labels = labels.unsqueeze(0)
    logits = F.one_hot(labels, 7).permute(0, 3, 1, 2).float() * 30 - 15
    masks = (labels == 1).float()
    mass = masks.sum((-2, -1)).clamp_min(1)
    x = torch.arange(256, device=device) / 255
    y = torch.arange(240, device=device) / 239
    position = torch.stack(
        ((masks.sum(-2) * x).sum(-1) / mass, (masks.sum(-1) * y).sum(-1) / mass), -1
    )
    from retroagi.core.smb_geometry import SEMANTICS
    from retroagi.core.vision import infer_agent_support_logits

    support = infer_agent_support_logits(
        logits,
        semantic_classes=SEMANTICS,
        agent_class="mario",
        ground_classes=("platform",),
        platform_classes=("moving_platform",),
        scan_depth=2,
    )
    return VisionOutput(
        position,
        logits,
        labels,
        F.adaptive_avg_pool2d(logits.softmax(1), (15, 16)).flatten(2).transpose(1, 2),
        support_logits=support,
        metadata={"canonical_scene_schema": SCENE_SCHEMA, "provider": "oracle_collision_labels"},
    )


class OracleSceneVision:
    """Diagnostic lane only. Target pixel playback must use DenseSMBPerception."""

    def __init__(self, scene, device="cpu"):
        self.scene = scene
        self.device = device

    def encode(self, _observation):
        return labels_to_vision(collision_labels(self.scene()), device=self.device)
