"""Canonical semantic policy inputs, independent of a ViT's latent basis."""

import torch
import torch.nn.functional as F

from retroagi.core.hierarchy import VisionHierarchyProjector
from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_geometry import SEMANTICS

SCENE_SCHEMA = "smb_scene_v2"
# Availability has dedicated model-input slots, not only diagnostic metadata.
AVAILABILITY_NAMES = (
    "mario",
    "support",
    "velocity",
    "enemy_vx",
    "enemy_patrol_bounds",
    "platform_vx",
    "platform_bounds",
    "power_state",
)
VOCABULARIES = {
    "block": tuple(range(7)),
    "full": (0, 2, 2, 2, 2, 3, 5, 5, 1, 0, 0, 0, 0),
    "legacy_cnn": (0, 2, 2, 2, 5, 1),
}


def canonical_vision(vision, vocabulary):
    if (vision.metadata or {}).get("canonical_scene_schema") == SCENE_SCHEMA:
        if vision.semantic_logits.shape[1] != 7:
            raise ValueError("Invalid canonical perception vocabulary")
        return vision
    mapping = VOCABULARIES[vocabulary]
    if vision.semantic_logits.shape[1] != len(mapping):
        raise ValueError("Perception vocabulary does not match declared interface")
    p = vision.semantic_logits.float().softmax(1)
    canonical = p.new_zeros((p.shape[0], len(SEMANTICS), *p.shape[2:]))
    for source, target in enumerate(mapping):
        canonical[:, target] += p[:, source]
    # The fixed spatial descriptor has a defined class/row/column ordering.
    # Raw learned token vectors never cross this boundary.
    spatial = F.adaptive_avg_pool2d(canonical, (15, 16))
    return VisionOutput(
        position=vision.position,
        semantic_logits=canonical.clamp_min(1e-12).log(),
        semantic_ids=canonical.argmax(1),
        tokens=spatial.flatten(2).transpose(1, 2),
        support_logits=vision.support_logits,
        support_ids=vision.support_ids,
        metadata={**(vision.metadata or {}), "canonical_scene_schema": SCENE_SCHEMA},
    )


class CanonicalSMBProjector(VisionHierarchyProjector):
    def project(self, vision, state=None, metadata=None, *, availability=None):
        if (vision.metadata or {}).get("canonical_scene_schema") != SCENE_SCHEMA:
            raise ValueError("Canonical projector requires declared semantic features")
        if state is None or torch.as_tensor(state).shape[-1] != 35:
            raise ValueError("Canonical SMB requires the ordered 35 physical features")
        if vision.support_logits is None or vision.support_logits.shape[-1] != 3:
            raise ValueError("Canonical SMB requires explicit three-state support probabilities")
        batch = super().project(vision, state, metadata)
        if availability is None:
            raise ValueError("An explicit observation availability mask is required")
        availability = torch.as_tensor(
            availability, device=batch.src_c.device, dtype=batch.src_c.dtype
        )
        if availability.ndim == 1:
            availability = availability.unsqueeze(0)
        if availability.shape != (batch.src_c.shape[0], 8) or not bool(
            ((availability >= 0) & (availability <= 1)).all()
        ):
            raise ValueError("Availability must have eight probability/mask slots")
        probabilities = vision.semantic_logits.softmax(1)
        spatial = F.adaptive_avg_pool2d(probabilities, (15, 16)).flatten(1)
        descriptor = F.adaptive_avg_pool1d(spatial.unsqueeze(1), 9).squeeze(1)
        batch.src_c = torch.cat((batch.src_c[:, :47], availability, descriptor), dim=1)
        batch.metadata["smb_observation_schema"] = SCENE_SCHEMA
        batch.metadata["vision_fusion"].update(c_availability=(47, 55), c_patch_tokens=(55, 64))
        return batch


def block_oracle_scene(env, *, terminated=False, truncated=False, objective_kind=None):
    """Express Block state in the same visible-window coordinates as NES RAM."""
    from types import SimpleNamespace

    import pygame

    from retroagi.core.smb_geometry import geometry_features
    from retroagi.core.smb_objectives import objective_goal, observable_objective

    scroll = int(env.camera_x)
    viewport = pygame.Rect(0, 0, 256, 240)
    platforms = []
    support = None
    for platform in env.platforms:
        rect = platform["rect"].move(-scroll, 0).clip(viewport)
        if not rect.w or not rect.h:
            continue
        p = {**platform, "rect": rect}
        for field in ("move_x", "move_min", "move_max"):
            if field in p:
                p[field] -= scroll
        platforms.append(p)
        if env.mario.get("_platform") is platform:
            support = p
    mario = {**env.mario, "x": env.mario["x"] - scroll, "_platform": support}
    enemies = [
        {
            **e,
            "x": e["x"] - scroll,
            "patrol_min": e["patrol_min"] - scroll,
            "patrol_max": e["patrol_max"] - scroll,
        }
        for e in env.enemies
        if not e["dead"] and 0 <= e["x"] - scroll + e["w"] and e["x"] - scroll < 256
    ]
    coins = [
        {**c, "rect": c["rect"].move(-scroll, 0)}
        for c in env.coins
        if c["rect"].move(-scroll, 0).colliderect(viewport)
    ]
    goal = (
        env.goal.move(-scroll, 0)
        if getattr(env, "render_goal", True) and env.goal is not None
        else None
    )
    if goal is None or not goal.colliderect(viewport):
        goal = pygame.Rect(0 if env._terrain_left else 240, 188, 16, 20)
    scene = SimpleNamespace(
        mario=mario,
        platforms=platforms,
        enemies=enemies,
        coins=coins,
        goal=goal,
        world_width=256,
        height=240,
        max_walk_speed=3.0,
        max_fall_speed=8.0,
        steps=env.steps,
        _terrain_left=env._terrain_left,
        _goal_credited=env._goal_credited,
    )
    scene.visible_goal = (
        env.goal.move(-scroll, 0)
        if getattr(env, "render_goal", True)
        and env.goal is not None
        and env.goal.move(-scroll, 0).colliderect(viewport)
        else None
    )
    objective = observable_objective(scene, objective_kind=objective_kind)
    features = geometry_features(
        scene,
        death=bool(terminated and not env._goal_credited),
        terminated=terminated,
        truncated=truncated,
    )
    features["state_vec"][[7, 8]] = 0
    features["motion_vec"][[1, 2, 6, 7]] = 0
    return dict(
        scene=scene,
        features=features,
        objective=objective,
        skill_goal=objective_goal(objective),
        support="ground" if mario["on_ground"] else "air",
        enemy_contact=False,
        bouncing=False,
        availability=[1, 1, 1, 1, 0, 1, 0, 1],
        unavailable_features=["enemy_patrol_bounds", "platform_bounds"],
        unsupported_objects=[],
        world_x=env.mario["x"],
        scroll=scroll,
        frame=env.steps,
        player_box=[mario["x"], mario["y"], mario["w"], mario["h"]],
        observation_provider="oracle",
    )


def canonical_rgb(observation):
    """Restore centered NES overscan crop without stretching physical pixels."""
    import numpy as np

    array = np.asarray(observation)
    h, w = array.shape[:2]
    if h > 240 or w > 256 or array.ndim != 3:
        raise ValueError("Unsupported SMB capture dimensions")
    top, left = (240 - h) // 2, (256 - w) // 2
    return np.pad(array, ((top, 240 - h - top), (left, 256 - w - left), (0, 0)), mode="edge")


def preserve_objective(geometry, tracker):
    """Keep a takeoff target fixed in world coordinates until support returns."""
    from retroagi.core.smb_objectives import objective_goal
    from retroagi.stages.block_smb.local_traversal import LocalObjective

    objective = geometry["objective"]
    scroll = geometry["scroll"]
    if geometry["support"] == "air" and tracker.target is not None:
        kind, left, right, top = tracker.target[:4]
        direction = tracker.target[4] if len(tracker.target) > 4 else 1
        objective = LocalObjective(kind, left - scroll, right - scroll, top, direction=direction)
    elif geometry["support"] != "air":
        tracker.target = (
            objective.kind,
            objective.left + scroll,
            objective.right + scroll,
            objective.top,
            objective.direction,
        )
    geometry["objective"] = objective
    geometry["skill_goal"] = objective_goal(objective, bouncing=geometry.get("bouncing", False))
    return geometry


def apply_local_target(geometry):
    """Use the common visible local objective, never a simulator finish marker."""
    import math

    target = geometry["objective"]
    m = geometry["scene"].mario
    dx = ((target.left + target.right) / 2 - m["x"] - m["w"] / 2) / 256
    dy = (target.top - m["y"] - m["h"] / 2) / 240
    geometry["features"]["state_vec"][15:18] = [dx, dy, min(math.hypot(dx, dy), 1.0)]
    return geometry
