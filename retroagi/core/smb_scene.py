"""Canonical semantic policy inputs, independent of a ViT's latent basis."""

import torch
import torch.nn.functional as F

from retroagi.core.hierarchy import VisionHierarchyProjector
from retroagi.core.interfaces import VisionOutput
from retroagi.core.smb_geometry import SEMANTICS

SCENE_SCHEMA = "smb_scene_v2"
SCENE_ENCODER = "canonical_semantic_motion_v4"


def enemy_relative_motion(geometry):
    """Closing velocity in pixels/frame; positive means enemy moves right
    relative to Mario. Pixel observations never require absolute camera motion.
    """
    if not geometry or "scene" not in geometry:
        return 0.0, False
    scene = geometry["scene"]
    enemy = min(
        (e for e in scene.enemies if not e.get("dead")),
        key=lambda e: abs(e["x"] + e["w"] / 2 - scene.mario["x"] - scene.mario["w"] / 2),
        default=None,
    )
    if enemy is None:
        memory = geometry.get("motion_memory", {}).get(5, {})
        return float(memory.get("relative_vx", 0)), False
    if geometry.get("observation_provider") == "perceived":
        speed = enemy.get("relative_vx")
        return (
            (float(speed), enemy.get("relative_age", 0) == 0) if speed is not None else (0.0, False)
        )
    support = scene.mario.get("_platform")
    carry = (
        support.get("move_speed", 0) * support.get("move_dir", 1)
        if support and support.get("moving")
        else 0
    )
    return enemy["speed"] * enemy["direction"] - scene.mario["vx"] - carry, True


def platform_relative_motion(geometry):
    if not geometry or "scene" not in geometry:
        return 0.0, False, 5
    scene = geometry["scene"]
    platform = min(
        (p for p in scene.platforms if p.get("moving")),
        key=lambda p: abs(p["rect"].centerx - scene.mario["x"]),
        default=None,
    )
    if platform is None:
        memory = geometry.get("motion_memory", {}).get(6, {})
        return float(memory.get("relative_vx", 0)), False, memory.get("relative_age", 5)
    if geometry.get("observation_provider") == "perceived":
        speed = platform.get("relative_vx")
        age = platform.get("relative_age", 5)
        return float(speed or 0), speed is not None and age == 0, age
    support = scene.mario.get("_platform")
    carry = (
        support.get("move_speed", 0) * support.get("move_dir", 1)
        if support and support.get("moving")
        else 0
    )
    return platform["move_speed"] * platform["move_dir"] - scene.mario["vx"] - carry, True, 0


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
        # Screen displacement is not world velocity when the camera is
        # unobservable. Do not let a learner use that changing surrogate
        # despite its unavailable flag. Facing derived from it is unknown too.
        unavailable_velocity = availability[:, 2] == 0
        batch.src_c[unavailable_velocity, 14] = 0.0
        batch.src_c[unavailable_velocity, 17] = 0.5
        geometry = (metadata or {}).get("smb_geometry")
        speed, known = enemy_relative_motion(geometry)
        enemies = [e for e in geometry["scene"].enemies if not e.get("dead")] if geometry else []
        enemy = min(
            enemies,
            key=lambda e: abs(
                e["x"]
                + e["w"] / 2
                - geometry["scene"].mario["x"]
                - geometry["scene"].mario["w"] / 2
            ),
            default={},
        )
        age = enemy.get(
            "relative_age",
            (
                0
                if known
                else (geometry or {}).get("motion_memory", {}).get(5, {}).get("relative_age", 5)
            ),
        )
        batch.src_c[:, 58] = min(1.0, age / 5.0)
        platform_speed, platform_known, platform_age = platform_relative_motion(geometry)
        batch.src_c[:, 59] = max(-1.0, min(1.0, platform_speed / 3.0))
        batch.src_c[:, 60] = float(platform_known)
        batch.src_c[:, 61] = min(1.0, platform_age / 5.0)
        # The final two descriptor slots now have explicit physical meanings.
        # A/B/C lengths and every existing physical/availability slot stay fixed.
        batch.src_c[:, 62] = max(-1.0, min(1.0, speed / 3.0))
        batch.src_c[:, 63] = float(known and age == 0)
        batch.metadata["smb_scene_encoder"] = SCENE_ENCODER
        batch.metadata["smb_observation_schema"] = SCENE_SCHEMA
        batch.metadata["vision_fusion"].update(
            c_availability=(47, 55),
            c_patch_tokens=(55, 58),
            c_enemy_motion_age=(58, 59),
            c_platform_relative_motion=(59, 62),
            c_enemy_relative_motion=(62, 64),
        )
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
    """Keep static takeoff targets fixed; required stomps follow the live enemy."""
    from retroagi.core.smb_objectives import objective_goal
    from retroagi.stages.block_smb.local_traversal import LocalObjective

    objective = geometry["objective"]
    scroll = geometry["scroll"]
    if (
        geometry["support"] == "air"
        and tracker.target is not None
        and objective.kind != "stomp"
        and tracker.target[0] != "stomp"
    ):
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
