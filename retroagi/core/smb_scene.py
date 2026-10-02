"""The SMB policy observation, identical in meaning for Block SMB and Full SMB.

Every frame a geometry observer reports the visible scene in screen coordinates:
the Block simulator (block_oracle_scene) or NES RAM (full_smb.geometry.NESGeometry).
SMBProjector turns that record, the enemy observation history and the frame's
semantic segmentation into the A/B/C streams. The policy sees nothing one game
can supply and the other cannot; teachers and labels may still read simulator
ground truth.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core.hierarchy import VisionHierarchyProjector
from retroagi.core.interfaces import StageBatch, VisionOutput
from retroagi.core.smb_enemy_history import HAZARD_NAMES
from retroagi.core.smb_geometry import FEATURE_NAMES, GOAL_FEATURES, SEMANTICS
from retroagi.core.smb_pixel_types import PIXEL_TYPES

# Availability has dedicated model-input slots, not only diagnostic metadata.
AVAILABILITY_NAMES = ("mario", "support", "velocity", "enemy_vx", "platform_vx")
# For each of smb_pixel_types.PIXEL_TYPES (the types both games' vision models
# give every pixel), its place in SEMANTICS: ground, brick, question block and
# pipe are all standable platform.
PIXEL_TYPE_SEMANTICS = (0, 1, 2, 2, 2, 2, 3, 5, 6, 3)
# C-stream layout. Positions, semantics and support come from the segmentation;
# state, enemy history, availability and relative motion from the geometry
# observer; the class layout is a coarse map of where each class is on screen.
C_SPANS = {}
_offset = 0
for _name, _width in (
    ("c_position", 2),
    ("c_semantic_probabilities", len(SEMANTICS)),
    ("c_support_state", 3),
    ("c_state", len(FEATURE_NAMES)),
    ("c_enemy_history", len(HAZARD_NAMES)),
    ("c_availability", len(AVAILABILITY_NAMES)),
    ("c_enemy_motion_age", 1),
    ("c_platform_relative_motion", 3),
    ("c_enemy_relative_motion", 2),
):
    C_SPANS[_name] = (_offset, _offset + _width)
    _offset += _width
C_SEMANTIC_LAYOUT_START = _offset
del _offset, _name, _width


def observation_spec():
    """What a policy's C stream means; checkpoints record it and loads compare it."""
    return {
        "features": list(FEATURE_NAMES),
        "enemy_history": list(HAZARD_NAMES),
        "availability": list(AVAILABILITY_NAMES),
        "c_spans": {name: list(span) for name, span in C_SPANS.items()},
    }


def c_feature_index(name):
    """Absolute C-stream index of a named geometry feature."""
    return C_SPANS["c_state"][0] + FEATURE_NAMES.index(name)


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
    support = scene.mario.get("_platform")
    carry = (
        support.get("move_speed", 0) * support.get("move_dir", 1)
        if support and support.get("moving")
        else 0
    )
    return platform["move_speed"] * platform["move_dir"] - scene.mario["vx"] - carry, True, 0


def canonical_vision(vision):
    """Map the vision model's pixel types (PIXEL_TYPES) onto SEMANTICS by meaning.

    Both games' vision models output PIXEL_TYPES. The policy never reads a
    vision model's internal square descriptions: two games' models do not
    share them. The tokens become the type map pooled to a 15x16 grid.
    """
    if (vision.metadata or {}).get("canonical_semantics"):
        if vision.semantic_logits.shape[1] != len(SEMANTICS):
            raise ValueError("Invalid canonical perception vocabulary")
        return vision
    declared = (vision.metadata or {}).get("semantic_classes")
    if vision.semantic_logits.shape[1] != len(PIXEL_TYPES) or (
        declared is not None and tuple(declared) != PIXEL_TYPES
    ):
        raise ValueError("Vision output must score smb_pixel_types.PIXEL_TYPES")
    p = vision.semantic_logits.float().softmax(1)
    canonical = p.new_zeros((p.shape[0], len(SEMANTICS), *p.shape[2:]))
    for source, target in enumerate(PIXEL_TYPE_SEMANTICS):
        canonical[:, target] += p[:, source]
    spatial = F.adaptive_avg_pool2d(canonical, (15, 16))
    return VisionOutput(
        position=vision.position,
        semantic_logits=canonical.clamp_min(1e-12).log(),
        semantic_ids=canonical.argmax(1),
        tokens=spatial.flatten(2).transpose(1, 2),
        support_logits=vision.support_logits,
        support_ids=vision.support_ids,
        metadata={
            **(vision.metadata or {}),
            "source_semantic_classes": (vision.metadata or {}).get("semantic_classes"),
            "semantic_classes": SEMANTICS,
            "canonical_semantics": True,
        },
    )


def observed_features(geometry, enemy_history):
    """The geometry observer's slots of the C stream, in C_SPANS order."""
    state = np.clip(np.asarray(geometry["features"]["state"], dtype=np.float32), -1.0, 1.0)
    if state.shape != (len(FEATURE_NAMES),):
        raise ValueError("Geometry features do not match FEATURE_NAMES")
    availability = np.asarray(geometry["availability"], dtype=np.float32)
    if availability.shape != (len(AVAILABILITY_NAMES),) or not bool(
        ((availability >= 0) & (availability <= 1)).all()
    ):
        raise ValueError("Availability must have one probability/mask slot per name")
    if not availability[AVAILABILITY_NAMES.index("velocity")]:
        # Screen displacement is not world velocity when the camera is
        # unobservable. Facing derived from it is unknown too.
        state[FEATURE_NAMES.index("vx")] = 0.0
        state[FEATURE_NAMES.index("facing")] = 0.5
    history = np.asarray(enemy_history, dtype=np.float32)
    if history.shape != (len(HAZARD_NAMES),):
        raise ValueError("Enemy history does not match HAZARD_NAMES")
    speed, known = enemy_relative_motion(geometry)
    mario = geometry["scene"].mario
    enemy = min(
        (e for e in geometry["scene"].enemies if not e.get("dead")),
        key=lambda e: abs(e["x"] + e["w"] / 2 - mario["x"] - mario["w"] / 2),
        default={},
    )
    age = enemy.get(
        "relative_age",
        0 if known else geometry.get("motion_memory", {}).get(5, {}).get("relative_age", 5),
    )
    platform_speed, platform_known, platform_age = platform_relative_motion(geometry)
    motion = np.asarray(
        [
            min(1.0, age / 5.0),
            max(-1.0, min(1.0, platform_speed / 3.0)),
            float(platform_known),
            min(1.0, platform_age / 5.0),
            max(-1.0, min(1.0, speed / 3.0)),
            float(known and age == 0),
        ],
        dtype=np.float32,
    )
    return np.concatenate((state, history, availability, motion))


class SMBProjector(VisionHierarchyProjector):
    """A/B semantic streams and the C stream laid out by C_SPANS.

    ``observed`` holds one row of observed_features per frame.
    """

    def project(self, vision, observed, metadata=None):
        if not (vision.metadata or {}).get("canonical_semantics"):
            raise ValueError("The SMB projector requires canonical semantics")
        if vision.support_logits is None or vision.support_logits.shape[-1] != 3:
            raise ValueError("The SMB projector requires three-state support probabilities")
        self._validate_vision(vision)
        probabilities = vision.semantic_logits.float().softmax(1)
        batch_size = probabilities.shape[0]
        observed = torch.as_tensor(np.asarray(observed), dtype=torch.float32)
        observed = observed.reshape(batch_size, -1).to(probabilities.device)
        position = vision.position.float()
        semantics = probabilities.mean(dim=(-2, -1))
        support = self._support_tensor(vision, batch_size, probabilities.device)
        layout_slots = self.stage_spec.seq_len_c - C_SEMANTIC_LAYOUT_START
        spatial = F.adaptive_avg_pool2d(probabilities, (15, 16)).flatten(1)
        layout = F.adaptive_avg_pool1d(spatial.unsqueeze(1), layout_slots).squeeze(1)
        src_c = torch.cat((position, semantics, support, observed, layout), dim=1)
        if src_c.shape[1] != self.stage_spec.seq_len_c:
            raise ValueError("C stream does not match the stage length")
        fusion = {
            "a_semantic_regions": (1, self.stage_spec.seq_len_a),
            "b_semantic_regions": (1, self.stage_spec.seq_len_b),
            **C_SPANS,
            "c_semantic_layout": (C_SEMANTIC_LAYOUT_START, self.stage_spec.seq_len_c),
        }
        batch_metadata = dict(metadata or {})
        batch_metadata.update({"vision": vision, "vision_fusion": fusion})
        return StageBatch(
            src_a=self._semantic_stream(probabilities, self.stage_spec.seq_len_a),
            target_a=None,
            src_b=self._semantic_stream(probabilities, self.stage_spec.seq_len_b),
            target_b=None,
            src_c=src_c,
            target_c=None,
            metadata=batch_metadata,
        )


# Simulator ground truth the NES never shows: patrol and lift travel limits.
HIDDEN_FIELDS = ("patrol_min", "patrol_max", "move_min", "move_max")


def block_screen_scene(env):
    """Block state in the visible-window coordinates and scales of NES RAM.

    Everything is shifted by the camera and clipped to the 256x240 screen;
    velocities use the NES geometry scales. The goal is the screen-edge
    placeholder the NES observer uses; the finish marker is simulator truth.
    """
    from types import SimpleNamespace

    import pygame

    scroll = int(env.camera_x)
    viewport = pygame.Rect(0, 0, 256, 240)
    platforms = []
    support = None
    for platform in env.platforms:
        rect = platform["rect"].move(-scroll, 0).clip(viewport)
        if not rect.w or not rect.h:
            continue
        p = {k: v for k, v in platform.items() if k not in HIDDEN_FIELDS}
        p["rect"] = rect
        if "move_x" in p:
            p["move_x"] -= scroll
        platforms.append(p)
        if env.mario.get("_platform") is platform:
            support = p
    mario = {**env.mario, "x": env.mario["x"] - scroll, "_platform": support}
    enemies = [
        {**{k: v for k, v in e.items() if k not in HIDDEN_FIELDS}, "x": e["x"] - scroll}
        for e in env.enemies
        if not e["dead"] and e["h"] > 0 and 0 <= e["x"] - scroll + e["w"] and e["x"] - scroll < 256
    ]
    coins = [
        {**c, "rect": c["rect"].move(-scroll, 0)}
        for c in env.coins
        if c["rect"].move(-scroll, 0).colliderect(viewport)
    ]
    return SimpleNamespace(
        mario=mario,
        platforms=platforms,
        enemies=enemies,
        coins=coins,
        goal=pygame.Rect(0 if env._terrain_left else 240, 188, 16, 20),
        world_width=256,
        height=240,
        max_walk_speed=3.0,
        max_fall_speed=8.0,
        steps=env.steps,
        _terrain_left=env._terrain_left,
        _goal_credited=env._goal_credited,
    )


def block_oracle_scene(env, *, terminated=False, truncated=False, objective_kind=None):
    """The Block geometry record, in the form every observer reports."""
    from retroagi.core.smb_geometry import geometry_features
    from retroagi.core.smb_objectives import objective_goal, observable_objective

    scene = block_screen_scene(env)
    mario = scene.mario
    objective = observable_objective(scene, objective_kind=objective_kind)
    features = geometry_features(
        scene,
        death=bool(terminated and not env._goal_credited),
        terminated=terminated,
        truncated=truncated,
    )
    return dict(
        scene=scene,
        features=features,
        objective=objective,
        skill_goal=objective_goal(objective),
        support="ground" if mario["on_ground"] else "air",
        enemy_contact=False,
        bouncing=False,
        availability=[1] * len(AVAILABILITY_NAMES),
        unavailable_features=[],
        unsupported_objects=[],
        world_x=env.mario["x"],
        scroll=int(env.camera_x),
        frame=env.steps,
        player_box=[mario["x"], mario["y"], mario["w"], mario["h"]],
    )


def canonical_rgb(observation):
    """Restore centered NES overscan crop without stretching physical pixels."""
    array = np.asarray(observation)
    h, w = array.shape[:2]
    if h > 240 or w > 256 or array.ndim != 3:
        raise ValueError("Unsupported SMB capture dimensions")
    top, left = (240 - h) // 2, (256 - w) // 2
    return np.pad(array, ((top, 240 - h - top), (left, 256 - w - left), (0, 0)), mode="edge")


class ObjectiveMemory:
    """What an observer carries between frames: the takeoff target and the bounce.

    A jump's target is chosen on the ground and kept, in world coordinates,
    until Mario lands; a stomp bounce lasts until he is grounded again.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.target = None
        self.bouncing = False


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
    target = geometry["objective"]
    m = geometry["scene"].mario
    dx = ((target.left + target.right) / 2 - m["x"] - m["w"] / 2) / 256
    dy = (target.top - m["y"] - m["h"] / 2) / 240
    features = dict(geometry["features"])
    features["state"] = np.array(features["state"], dtype=np.float32, copy=True)
    features["state"][GOAL_FEATURES] = [dx, dy, min(math.hypot(dx, dy), 1.0)]
    geometry["features"] = features
    return geometry
