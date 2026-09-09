"""Observable traversal goals shared by Block and NES pixel/oracle adapters.

This module chooses targets, never actions. It does not read task-family names,
future platform bounds, simulator credit, or a training teacher.
"""

import torch

from retroagi.core.skills import SKILL_GOAL_ENCODING_DIM, skill_goal_encoding
from retroagi.stages.block_smb.local_traversal import LocalObjective, local_objective

OBJECTIVE_CONTRACT = "observable_traversal_v2"


def required_stomp(scene):
    """Required contact remains a target even after the player passes it."""
    m = scene.mario
    candidates = [(i, e) for i, e in enumerate(scene.enemies) if not e.get("dead", False)]
    if not candidates:
        return None
    i, e = min(
        candidates, key=lambda pair: abs(pair[1]["x"] + pair[1]["w"] / 2 - m["x"] - m["w"] / 2)
    )
    direction = 1 if e["x"] + e["w"] / 2 >= m["x"] + m["w"] / 2 else -1
    return LocalObjective(
        "stomp", e["x"], e["x"] + e["w"], e["y"], enemy_index=i, direction=direction
    )


def observable_objective(scene, *, objective_kind=None):
    objective = local_objective(scene)
    m = scene.mario
    direction = -1 if scene._terrain_left else 1
    feet = m["y"] + m["h"]
    support = m.get("_platform")
    if support is None and m["on_ground"]:
        supports = [
            p
            for p in scene.platforms
            if p["rect"].left < m["x"] + m["w"]
            and p["rect"].right > m["x"]
            and abs(p["rect"].top - feet) <= 3
        ]
        support = max(supports, key=lambda p: bool(p.get("moving")), default=None)
    bridges = [
        p
        for p in scene.platforms
        if p.get("moving")
        and abs(p["rect"].top - feet) <= 3
        and (p["rect"].right > m["x"] if direction > 0 else p["rect"].left < m["x"] + m["w"])
    ]
    bridge = (
        support
        if support is not None and support.get("moving")
        else min(bridges, key=lambda p: abs(p["rect"].centerx - m["x"]), default=None)
    )
    if bridge is not None and m["on_ground"]:
        b = bridge["rect"]
        riding = support is bridge and (
            m["x"] >= b.left + 3 if direction > 0 else m["x"] + m["w"] <= b.right - 3
        )
        if riding:
            shores = [
                p["rect"]
                for p in scene.platforms
                if not p.get("moving")
                and abs(p["rect"].top - b.top) <= 3
                and (
                    p["rect"].right > m["x"] and p["rect"].left > b.left
                    if direction > 0
                    else p["rect"].left < m["x"] + m["w"] and p["rect"].right < b.right
                )
            ]
            shore = min(shores, key=lambda r: abs(r.centerx - m["x"]), default=None)
            left, right = (
                (shore.left, min(shore.right, shore.left + 32))
                if shore is not None and direction > 0
                else (
                    (max(shore.left, shore.right - 32), shore.right)
                    if shore is not None
                    else ((240, 256) if direction > 0 else (0, 16))
                )
            )
            adjacent = (
                shore is not None
                and (shore.left - b.right if direction > 0 else b.left - shore.right) <= 2
            )
            return LocalObjective(
                "bridge_exit" if adjacent else "bridge_ride",
                left,
                right,
                b.top,
                direction=direction,
            )
        if support is bridge:
            return LocalObjective(
                "bridge_board", b.left + 4, b.right - 4, b.top, direction=direction
            )
        if support is not None:
            s = support["rect"]
            edge_distance = s.right - m["x"] - m["w"] if direction > 0 else m["x"] - s.left
            bridge_gap = b.left - s.right if direction > 0 else s.left - b.right
            # A moving platform over uninterrupted floor is not a bridge task.
            extends = b.right > s.right if direction > 0 else b.left < s.left
            if extends:
                kind = (
                    "bridge_approach"
                    if edge_distance > 32
                    else "bridge_board" if bridge_gap <= 2 else "bridge_wait"
                )
                return LocalObjective(kind, b.left + 4, b.right - 4, b.top, direction=direction)
    if objective_kind == "stomp":
        objective = required_stomp(scene) or objective
    return objective


def objective_goal(objective, *, bouncing=False):
    skill = {
        "gap": "clear_gap",
        "mount": "mount_platform",
        "enemy": "enemy_clear",
        "stomp": "enemy_clear",
        "retreat": "retreat_recover",
        "bridge_wait": "wait_pass",
        "bridge_ride": "wait_pass",
        "bridge_board": "mount_platform",
        "bridge_approach": "mount_platform",
        "bridge_exit": "clear_gap",
    }.get(objective.kind)
    # A required stomp differs from an optional enemy clear in the existing
    # goal magnitude slot. It is an explicit task request in both domains.
    return (
        skill_goal_encoding(skill, 128 if objective.kind == "stomp" else 0)
        if skill and not bouncing
        else torch.zeros(1, SKILL_GOAL_ENCODING_DIM)
    )
