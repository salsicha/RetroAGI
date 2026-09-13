"""Versioned SMB geometry features shared by simulator and emulator adapters.

Coordinates are pixels in a local world frame, y grows downward. Width/height
and velocity scales belong to the scene. The feature order preserves the
qualified Block SMB policy; missing physical observations must be declared in
metadata, never confused with measured zero velocity or a known patrol bound.
"""

import math

import numpy as np
import pygame

SCHEMA = "smb_geometry_v1"
SEMANTICS = ("background", "mario", "platform", "coin", "goal", "enemy", "moving_platform")
STATE_NAMES = (
    "x",
    "y",
    "vx",
    "vy",
    "grounded",
    "facing",
    "skidding",
    "coyote",
    "jump_buffer",
    "coin_dx",
    "coin_dy",
    "coin_distance",
    "enemy_dx",
    "enemy_distance",
    "elapsed",
    "goal_dx",
    "goal_dy",
    "goal_distance",
    "support_edge",
    "next_platform_dx",
    "next_platform_dy",
    "ground_24",
    "ground_48",
    "ground_72",
    "death",
    "terminated",
    "truncated",
)
MOTION_NAMES = (
    "enemy_vx",
    "enemy_patrol_min",
    "enemy_patrol_max",
    "enemy_dy",
    "bridge_dx",
    "bridge_vx",
    "bridge_min",
    "bridge_max",
)


def geometry_features(
    scene, *, death=False, terminated=False, truncated=False, coyote_frames=5, jump_buffer_frames=6
):
    m = scene.mario
    ww = scene.world_width
    wh = scene.height

    def _nearest(items, get_rect):
        best_dist = float("inf")
        best_dx = best_dy = 1.0
        mx, my = m["x"] + m["w"] / 2, m["y"] + m["h"] / 2
        for item in items:
            r = get_rect(item)
            dx = (r.centerx - mx) / ww
            dy = (r.centery - my) / wh
            d = math.hypot(dx, dy)
            if d < best_dist:
                best_dist, best_dx, best_dy = d, dx, dy
        return {"dx": best_dx, "dy": best_dy, "dist": min(best_dist, 1.0)}

    active_coins = [c for c in scene.coins if not c["collected"]]
    active_enemies = [e for e in scene.enemies if not e["dead"]]

    nc = (
        _nearest(active_coins, lambda c: c["rect"])
        if active_coins
        else {"dx": 1.0, "dy": 1.0, "dist": 1.0}
    )
    ne = (
        _nearest(active_enemies, lambda e: pygame.Rect(e["x"], e["y"], e["w"], e["h"]))
        if active_enemies
        else {"dx": 1.0, "dy": 1.0, "dist": 1.0}
    )

    # Distance to nearest platform below Mario's feet
    mario_bottom = m["y"] + m["h"]
    plat_below = 1.0
    for p in scene.platforms:
        r = p["rect"]
        if r.left <= m["x"] + m["w"] and r.right >= m["x"]:
            if r.top >= mario_bottom:
                dist = (r.top - mario_bottom) / wh
                plat_below = min(plat_below, dist)

    mx, my = m["x"] + m["w"] / 2, m["y"] + m["h"] / 2
    mario_right = m["x"] + m["w"]
    if scene.goal is None:
        goal_dx = goal_dy = goal_dist = 1.0
    else:
        goal_dx = (scene.goal.centerx - mx) / ww
        goal_dy = (scene.goal.centery - my) / wh
        goal_dist = min(math.hypot(goal_dx, goal_dy), 1.0)

    facing_left = scene._terrain_left
    support_candidates = []
    for p in scene.platforms:
        r = p["rect"]
        if r.left <= mx <= r.right and (mario_bottom - 2.0) <= r.top <= (mario_bottom + 8.0):
            support_candidates.append(r)
    if support_candidates:
        support = min(support_candidates, key=lambda r: abs(r.top - mario_bottom))
        support_forward_dx = (
            (m["x"] - support.left) if facing_left else (support.right - mario_right)
        ) / ww
    else:
        support_forward_dx = 1.0

    next_platform_dx = 1.0
    next_platform_dy = 1.0
    ahead_platforms = [
        p["rect"]
        for p in scene.platforms
        if (p["rect"].right < m["x"] if facing_left else p["rect"].left > mario_right)
        or (
            p["rect"].top < mario_bottom - 1
            and p["rect"].bottom > m["y"]
            and (
                p["rect"].left < m["x"] <= p["rect"].right
                if facing_left
                else p["rect"].left <= mario_right < p["rect"].right
            )
        )
    ]
    if ahead_platforms:
        next_platform = min(
            ahead_platforms,
            key=lambda r: m["x"] - r.right if facing_left else r.left - mario_right,
        )
        next_platform_dx = (
            max(
                0.0,
                (
                    (m["x"] - next_platform.right)
                    if facing_left
                    else (next_platform.left - mario_right)
                ),
            )
            / ww
        )
        next_platform_dy = (next_platform.top - mario_bottom) / wh

    def _ground_ahead(offset: float) -> float:
        probe_x = m["x"] - offset if facing_left else mario_right + offset
        best: float | None = None
        for p in scene.platforms:
            r = p["rect"]
            if r.left <= probe_x <= r.right:
                dy = (r.top - mario_bottom) / wh
                if best is None or abs(dy) < abs(best):
                    best = dy
        return 1.0 if best is None else best

    # Dynamic objects need direction and reversal geometry, not only
    # their distance in a single rendered frame. Kept separate so legacy
    # 27-slot checkpoints retain their exact observation layout.
    nearest_enemy = min(active_enemies, key=lambda e: abs(e["x"] - m["x"]), default=None)
    bridge = next((p for p in scene.platforms if p.get("moving")), None)
    motion_vec = np.array(
        [
            (
                nearest_enemy["speed"] * nearest_enemy["direction"] / scene.max_walk_speed
                if nearest_enemy
                else 0.0
            ),
            (nearest_enemy["patrol_min"] - nearest_enemy["x"]) / ww if nearest_enemy else 0.0,
            (nearest_enemy["patrol_max"] - nearest_enemy["x"]) / ww if nearest_enemy else 0.0,
            (nearest_enemy["y"] - m["y"]) / wh if nearest_enemy else 0.0,
            (bridge["move_x"] - m["x"]) / ww if bridge else 0.0,
            bridge["move_speed"] * bridge["move_dir"] / scene.max_walk_speed if bridge else 0.0,
            (bridge["move_min"] - bridge["move_x"]) / ww if bridge else 0.0,
            (bridge["move_max"] - bridge["move_x"]) / ww if bridge else 0.0,
        ],
        dtype=np.float32,
    )

    state_vec = np.array(
        [
            m["x"] / ww,
            m["y"] / wh,
            m["vx"] / scene.max_walk_speed,
            m["vy"] / scene.max_fall_speed,
            float(m["on_ground"]),
            float(m["facing"]),
            float(m["skidding"]),
            float(m["coyote_frames"]) / coyote_frames,
            float(m["jump_buffer"]) / jump_buffer_frames,
            nc["dx"],
            nc["dy"],
            nc["dist"],
            ne["dx"],
            ne["dist"],
            min(float(scene.steps) / 200.0, 1.0),
            goal_dx,
            goal_dy,
            goal_dist,
            support_forward_dx,
            next_platform_dx,
            next_platform_dy,
            _ground_ahead(24.0),
            _ground_ahead(48.0),
            _ground_ahead(72.0),
            float(death),
            float(terminated),
            float(truncated),
        ],
        dtype=np.float32,
    )
    return {
        "state_vec": state_vec,
        "motion_vec": motion_vec,
        "nearest_coin": nc,
        "nearest_enemy": ne,
        "platform_below_dist": plat_below,
        "goal_delta": {"dx": goal_dx, "dy": goal_dy, "dist": goal_dist},
        "support_right_dx": (support.right - mario_right) / ww if support_candidates else 1.0,
        "support_forward_dx": support_forward_dx,
        "terrain_direction": -1 if facing_left else 1,
        "next_platform_delta": {"dx": next_platform_dx, "dy": next_platform_dy},
        "ground_ahead": {str(n): _ground_ahead(float(n)) for n in (24, 48, 72)},
    }
