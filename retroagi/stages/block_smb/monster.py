"""Teacher for a monster segment: keep away from a monster, then jump over it.

Training only; it reads the simulator. The monster (env enemy kind "monster")
cannot be stomped and kills Mario from any side. Its segment (tactic_schedule,
kind "monster") ends once Mario stands beyond it, and may name ``keep_behind``:
a line Mario should stay behind, such as the mouth of a tunnel too low to jump
in. At each grounded decision the teacher:

- jumps over it (advance) when a jump from here is certified to land beyond it
  alive (local_traversal.safe_jump_holds);
- approaches an offscreen sleeping monster while reserving stopping and
  retreat space, before entering the wait/retreat/jump cycle;
- otherwise backs away (retreat) while he is past the line, or while it is
  coming at him closer than KEEP_DISTANCE, as long as the screen edge allows;
- otherwise stands (hold_area).
"""

from __future__ import annotations

from copy import copy
from typing import Optional

from . import tactic_schedule

# Closer than this (pixels between bodies) and coming: back away.
KEEP_DISTANCE = 20
# A jump over it is only tested this close.
JUMP_REACH = 80
# Keep clearance at the tunnel mouth and a useful retreat runway.
APPROACH_MARGIN = 4
RETREAT_ROOM = 40


def approach_action(env, seg, enemy):
    """Reveal a sleeping monster without spending the escape space.

    Training teacher only. Test one more forward frame followed by releasing
    direction until stopped. Reserve the whole braking distance, the tunnel
    lip, continuous floor behind Mario and the camera's irreversible boundary.
    The decision is repeated each frame, so visibility ends the approach.
    """
    if enemy.get("awake") or enemy["x"] < env.camera_x + env.width:
        return None
    toward = seg["direction"]
    if toward < 0:  # This simulator activates enemies off the left edge already.
        return None
    m = env.mario
    support = m.get("_platform")
    if support is None:
        return 0
    rect = support["rect"]
    motion = copy(env.motion)
    x, peak = m["x"], m["x"]
    for frame in range(96):
        dx, _, _ = motion.advance(
            direction=1 if frame == 0 else 0, jump=False, grounded=True, y=m["y"]
        )
        x += dx
        peak = max(peak, x)
        if frame > 0 and motion.x_speed <= 0:
            break
    line = min(seg.get("keep_behind", rect.right), rect.right)
    line = min(line, enemy["x"] - KEEP_DISTANCE)
    future_camera = max(env.camera_x, peak - env.width // 3)
    retreat_edge = max(rect.left, future_camera + 1)
    if peak + m["w"] + APPROACH_MARGIN > line or peak - retreat_edge < RETREAT_ROOM:
        # Brake if momentum is already too large for neutral coasting. This
        # also handles entering the segment at a run rather than from rest.
        return 3 if m["vx"] > 1 else 0
    return 1


def active_monster(env) -> Optional[int]:
    """The current monster segment's monster, while Mario has not passed it."""
    seg = tactic_schedule.current(env)
    if seg["kind"] != "monster":
        return None
    index = seg["end"].get("past_enemy")
    if index is None or env.enemies[index]["dead"]:
        return None
    return index


def monster_choice(env) -> Optional[tuple[str, int, list[int]]]:
    """(stance, action, certified jump holds) at a grounded decision, or None
    when no monster segment is current or Mario is in the air."""
    from .local_traversal import LocalObjective, safe_jump_holds

    index = active_monster(env)
    if index is None or not env.mario["on_ground"]:
        return None
    seg = tactic_schedule.current(env)
    toward = seg["direction"]
    m, enemy = env.mario, env.enemies[index]
    approach = approach_action(env, seg, enemy)
    if approach is not None:
        return {0: "hold_area", 1: "advance", 3: "retreat"}[approach], approach, []
    gap = enemy["x"] - (m["x"] + m["w"]) if toward > 0 else m["x"] - (enemy["x"] + enemy["w"])
    if gap < JUMP_REACH:
        objective = LocalObjective(
            "enemy",
            enemy["x"],
            enemy["x"] + enemy["w"] + 24,
            enemy["y"],
            enemy_index=index,
            direction=toward,
        )
        holds = safe_jump_holds(env, objective, toward)
        if holds:
            return "advance", 2 if toward > 0 else 4, holds
    line = seg.get("keep_behind")
    past_line = (
        line is not None and ((m["x"] + m["w"] if toward > 0 else m["x"]) - line) * toward > 0
    )
    coming = enemy["speed"] > 0 and enemy["direction"] == -toward
    room = m["x"] > env.camera_x + 1 if toward > 0 else True
    if (past_line or (coming and gap < KEEP_DISTANCE)) and room:
        return "retreat", 3 if toward > 0 else 1, []
    return "hold_area", 0, []
