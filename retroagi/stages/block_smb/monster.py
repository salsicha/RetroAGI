"""Teacher for a monster segment: keep away from a monster, then jump over it.

Training only; it reads the simulator. The monster (env enemy kind "monster")
cannot be stomped and kills Mario from any side. Its segment (tactic_schedule,
kind "monster") ends once Mario stands beyond it, and may name ``keep_behind``:
a line Mario should stay behind, such as the mouth of a tunnel too low to jump
in. At each grounded decision the teacher:

- jumps over it (advance) when a jump from here is certified to land beyond it
  alive (local_traversal.safe_jump_holds);
- otherwise backs away (retreat) while he is past the line, or while it is
  coming at him closer than KEEP_DISTANCE, as long as the screen edge allows;
- otherwise stands (hold_area).
"""

from __future__ import annotations

from typing import Optional

from . import tactic_schedule

# Closer than this (pixels between bodies) and coming: back away.
KEEP_DISTANCE = 20
# A jump over it is only tested this close.
JUMP_REACH = 80


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
