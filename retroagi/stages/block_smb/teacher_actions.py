"""The teacher's actions: each command it chooses, named as a verb aimed at an
object the vision reports (core/action_tokens), with the outcome its trial
reached. Training labels only: simulator state finds the objects.

- A jump that kills an enemy stomps it; one that ends on the other side of an
  enemy it leaves alive jumps over it; any other jump lands on the surface it
  ends on.
- A run before a remembered jump (controller_teacher._commit) approaches what
  that jump is aimed at; a run away from the goal's side with an enemy near
  backs away from it; any other run runs to the surface it ends on.
- A hold holds.

The outcome of an action is where Mario's feet end and where its object is
then, both from where his feet are now, and how long it takes: the action
predictor's targets (docs/action-predictor-controller.md).
"""

import math
from collections import defaultdict
from typing import Optional

from retroagi.core.action_tokens import ActionToken
from retroagi.core.smb_observer import packed_lists
from retroagi.core.smb_pixel_types import VISIBLE_COLUMNS
from retroagi.core.smb_scene_labels import scene_from_labels

# How far (pixels) a reported enemy may be from the simulator's to be it.
MATCH = 12
# Enemies within this many pixels of Mario's middle make a run away from the
# goal's side a retreat from the nearest of them.
BACK_AWAY_RANGE = 96


def _scene(env, state):
    return getattr(state, "scene", None) or scene_from_labels(env.scene_labels())


def platform_at(env, x, feet):
    """The index of the platform whose top is at ``feet`` (within 3 pixels)
    under world x, or None."""
    for index, p in enumerate(env.platforms):
        r = p["rect"]
        if r.left - 2 <= x <= r.right + 2 and abs(r.top - feet) <= 3:
            return index
    return None


def surface_slot(env, scene, x, top):
    """The slot of the reported surface (or moving platform, pipe or block
    top) under world x whose top is at ``top``, or None."""
    lists = packed_lists(scene)
    # A point beyond the window the vision sees is matched at its edge, where
    # it reports the part of a surface it sees.
    sx = min(VISIBLE_COLUMNS[1] - 1, max(VISIBLE_COLUMNS[0], x - env.camera_x))
    for slot, s in enumerate(lists["surfaces"]):
        if s.x0 - 2 <= sx <= s.x1 + 2 and abs(s.top - top) <= 3:
            return "surfaces", slot
    for name in ("moving_platforms", "pipes"):
        for slot, box in enumerate(lists[name]):
            if box[0] - 2 <= sx <= box[2] + 2 and abs(box[1] - top) <= 3:
                return name, slot
    for slot, block in enumerate(lists["blocks"]):
        box = block.box
        if box[0] - 2 <= sx <= box[2] + 2 and abs(box[1] - top) <= 3:
            return "blocks", slot
    return None


def enemy_slot(env, scene, index):
    """The slot of the reported enemy that is the simulator's enemy ``index``
    (nearest box middle within MATCH pixels), or None."""
    e = env.enemies[index]
    ex, ey = e["x"] + e["w"] / 2 - env.camera_x, e["y"] + e["h"] / 2
    best, slot = MATCH, None
    for i, view in enumerate(packed_lists(scene)["enemies"]):
        b = view.box
        d = math.hypot((b[0] + b[2]) / 2 - ex, (b[1] + b[3]) / 2 - ey)
        if d <= best:
            best, slot = d, i
    return None if slot is None else ("enemies", slot)


def _passed(env, result):
    """The enemies a trial ended on the other side of (alive), nearest first."""
    if not result.outcome:
        return []
    m = env.mario
    cx = m["x"] + m["w"] / 2
    killed, sides = result.outcome[2], result.outcome[4]
    alive = [i for i, e in enumerate(env.enemies) if not e["dead"] and i not in killed]
    flipped = [
        i
        for i, after in zip(alive, sides)
        if after != (cx < env.enemies[i]["x"] + env.enemies[i]["w"] / 2)
    ]
    return sorted(flipped, key=lambda i: abs(env.enemies[i]["x"] - m["x"]))


def _enemy_at(env, x, feet):
    """The live enemy whose top is at a world point, or None."""
    for i, e in enumerate(env.enemies):
        if not e["dead"] and e["x"] - 2 <= x <= e["x"] + e["w"] + 2 and abs(e["y"] - feet) <= 4:
            return i
    return None


def _classify(env, state, command, result):
    """(verb, the simulator object as ("enemy" | "platform", index) or None,
    the reported slot or None) for ``command`` with its trial ``result``."""
    if command.mode == "hold":
        return "hold", None, None
    scene = _scene(env, state)
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    end_x, end_feet = cx + result.end_dx, feet + result.end_dy

    def enemy(verb, index):
        return verb, ("enemy", index), enemy_slot(env, scene, index)

    def surface(verb, x, top):
        index = platform_at(env, x, top)
        thing = None if index is None else ("platform", index)
        return verb, thing, surface_slot(env, scene, x, top)

    if command.mode == "jump":
        killed = result.outcome[2] if result.outcome else ()
        if killed:
            return enemy("stomp", min(killed))
        for i in _passed(env, result):
            if enemy_slot(env, scene, i):
                return enemy("jump_over", i)
        return surface("land_on", end_x, end_feet)
    planned = state.notes.get("planned_jump") if hasattr(state, "notes") else None
    if planned is not None and planned[0] == command:
        _, jump, stop = planned
        landing_x, landing_feet = stop + m["w"] / 2 + jump.x, feet + jump.y
        index = _enemy_at(env, landing_x, landing_feet)
        if index is not None:
            return enemy("approach", index)
        return surface("approach", landing_x, landing_feet)
    from .controller_teacher import teacher_target

    if command.x * getattr(teacher_target(env), "direction", 1) < 0:
        near = [
            i
            for i, e in enumerate(env.enemies)
            if not e["dead"] and abs(e["x"] + e["w"] / 2 - cx) <= BACK_AWAY_RANGE
        ]
        if near:
            return enemy("back_away", min(near, key=lambda i: abs(env.enemies[i]["x"] - m["x"])))
    return surface("run_to", end_x, end_feet)


def _trial_of(env, state, command):
    from .controller_teacher import _trial, teacher_target

    return _trial(env, state, command, teacher_target(env))


def teacher_action(env, state, command, result=None) -> Optional[ActionToken]:
    """The teacher's ``command`` as an action, or None when its object is not
    one the vision reports (then it cannot be taught as an action)."""
    if command is None:
        return None
    if command.mode == "hold":
        return ActionToken("hold")
    if result is None:
        result = _trial_of(env, state, command)
    verb, _, slot = _classify(env, state, command, result)
    return ActionToken(verb, slot) if slot else None


def object_point(env, thing, frames=0):
    """Where a simulator object is after ``frames`` frames (world pixels): an
    enemy's middle and top by its patrol rule, a platform's left edge and top
    (a moving one carried by its rule)."""
    kind, index = thing
    if kind == "enemy":
        from .controller_teacher import _enemy_paths

        e = env.enemies[index]
        boxes = _enemy_paths(env, frames).get(index)
        x0, y0, x1, _ = boxes[-1] if boxes else (e["x"], e["y"], e["x"] + e["w"], e["y"])
        return (x0 + x1) / 2, y0
    p = env.platforms[index]
    x, d = p["move_x"], p["move_dir"]
    if p["moving"]:
        for _ in range(frames):
            x += p["move_speed"] * d
            if x <= p["move_min"]:
                x, d = float(p["move_min"]), 1
            elif x >= p["move_max"]:
                x, d = float(p["move_max"]), -1
        x = round(x)
    return float(x), float(p["rect"].top)


def action_outcome(env, result, thing=None) -> dict:
    """What an action reached (pixels from Mario's feet now; x right, y
    down): where his feet ended, where its object (``thing``) was then, in
    how many frames, and whether he survived and won."""
    out = {
        "dx": float(result.end_dx),
        "dy": float(result.end_dy),
        "frames": int(result.frames),
        "safe": bool(result.safe),
        "won": bool(result.won),
        "room": float(min(64.0, result.margin)),
    }
    if thing is not None:
        m = env.mario
        x, top = object_point(env, thing, result.frames)
        out["object_dx"] = x - (m["x"] + m["w"] / 2)
        out["object_dy"] = top - (m["y"] + m["h"])
    return out


def teacher_example(env, state, command, result=None) -> Optional[dict]:
    """The teacher's choice as {"action", "outcome"}, or None when its object
    is not reported."""
    if command is None:
        return None
    if result is None:
        result = _trial_of(env, state, command)
    verb, thing, slot = _classify(env, state, command, result)
    if verb != "hold" and slot is None:
        return None
    action = ActionToken("hold") if verb == "hold" else ActionToken(verb, slot)
    return {"action": action, "outcome": action_outcome(env, result, thing)}


def tried_actions(env, state, target=None) -> list:
    """Every jump Mario can make from here (controller_teacher._jump_table,
    each tried once), as actions: [{"action", "outcome", "versions",
    "safe_versions"}]. Jumps naming the same action are versions of it; its
    outcome is the safe version the teacher ranks best (the most room from
    every threat), else the best of the unsafe ones."""
    from .controller_teacher import _jump_table, _rank, teacher_target

    if target is None:
        target = teacher_target(env)
    groups = defaultdict(list)
    for option in _jump_table(env, state, target, {}):
        verb, thing, slot = _classify(env, state, option.command, option.result)
        if slot is not None:
            groups[ActionToken(verb, slot)].append((option, thing))
    out = []
    for action, versions in groups.items():
        safe = [v for v in versions if v[0].result.safe]
        option, thing = max(safe or versions, key=lambda v: _rank(v[0].command, v[0].result))
        out.append(
            {
                "action": action,
                "outcome": action_outcome(env, option.result, thing),
                "versions": len(versions),
                "safe_versions": len(safe),
            }
        )
    return out


def action_rows(env, state, command):
    """The teacher's choice and every jump it could make here, as records for
    the action predictor: [ACTIONS_PER_DECISION, len(ACTION_RECORD)] numbers
    (core/action_predictor), the choice first (when its object is reported),
    padded with verb -1."""
    from retroagi.core.action_predictor import ACTIONS_PER_DECISION, action_record, empty_records

    rows = empty_records()
    count = 0
    chosen = teacher_example(env, state, command)
    if chosen is not None:
        rows[0] = action_record(chosen["action"], chosen["outcome"], True)
        count = 1
    for tried in tried_actions(env, state):
        if count == ACTIONS_PER_DECISION:
            break
        if chosen is not None and tried["action"] == chosen["action"]:
            continue
        rows[count] = action_record(tried["action"], tried["outcome"], False)
        count += 1
    return rows
