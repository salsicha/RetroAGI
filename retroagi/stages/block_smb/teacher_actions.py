"""The teacher's actions: each command it chooses, named as a verb (run,
jump or wait: core/action_tokens) and, for a run or a jump, where Mario ends
up, an object the vision reports, with the outcome its trial reached.
Training labels only: simulator state finds the objects.

- A hold is a wait; a jump is a jump; any other command is a run: along or
  off a floor, or a steer in the air.
- A run or a jump points at the enemy it kills (Mario lands on it), else at
  the surface it ends on.

The outcome of an action is where Mario's feet end and where its object is
then, both from where his feet are now, and how long it takes: the world
model's targets (core/world_model).
"""

import math
from collections import defaultdict
from typing import Optional

from retroagi.core.action_tokens import VERB_MODES, ActionToken
from retroagi.core.smb_observer import packed_lists
from retroagi.core.smb_pixel_types import VISIBLE_COLUMNS
from retroagi.core.smb_scene_labels import (
    ENEMY_ON_SURFACE,
    MIN_SURFACE_WIDTH,
    STOMP_ROWS,
    scene_from_labels,
)

# How far (pixels) a reported enemy may be from the simulator's to be it.
MATCH = 12
# Mario stands on a surface while any of his body is over it: his middle may
# be up to half his width (10 pixels) past its edge.
STANDING_REACH = 5


def _scene(env, state):
    return getattr(state, "scene", None) or scene_from_labels(env.scene_labels())


def _nearest_under(spans, x, top):
    """The key of the span (key, left, right, span top) whose top is at
    ``top`` (within 3 pixels) under ``x``: the first within 2 pixels of it,
    else the nearest within STANDING_REACH; or None."""
    near = [(key, left, right) for key, left, right, at in spans if abs(at - top) <= 3]
    for key, left, right in near:
        if left - 2 <= x <= right + 2:
            return key
    reach = [(max(left - x, x - right), key) for key, left, right in near]
    reach = [r for r in reach if r[0] <= STANDING_REACH]
    return min(reach, key=lambda r: r[0])[1] if reach else None


def platform_at(env, x, feet):
    """The index of the platform whose top is at ``feet`` (within 3 pixels)
    under world x (or within STANDING_REACH of it), or None."""
    spans = [
        (index, p["rect"].left, p["rect"].right, p["rect"].top)
        for index, p in enumerate(env.platforms)
    ]
    return _nearest_under(spans, x, feet)


def surface_slot(env, scene, x, top):
    """The slot of the reported surface (or moving platform, pipe or block
    top) under world x whose top is at ``top`` (or within STANDING_REACH of
    it: Mario standing on its edge), or None.

    A point where the floor is cut at an enemy standing on it (the stretch
    under the enemy, and a piece beside it too narrow to report:
    smb_scene_labels.split_at_enemies) is on the nearest reported piece of
    that floor."""
    lists = packed_lists(scene)
    # A point beyond the window the vision sees is matched at its edge, where
    # it reports the part of a surface it sees.
    sx = min(VISIBLE_COLUMNS[1] - 1, max(VISIBLE_COLUMNS[0], x - env.camera_x))
    spans = [(("surfaces", slot), s.x0, s.x1, s.top) for slot, s in enumerate(lists["surfaces"])]
    for name in ("moving_platforms", "pipes"):
        spans += [((name, slot), box[0], box[2], box[1]) for slot, box in enumerate(lists[name])]
    spans += [
        (("blocks", slot), block.box[0], block.box[2], block.box[1])
        for slot, block in enumerate(lists["blocks"])
    ]
    found = _nearest_under(spans, sx, top)
    if found is not None:
        return found
    for enemy in scene.enemies:
        x0, y0, x1, y1 = enemy.box
        if enemy.kind == "defeated" or not (
            y0 <= top + ENEMY_ON_SURFACE and y1 >= top - ENEMY_ON_SURFACE
        ):
            continue
        if x0 - MIN_SURFACE_WIDTH <= sx <= x1 + MIN_SURFACE_WIDTH:
            reach = [
                (max(left - sx, sx - right), key)
                for key, left, right, at in spans
                if key[0] == "surfaces" and abs(at - top) <= 3
            ]
            reach = [r for r in reach if r[0] <= (x1 - x0) + MIN_SURFACE_WIDTH + STANDING_REACH]
            if reach:
                return min(reach)[1]
    return None


def _enemy_box_slot(scene, box, camera_x):
    """The slot of the reported enemy nearest a simulator enemy's ``box`` (x,
    y, w, h; world) by box middle, within MATCH pixels, or None."""
    x, y, w, h = box
    ex, ey = x + w / 2 - camera_x, y + h / 2
    best, slot = MATCH, None
    for i, view in enumerate(packed_lists(scene)["enemies"]):
        b = view.box
        d = math.hypot((b[0] + b[2]) / 2 - ex, (b[1] + b[3]) / 2 - ey)
        if d <= best:
            best, slot = d, i
    return None if slot is None else ("enemies", slot)


def enemy_slot(env, scene, index):
    """The slot of the reported enemy that is the simulator's enemy ``index``
    (nearest box middle within MATCH pixels), or None."""
    e = env.enemies[index]
    return _enemy_box_slot(scene, (e["x"], e["y"], e["w"], e["h"]), env.camera_x)


def _classify(env, state, command, result):
    """(verb, the simulator object as ("enemy" | "platform", index) or None,
    the reported slot or None) for ``command`` with its trial ``result``
    (not needed for a hold: a wait aims at nothing).

    An action that reaches the goal before it ends (in the air) is named by
    where it was going: its destination."""
    if command.mode == "hold":
        return "wait", None, None
    verb = "jump" if command.mode == "jump" else "run"
    scene = _scene(env, state)
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]

    def surface(x, top):
        slot = surface_slot(env, scene, x, top)
        if slot is None and result is not None and result.won:
            x, top = cx + command.x, feet + command.y
            slot = surface_slot(env, scene, x, top)
        index = platform_at(env, x, top)
        thing = None if index is None else ("platform", index)
        return verb, thing, slot

    # An enemy whose top Mario's falling feet are on was stomped by the action
    # before: vision ends it there, the simulator kills it a frame later
    # (once the boxes overlap), in this action. It does not name this one.
    underfoot = {
        i
        for i, e in enumerate(env.enemies)
        if not m["on_ground"]
        and m["vy"] >= 0
        and not e["dead"]
        and e["h"] > 0
        and 0 <= e["y"] - feet <= STOMP_ROWS
        and m["x"] < e["x"] + e["w"]
        and e["x"] < m["x"] + m["w"]
    }
    killed = set(result.outcome[2]) - underfoot if result.outcome else ()
    if killed:
        # Mario lands on the enemy (a jump, a steer in the air or a walk off
        # a ledge that comes down on it).
        index = min(killed)
        return verb, ("enemy", index), enemy_slot(env, scene, index)
    return surface(cx + result.end_dx, feet + result.end_dy)


def _trial_of(env, state, command):
    from .controller_teacher import _trial, teacher_target

    return _trial(env, state, command, teacher_target(env))


def teacher_action(env, state, command, result=None) -> Optional[ActionToken]:
    """The teacher's ``command`` as an action, or None when where it ends is
    not an object the vision reports (then it cannot be taught as an
    action)."""
    if command is None:
        return None
    if command.mode == "hold":
        return ActionToken("wait")
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
    down): where his feet ended (for a stomp, on the enemy's top), where its
    object (``thing``) was then, in how many frames, and whether he survived
    and won."""
    from .controller_teacher import _landing_point

    dx, dy = _landing_point(env, result)
    out = {
        "dx": float(dx),
        "dy": float(dy),
        "frames": int(result.frames),
        "success": bool(result.safe),
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
    """The teacher's choice as {"action", "outcome"}, or None when where it
    ends is not reported."""
    if command is None:
        return None
    if result is None:
        result = _trial_of(env, state, command)
    verb, thing, slot = _classify(env, state, command, result)
    if verb != "wait" and slot is None:
        return None
    action = ActionToken("wait") if verb == "wait" else ActionToken(verb, slot)
    return {"action": action, "outcome": action_outcome(env, result, thing)}


def tried_actions(env, state, target=None) -> list:
    """Every jump Mario can make from here (controller_teacher._jump_table,
    each tried once), as actions: [{"action", "outcome", "versions"}]. Jumps
    naming the same action are versions of it; its outcome is the version
    the teacher ranks best, the one carry_out carries out."""
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
        option, thing = max(versions, key=lambda v: _rank(v[0].command, v[0].result))
        out.append(
            {
                "action": action,
                "outcome": action_outcome(env, option.result, thing),
                "versions": len(versions),
            }
        )
    return out


# Runs recorded per decision for the world model (tried_runs): to at most
# this many surfaces.
RUN_SURFACES = 4


def tried_runs(env, state, target=None) -> list:
    """The teacher's version (carry_out) of running to each reported surface
    Mario can run to, those its screening ranks best first (at most
    RUN_SURFACES), as actions with their outcomes: [{"action", "outcome"}].
    On the ground only. (Its jumps are tried_actions.)"""
    from .controller_teacher import _screened_runs, teacher_target

    m = env.mario
    if not m["on_ground"]:
        return []
    if target is None:
        target = teacher_target(env)
    scene = _scene(env, state)
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    actions = []
    for command in _screened_runs(env, state, target):
        slot = surface_slot(env, scene, cx + command.x, feet + command.y)
        if slot is not None and ActionToken("run", slot) not in actions:
            actions.append(ActionToken("run", slot))
            if len(actions) == RUN_SURFACES:
                break
    out = []
    for action in actions:
        command = carry_out(env, state, action)
        result = _trial_of(env, state, command)
        verb, thing, slot = _classify(env, state, command, result)
        if slot is not None and ActionToken(verb, slot) == action:
            out.append({"action": action, "outcome": action_outcome(env, result, thing)})
    return out


def action_rows(env, state, command):
    """The teacher's choice, its runs (tried_runs) and every jump it could
    make here (tried_actions), as records for the world model:
    [ACTIONS_PER_DECISION, len(ACTION_RECORD)] numbers
    (core/world_model), the choice first (when its object is reported),
    the runs before the jumps (far fewer of them), padded with verb -1."""
    from retroagi.core.world_model import ACTIONS_PER_DECISION, action_record, empty_records

    rows = empty_records()
    count = 0
    chosen = teacher_example(env, state, command)
    if chosen is not None:
        rows[0] = action_record(chosen["action"], chosen["outcome"], True)
        count = 1
    for tried in tried_runs(env, state) + tried_actions(env, state):
        if count == ACTIONS_PER_DECISION:
            break
        if chosen is not None and tried["action"] == chosen["action"]:
            continue
        rows[count] = action_record(tried["action"], tried["outcome"], False)
        count += 1
    return rows


def carry_out(env, state, action, teacher_label=None, teacher_command=None):
    """The teacher's version of the skill's own chosen ``action`` from here,
    as the executor's destination. Training only: the simulator finds and
    tries it. Used instead of the world model while the layers train
    (docs/design-decisions.md).

    The teacher's own command when the action is the teacher's choice
    (``teacher_label``, ``teacher_command``: a wait as long as the teacher
    would hold, a run-up that keeps its speed into the jump after it); a
    wait where the teacher would move holds for a single frame. Else, of the
    versions the teacher tries that it names as this action, the one it
    ranks best: a jump of its jump table, a run it screens; in the air, a
    steer. Where none is named as it, a move straight at where it ends
    (_straight_at). The skill learns to imitate the teacher: its action is
    carried out whatever follows (user, 2026-10-17)."""
    from retroagi.core.tokens import SkillToken

    from .controller_teacher import _rank

    if action.verb == "wait":
        # As long as the teacher's own wait (its length the simulator's
        # truth: a wait is one action).
        if teacher_command is not None and teacher_command.mode == "hold":
            return teacher_command
        return SkillToken("hold", 0, 0)
    if teacher_command is not None and teacher_label == action:
        return teacher_command
    named = _named_versions(env, state, action)
    if named:
        return max(named, key=lambda v: _rank(*v))[0]
    return _straight_at(env, state, action)


def _named_versions(env, state, action):
    """The versions of ``action`` the teacher tries from here and names as
    it: [(command, result)] (carry_out)."""
    from .controller_teacher import _air_options, _jump_table, teacher_target

    m = env.mario
    target = teacher_target(env)
    memo = {}
    if not m["on_ground"]:
        candidates = _air_options(env, state, target, memo)
    elif action.verb == "jump":
        candidates = [
            (o.command, o.result)
            for o in _jump_table(env, state, target, memo, described_only=False)
        ]
    else:
        candidates = _run_versions(env, state, target, memo, action)
    out = []
    for command, result in candidates:
        verb, _, slot = _classify(env, state, command, result)
        if slot is not None and ActionToken(verb, slot) == action:
            out.append((command, result))
    return out


def _straight_at(env, state, action):
    """A move straight at where ``action`` ends, where no version the teacher
    tries is named as it (carry_out): a run or a jump (its verb) to the
    nearest point of its object Mario fits on; a hold where it names nothing
    reported."""
    from retroagi.core.tokens import SkillToken

    m = env.mario
    cx, feet, half = m["x"] + m["w"] / 2, m["y"] + m["h"], m["w"] / 2
    kind, slot = action.target
    lists = packed_lists(_scene(env, state))
    if slot >= len(lists[kind]):
        return SkillToken("hold", 0, 0)
    item = lists[kind][slot]
    if kind == "surfaces":
        x0, top, x1 = item.x0, item.top, item.x1
    else:
        box = item.box if kind in ("blocks", "enemies") else item
        x0, top, x1 = box[0], box[1], box[2]
    x0, x1 = x0 + env.camera_x, x1 + env.camera_x  # the scene's x is the screen's
    aim = min(max(cx, x0 + half), x1 - half)
    return SkillToken(VERB_MODES[action.verb], round(aim - cx), round(top - feet))


def _run_versions(env, state, target, memo, action):
    """Tried runs that may be ``action`` (ending on its surface): the best
    CERTIFY of those the teacher screens. On the ground only (carry_out)."""
    from .controller_teacher import CERTIFY, _screened_runs, _trial

    scene = _scene(env, state)
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    out = []
    for command in _screened_runs(env, state, target):
        if surface_slot(env, scene, cx + command.x, feet + command.y) != action.target:
            continue
        if command not in memo:
            memo[command] = _trial(env, state, command, target)
        out.append((command, memo[command]))
        if len(out) >= CERTIFY:
            break
    return out
