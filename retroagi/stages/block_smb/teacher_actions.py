"""The teacher's actions: each command it chooses, named as a verb aimed at an
object the vision reports (core/action_tokens), with the outcome its trial
reached. Training labels only: simulator state finds the objects.

- A jump that kills an enemy stomps it; one that ends on the other side of an
  enemy it leaves alive jumps over it; any other jump lands on the surface it
  ends on. A steer in the air that comes down on an enemy stomps it too.
- A run-up (a run that keeps its speed into the jump after it:
  SkillToken.carry) approaches what that jump is aimed at, named in the same
  order (run_up_slot): the jump remembered (controller_teacher._commit) or,
  in training labels, the jump the teacher makes next
  (layered_train._Lane._name_run_ups). A run-up before a jump that lands back
  on the surface it started on is a plain run, as is any run that stops.
- A run away from the goal's side with an enemy near backs away from it; any
  other run runs to the surface it ends on.
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
from retroagi.core.smb_scene_labels import (
    ENEMY_ON_SURFACE,
    MIN_SURFACE_WIDTH,
    STOMP_ROWS,
    scene_from_labels,
)

# How far (pixels) a reported enemy may be from the simulator's to be it.
MATCH = 12
# Enemies within this many pixels of Mario's middle make a run away from the
# goal's side a retreat from the nearest of them.
BACK_AWAY_RANGE = 96
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


def _enemy_boxes(env) -> list:
    return [(e["x"], e["y"], e["w"], e["h"], e["dead"]) for e in env.enemies]


def jump_aim(env, result) -> list:
    """What a jump from where Mario is now (its trial ``result``) is aimed at,
    in the order a jump is named (_classify), for naming the run-up before it
    (run_up_slot): the enemy it stomps; else each live enemy it ends on the
    other side of, nearest first, then ("point", world x, feet) where it lands."""
    if not result.outcome:
        return []
    killed = result.outcome[2]
    if killed:
        return [("enemy", min(killed))]
    m = env.mario
    point = ("point", m["x"] + m["w"] / 2 + result.end_dx, m["y"] + m["h"] + result.end_dy)
    return [("enemy", i) for i in _passed(env, result)] + [point]


def run_up_slot(aim, scene, camera_x, enemies, start):
    """(slot, simulator object) of what a run-up is named after, in the scene
    of the run's decision (``scene``, with the camera at ``camera_x``;
    ``enemies``: the simulator's enemy boxes then, _enemy_boxes): the first
    thing of its jump's ``aim`` (jump_aim) the scene reports, as the jump
    itself is named. None when it reports none of them, or when the jump lands
    back on the surface the run starts on (``start``: Mario's middle and feet
    then, world): that run is a plain run, not a run-up."""
    from types import SimpleNamespace

    view = SimpleNamespace(camera_x=camera_x)
    for item in aim:
        if item[0] == "enemy":
            x, y, w, h, dead = enemies[item[1]]
            slot = None if dead else _enemy_box_slot(scene, (x, y, w, h), camera_x)
            if slot is not None:
                return slot, ("enemy", item[1])
            continue
        _, x, feet = item
        slot = surface_slot(view, scene, x, feet)
        if slot is None or slot == surface_slot(view, scene, *start):
            return None
        return slot, ("point", x, feet)
    return None


def run_context(env, state) -> dict:
    """What a run's label needs from its decision, to be named again at the
    next one (approach_slot): the scene the skill saw, the camera's position,
    Mario's middle and feet and the enemies' boxes then."""
    m = env.mario
    return {
        "scene": _scene(env, state),
        "camera_x": env.camera_x,
        "mario": (m["x"] + m["w"] / 2, m["y"] + m["h"]),
        "enemies": _enemy_boxes(env),
    }


def approach_slot(context, env, result):
    """The slot, in the scene of an earlier decision (``context``,
    run_context), of what a jump from here (its trial ``result``) is aimed
    at, as a run-up approaches it (run_up_slot); None when that run is not
    named a run-up."""
    found = run_up_slot(
        jump_aim(env, result),
        context["scene"],
        context["camera_x"],
        context["enemies"],
        context["mario"],
    )
    return found[0] if found else None


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


def _classify(env, state, command, result):
    """(verb, the simulator object as ("enemy" | "platform", index) or None,
    the reported slot or None) for ``command`` with its trial ``result``.

    An action that reaches the goal before it ends (in the air) is named by
    where it was going: its destination."""
    if command.mode == "hold":
        return "hold", None, None
    scene = _scene(env, state)
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    end_x, end_feet = cx + result.end_dx, feet + result.end_dy

    def enemy(verb, index):
        return verb, ("enemy", index), enemy_slot(env, scene, index)

    def surface(verb, x, top):
        slot = surface_slot(env, scene, x, top)
        if slot is None and result.won:
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
    if command.mode == "jump":
        if killed:
            return enemy("stomp", min(killed))
        for i in _passed(env, result):
            if enemy_slot(env, scene, i):
                return enemy("jump_over", i)
        return surface("land_on", end_x, end_feet)
    if killed:
        # A steer in the air (or a walk off a ledge) that comes down on an
        # enemy stomps it.
        return enemy("stomp", min(killed))
    planned = state.notes.get("planned_jump") if hasattr(state, "notes") else None
    if planned is not None and planned[0] == command and command.carry:
        # A run-up (it keeps its speed into the jump remembered): named after
        # what that jump is aimed at (jump_aim, found when it was certified);
        # one before a jump back onto this surface is a plain run, as is a
        # step back that stops before a jump from rest.
        found = run_up_slot(planned[3], scene, env.camera_x, _enemy_boxes(env), (cx, feet))
        if found is not None:
            slot, (kind, *where) = found
            if kind == "enemy":
                thing = ("enemy", where[0])
            else:
                index = platform_at(env, *where)
                thing = None if index is None else ("platform", index)
            return "approach", thing, slot
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
    down): where his feet ended (for a stomp, on the enemy's top), where its
    object (``thing``) was then, in how many frames, and whether he survived
    and won."""
    from .controller_teacher import _landing_point

    dx, dy = _landing_point(env, result)
    out = {
        "dx": float(dx),
        "dy": float(dy),
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


# Runs recorded per decision for the action predictor (tried_runs): to at
# most this many surfaces.
RUN_SURFACES = 4


def tried_runs(env, state, target=None) -> list:
    """The teacher's version (carry_out) of running to each reported surface
    Mario can run to, those its screening ranks best first (at most
    RUN_SURFACES), and of backing away from the nearest enemy, as actions
    with their outcomes: [{"action", "outcome"}]. On the ground only; an
    action with no safe version is left out. (Its jumps are tried_actions.)"""
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
        if slot is not None and ActionToken("run_to", slot) not in actions:
            actions.append(ActionToken("run_to", slot))
            if len(actions) == RUN_SURFACES:
                break
    near = [
        i
        for i, e in enumerate(env.enemies)
        if not e["dead"] and abs(e["x"] + e["w"] / 2 - cx) <= BACK_AWAY_RANGE
    ]
    if near:
        slot = enemy_slot(env, scene, min(near, key=lambda i: abs(env.enemies[i]["x"] - m["x"])))
        if slot is not None:
            actions.append(ActionToken("back_away", slot))
    out = []
    for action in actions:
        command = carry_out(env, state, action)
        if command is None:
            continue
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
    as the executor's destination, or None when it has no safe version (the
    choice then counts as a failure). Training only: the simulator finds and
    tries it. Used instead of the action predictor until that predicts well
    (docs/design-decisions.md).

    The teacher's own command when the action is the teacher's choice
    (``teacher_label``, ``teacher_command``); else, of the versions the
    teacher tries that it names as this action and that are safe, the one it
    ranks best: a jump of its jump table (land_on, stomp, jump_over), a run
    it screens (run_to, back_away), or a run-up before a jump aimed at the
    action's object (approach); in the air, a steer of any verb.

    An approach to the surface Mario stands on has none: a run before a jump
    back onto it is named a plain run (run_up_slot).
    """
    from retroagi.core.tokens import SkillToken

    from .controller_teacher import _air_options, _jump_table, _rank, teacher_target

    if action.verb == "hold":
        # As long as the teacher's own wait (its length the simulator's
        # truth: a wait is one action); a single frame where the teacher
        # would move now.
        if teacher_command is not None and teacher_command.mode == "hold":
            return teacher_command
        return SkillToken("hold", 0, 0)
    if teacher_command is not None and teacher_label == action:
        return teacher_command
    m = env.mario
    if action.verb == "approach" and action.target == surface_slot(
        env, _scene(env, state), m["x"] + m["w"] / 2, m["y"] + m["h"]
    ):
        return None
    if action.verb == "approach" and _runs_up_to(env, state, action, teacher_command):
        # The teacher's run toward the object, stopping before it: a run-up
        # (the jump that follows is the next decision's).
        return teacher_command
    target = teacher_target(env)
    memo = {}
    if not m["on_ground"]:
        candidates = _air_options(env, state, target, memo)
    elif action.verb in ("land_on", "stomp", "jump_over"):
        candidates = [
            (o.command, o.result)
            for o in _jump_table(env, state, target, memo, described_only=False)
        ]
    elif action.verb == "approach":
        candidates = _approach_versions(env, state, target, memo, action)
    else:
        candidates = _run_versions(env, state, target, memo, action)
    found = []
    for command, result in candidates:
        verb, _, slot = _classify(env, state, command, result)
        named = ActionToken("hold") if verb == "hold" else ActionToken(verb, slot) if slot else None
        if named == action and (result.safe or result.won):
            found.append((command, result))
    if not found:
        return None
    return max(found, key=lambda v: _rank(*v))[0]


def _runs_up_to(env, state, action, command) -> bool:
    """Whether ``command`` (the teacher's) is a run-up along Mario's floor
    toward ``action``'s object (it keeps its speed) that ends before it."""
    m = env.mario
    if command is None or command.mode != "run" or not command.carry or not m["on_ground"]:
        return False
    kind, slot = action.target
    lists = packed_lists(_scene(env, state))
    if kind == "gaps" or slot >= len(lists[kind]):
        return False
    item = lists[kind][slot]
    if kind == "surfaces":
        x0, x1 = item.x0, item.x1
    else:
        box = item.box if kind in ("blocks", "enemies") else item
        x0, x1 = box[0], box[2]
    ahead = (x0 + x1) / 2 + env.camera_x - (m["x"] + m["w"] / 2)
    return (command.x > 0) == (ahead > 0) and abs(command.x) < abs(ahead)


def _run_versions(env, state, target, memo, action):
    """Tried runs that may be ``action`` (a run_to ending on its surface, a
    back_away away from the target's side): the best CERTIFY of those the
    teacher screens. On the ground only (carry_out)."""
    from .controller_teacher import CERTIFY, _screened_runs, _trial

    scene = _scene(env, state)
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    side = getattr(target, "direction", 1)

    def may_be(command):
        if action.verb == "back_away":
            return command.x * side < 0
        return surface_slot(env, scene, cx + command.x, feet + command.y) == action.target

    out = []
    for command in _screened_runs(env, state, target):
        if not may_be(command):
            continue
        if command not in memo:
            memo[command] = _trial(env, state, command, target)
        out.append((command, memo[command]))
        if len(out) >= CERTIFY:
            break
    return out


def _approach_versions(env, state, target, memo, action):
    """A run-up along Mario's floor before a jump aimed at ``action``'s
    object (landing on its platform, or stomping its enemy), certified as
    the teacher certifies its own (controller_teacher._best_certified, which
    remembers the jump for the next decision): [(command, result)]."""
    from .controller_teacher import REACH, _best_certified

    m = env.mario
    support = m.get("_platform")
    if not m["on_ground"] or support is None or support.get("moving"):
        return []
    scene = _scene(env, state)
    kind, slot = action.target
    if kind == "gaps":
        # The teacher names no action aimed at a gap (a jump across one lands
        # on the surface beyond it): it has no version of this.
        return []
    cx, half = m["x"] + m["w"] / 2, m["w"] / 2
    if kind == "enemies":
        index = next(
            (i for i in range(len(env.enemies)) if enemy_slot(env, scene, i) == action.target),
            None,
        )
        if index is None:
            return []
        e = env.enemies[index]
        goal_x = e["x"] + e["w"] / 2

        def wanted(option, landing, killed):
            return index in killed

    else:
        lists = packed_lists(scene)
        if slot >= len(lists[kind]):
            return []
        item = lists[kind][slot]
        if kind == "surfaces":
            x0, top, x1 = item.x0, item.top, item.x1
        else:
            box = item.box if kind == "blocks" else item
            x0, top, x1 = box[0], box[1], box[2]
        goal_x = (x0 + x1) / 2 + env.camera_x
        index = platform_at(env, goal_x, top)
        if index is None:
            return []

        def wanted(option, landing, killed):
            return landing == index

    side = 1 if goal_x >= cx else -1
    floor = support["rect"]
    travels = [
        t
        for t in range(-48, REACH + 1)
        if t and floor.left + half + 2 <= cx + t * side <= floor.right - half - 2
    ]
    best = _best_certified(
        env, state, target, memo, side, travels, wanted, lambda r: r.won or r.safe
    )
    return [best] if best is not None else []

