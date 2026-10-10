"""Training-only certification and repair of spatial teacher destinations.

Every candidate goes through the unmodified ground/flight controllers. Hidden
geometry is used only to propose labels and score simulator trials.

Among the commands that work, the teacher labels the one with the most margin
(Result.margin): the farthest Mario stays from every enemy he does not stomp,
and, where he comes to stand on a platform, the farthest from its edges. A
command that only just works is not taught when a safer one does the same.
"""

import math
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Optional

from retroagi.core.smb_agent import LandingWatch
from retroagi.core.smb_coaching import probe_state, training_target
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken

from . import tactic_schedule
from .env_state import restore_env_state, snapshot_env_state


@dataclass
class Result:
    safe: bool = False
    won: bool = False
    reached: bool = False
    advanced: bool = False
    progress: float = 0
    frames: int = 0
    clearance: float = 999
    snapshot: object = None
    spatial: object = None
    controller: object = None
    # The closest Mario's body came to any live enemy he did not kill, over the
    # whole trial (pixels between boxes), and his distance from the nearer
    # edge of the platform he ended standing on (None when not standing).
    enemy_gap: float = 999.0
    edge_margin: Optional[float] = None
    # What the command achieved: the goal, the platform he ended on, the
    # enemies it killed, the route mark reached and which side of each other
    # enemy he ended on. Commands with the same outcome are versions of the
    # same move.
    outcome: tuple = field(default=())

    @property
    def margin(self) -> float:
        """The tighter of the enemy gap and the edge margin."""
        return min(self.enemy_gap, 999.0 if self.edge_margin is None else self.edge_margin)


def _box_gaps(env, gaps):
    """Lower each live enemy's closest distance to Mario's box (pixels)."""
    m = env.mario
    for i, e in enumerate(env.enemies):
        if e["dead"] or e["h"] <= 0:
            continue
        dx = max(e["x"] - (m["x"] + m["w"]), m["x"] - (e["x"] + e["w"]), 0)
        dy = max(e["y"] - (m["y"] + m["h"]), m["y"] - (e["y"] + e["h"]), 0)
        gaps[i] = min(gaps.get(i, 999.0), math.hypot(dx, dy))


def _measured(env, gaps, alive, mark):
    """(enemy_gap, edge_margin, outcome) at the end of a trial."""
    killed = frozenset(i for i in alive if env.enemies[i]["dead"])
    enemy_gap = min((g for i, g in gaps.items() if i not in killed), default=999.0)
    m = env.mario
    support = m.get("_platform") if m["on_ground"] else None
    edge = None
    index = None
    if support is not None:
        r = support["rect"]
        edge = float(min(m["x"] - r.left, r.right - (m["x"] + m["w"])))
        index = next((i for i, p in enumerate(env.platforms) if p is support), None)
    cx = m["x"] + m["w"] / 2
    sides = tuple(
        cx < env.enemies[i]["x"] + env.enemies[i]["w"] / 2 for i in alive if i not in killed
    )
    mark = env._tactic_index, env._route_done
    outcome = (bool(env._goal_credited), index, killed, mark, sides)
    return enemy_gap, edge, outcome


def distance(env, target):
    m = env.mario
    x, y = m["x"] + m["w"] / 2, m["y"] + m["h"]
    left, right, top = target.left, target.right, target.top
    if target.platform_index is not None:
        r = env.platforms[target.platform_index]["rect"]
        left, right, top = r.left, r.right, r.top
    if target.enemy_index is not None:
        e = env.enemies[target.enemy_index]
        if e["dead"]:
            return 0
        left, right, top = e["x"], e["x"] + e["w"], e["y"]
        if target.kind != "stomp":
            left = right + 8 if target.direction > 0 else left - 8
            right = left
            top = y
    return max(left + 2 - x, x - right + 2, 0) + abs(y - top)


def trial(env, state, goal, target, *, save=False):
    result = _trial(env, state, goal, target, save=save)
    if (
        (result.safe or result.won)
        and goal.mode == "jump"
        and getattr(state, "visual_observer", None) is not None
    ):
        return _trial(env, state, goal, target, save=save, visual=True)
    return result


def _trial(env, state, goal, target, *, save=False, visual=False):
    def render():
        return type(env).render(env)

    with probe_state(env):
        spatial = deepcopy(state.execution) if state.execution is not None else SpatialFeedback()
        scene = state.scene or scene_from_labels(env.scene_labels())
        if state.execution is None:
            spatial.motion = deepcopy(env.motion)
            spatial.observe(scene)
        executor = SMBExecutor(last_button=2 if spatial.motion.previous_jump else 0)
        controller = getattr(state, "controller", None)
        if controller is not None:
            executor.hold = deepcopy(controller.hold)
            executor.last_button = controller.last_button
        observer = getattr(state, "visual_observer", None)
        unsettled_contact = (
            observer is not None and scene.mario.on_something != env.mario["on_ground"]
        )
        landing = LandingWatch(airborne=not scene.mario.on_something)
        plan = spatial.begin(goal, scene)
        executor.start(plan, flight=spatial.flight, travel=spatial.travel)
        before = distance(env, target)
        mark = env._tactic_index, env._route_done
        start = env.steps
        points = env.points()
        x, feet = env.mario["x"], env.mario["y"] + env.mario["h"]
        alive = [i for i, e in enumerate(env.enemies) if not e["dead"]]
        gaps: dict = {}
        _box_gaps(env, gaps)
        for _ in range(192):
            button = executor.press(scene)
            spatial.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            _box_gaps(env, gaps)
            if done or truncated:
                enemy_gap, edge, outcome = _measured(env, gaps, alive, mark)
                return Result(
                    safe=env._goal_credited,
                    won=env._goal_credited,
                    frames=env.steps - start,
                    enemy_gap=enemy_gap,
                    edge_margin=edge,
                    outcome=outcome,
                )
            if visual or unsettled_contact:
                scene = observer.observe([render()])[0]
                unsettled_contact = scene.mario.on_something != env.mario["on_ground"]
            else:
                scene = scene_from_labels(env.scene_labels())
            spatial.observe(scene)
            ended = (
                landing.landed(scene)
                or executor.finished
                or executor.reconsider
                or spatial.arrived()
            )
            stalled = spatial.stalled and (
                ended or spatial.observed_frames >= spatial.positions.maxlen
            )
            if ended or stalled:
                m = env.mario
                advanced = mark != (env._tactic_index, env._route_done)
                reached = spatial.report(scene)[2] > 0.5
                landed = m["on_ground"] or env.stomped
                moved = abs(m["x"] - x) + abs(m["y"] + m["h"] - feet)
                clearance = min(
                    (
                        max(e["x"] - m["x"] - m["w"], m["x"] - e["x"] - e["w"])
                        for e in env.enemies
                        if not e["dead"]
                        and e["h"] > 0
                        and abs(e["y"] + e["h"] - (m["y"] + m["h"])) < 20
                    ),
                    default=999,
                )
                objective = (
                    env.enemies[target.enemy_index]["dead"]
                    if target.kind == "stomp" and target.enemy_index is not None
                    else target.reached(env)
                )
                enemy_gap, edge, outcome = _measured(env, gaps, alive, mark)
                return Result(
                    safe=bool(
                        route_available(env)
                        and not boxed_with_enemy(env, clearance)
                        and plant_stop_safe(env, spatial, executor)
                        and landed
                        and (reached or advanced or objective or env.stomped)
                        and (moved > 0 or goal.mode == "hold")
                    ),
                    reached=reached,
                    advanced=advanced or objective or env.points() > points,
                    progress=before - distance(env, target),
                    frames=env.steps - start,
                    clearance=clearance,
                    snapshot=snapshot_env_state(env) if save else None,
                    spatial=spatial if save else None,
                    controller=executor if save else None,
                    enemy_gap=enemy_gap,
                    edge_margin=edge,
                    outcome=outcome,
                )
        return Result(frames=192)


def candidates(env, target, proposal):
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    side = target.direction
    seg = tactic_schedule.current(env)
    forbidden = set(seg.get("forbidden", ())) | set(seg.get("avoid", ()))
    choices = []

    def point(mode, x, y):
        if abs(x - cx) <= 256 and abs(y - feet) <= 240:
            choices.append(SkillToken(mode, round(x - cx), round(y - feet)))

    if proposal is not None:
        choices.append(proposal)
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        if i in forbidden or r.width < m["w"] + 4 or r.right < cx - 128 or r.left > cx + 192:
            continue
        lo, hi = r.left + m["w"] / 2 + 2, r.right - m["w"] / 2 - 2
        points = [
            max(lo, min(hi, cx + side * d))
            for d in (4, 8, 16, 24, 32, 48, 72, 96, -24, -48, -72, -96)
        ]
        points += [
            lo,
            hi,
            lo - 2,
            hi + 2,
            # The foot center may reach the support's edge. Restricting it
            # to whole-body insets can omit the only usable takeoff waypoint
            # when vision slightly offsets Mario near a ledge.
            r.left + 2,
            r.right - 2,
            r.left,
            r.right,
            (lo + hi) / 2,
            max(lo, min(hi, env.goal.centerx)),
            max(lo, min(hi, target.center)),
        ]
        if not m["on_ground"]:
            points += [max(lo, min(hi, cx + d)) for d in range(-32, 33, 4)]
        for x in points:
            if abs(x - cx) <= 128:
                point("run", x, r.top)
            if m["on_ground"] and abs(x - cx) <= 128 and -88 <= r.top - feet <= 160:
                point("jump", x, r.top)
    for e in env.enemies:
        if e["dead"] or e["h"] <= 0 or e.get("kind") in ("monster", "piranha_plant"):
            continue
        if abs(e["x"] - cx) <= 128 and m["on_ground"]:
            for x in (e["x"] + e["w"] / 2, e["x"] + 2, e["x"] + e["w"] - 2):
                point("jump", x, e["y"])
    # Preserve the current hold anchor while waiting for a moving destination.
    choices.append(SkillToken("hold", 0, 0))
    return list(dict.fromkeys(choices))


def destination(env, state, proposal):
    if tactic_schedule.current(env)["kind"] == "monster" and env.mario["on_ground"]:
        return monster_destination(env, state)
    if tactic_schedule.current(env)["kind"] == "plant":
        plant = plant_destination(env, state)
        if plant is not None:
            return plant
    target = teacher_target(env)
    airborne = not env.mario["on_ground"]
    if target != training_target(env):
        proposal = None
    mount = mount_destination(env, state, target)
    if mount is not None:
        return mount
    if proposal is not None:
        result = trial(env, state, proposal, target)
        if result.won or (
            result.safe
            and result.clearance >= 8
            and closes_on_enemy(env, target, proposal, result)
            and (proposal.mode != "hold" or _waiting(env, state))
        ):
            return _with_margin(env, state, target, proposal, result)
    # Rank every candidate on label-only trials; confirm in order through the
    # full trial (which replays jumps through vision when it is available).
    # The label-only trials are kept for the margin search that follows.
    winners, progressing, memo = [], [], {}
    for goal in candidates(env, target, proposal):
        if goal == proposal or goal.mode == "hold":
            continue
        result = memo[goal] = _trial(env, state, goal, target)
        if result.won:
            winners.append((_rank(goal, result), goal))
        elif (
            result.safe
            and result.clearance >= 8
            and closes_on_enemy(env, target, goal, result)
            and (result.advanced or result.progress > 0 or airborne)
        ):
            value = (100 if result.advanced else result.progress) / max(1, result.frames)
            progressing.append(((value, result.margin), goal))
    for _, goal in sorted(winners, key=lambda w: w[0], reverse=True):
        confirmed = trial(env, state, goal, target)
        if confirmed.won:
            return _with_margin(env, state, target, goal, confirmed, memo)
    for _, goal in sorted(progressing, key=lambda p: p[0], reverse=True):
        result = trial(env, state, goal, target)
        if (
            result.safe
            and result.clearance >= 8
            and closes_on_enemy(env, target, goal, result)
            and (result.advanced or result.progress > 0 or airborne)
        ):
            return _with_margin(env, state, target, goal, result, memo)
    descent = descent_destination(env, state, target)
    if descent is not None:
        return descent
    hold = SkillToken("hold", 0, 0)
    if any(p.get("moving") for p in env.platforms) or any(not e["dead"] for e in env.enemies):
        if trial(env, state, hold, target).safe:
            return hold
    return None


def _rank(goal, result):
    """Larger is better among versions of one move: the most margin (in 2-pixel
    steps, so near-equal margins tie), then the destination farthest from one
    that fails, then (for a stomp) a landing nearest the enemy's middle, then
    the quickest."""
    return (
        math.floor(result.margin / 2),
        getattr(result, "room", 0.0),
        -getattr(result, "aim_error", 0.0),
        -result.frames,
    )


def _works(result, chosen, hop=False):
    """A version of the chosen move: it achieves the same outcome as safely.

    A ``hop`` lands back on the platform it took off from without winning,
    killing or reaching a route mark: what it achieves is its progress toward
    the target, so a version must make at least as much (else the shortest
    hop, the farthest from every enemy, would always have the most margin).
    """
    if result.outcome != chosen.outcome:
        return False
    if hop and result.progress < chosen.progress - 2:
        return False
    return result.won or (result.safe and result.clearance >= 8)


def _is_hop(env, result):
    """Whether ``result`` lands back where Mario stands now and achieves
    nothing but progress (see _works)."""
    won, support, killed, mark, _ = result.outcome
    here = env.mario.get("_platform") if env.mario["on_ground"] else None
    start = next((i for i, p in enumerate(env.platforms) if p is here), None)
    return (
        not won
        and not killed
        and support is not None
        and support == start
        and mark == (env._tactic_index, env._route_done)
    )


def _versions(env, state, target, goal, result, memo=None):
    """Every jump that makes the same move as ``goal`` (same landing platform,
    same enemies killed, same route mark), with its label-only trial.
    ``memo`` holds label-only trials already run from this same state."""
    memo = {} if memo is None else memo
    found = [(goal, result)]
    tested = [(goal.x, True)]
    cx = env.mario["x"] + env.mario["w"] / 2
    hop = _is_hop(env, result)
    _, support, killed, _, _ = result.outcome
    if killed:
        e = env.enemies[min(killed)]
        span = (e["x"] - 8, e["x"] + e["w"] + 8)
    elif support is not None:
        r = env.platforms[support]["rect"]
        span = (r.left - 4, r.right + 4)
    else:
        span = (-1e9, 1e9)
    for other in candidates(env, target, None):
        # Only jumps that can make the same move: the same direction, the same
        # height, aimed at the same platform or enemy.
        if (
            other.mode != "jump"
            or other == goal
            or abs(other.y - goal.y) > 2
            or (other.x > 0) != (goal.x > 0)
            or not span[0] <= cx + other.x <= span[1]
        ):
            continue
        if other not in memo:
            memo[other] = _trial(env, state, other, target)
        trial_result = memo[other]
        works = _works(trial_result, result, hop)
        tested.append((other.x, works))
        if works:
            found.append((other, trial_result))
    _command_room(found, tested)
    for g, r in found:
        if r.outcome and r.outcome[2]:
            # A stomp: aim at the middle of the enemy it kills.
            e = env.enemies[min(r.outcome[2])]
            middle = e["x"] + e["w"] / 2 - (env.mario["x"] + env.mario["w"] / 2)
            r.aim_error = abs(g.x - middle)
    return found


def _command_room(found, tested):
    """Set each version's room: how far its destination x is from the nearer
    end of the unbroken range of tested destinations that work.

    Several destinations often give the very same jump (a standing jump can
    only go so far). Then the margin ties, and the label is taken from the
    middle of the working range, so a skill output a few pixels off still
    makes the same move. A range ends halfway to the nearest destination that
    fails, or at the last one tested.
    """
    xs = sorted({x for x, _ in tested})
    works = {x: all(w for t, w in tested if t == x) for x in xs}
    room = {}
    i = 0
    while i < len(xs):
        if not works[xs[i]]:
            i += 1
            continue
        j = i
        while j + 1 < len(xs) and works[xs[j + 1]]:
            j += 1
        left = (xs[i] + xs[i - 1]) / 2 if i > 0 else xs[i]
        right = (xs[j] + xs[j + 1]) / 2 if j + 1 < len(xs) else xs[j]
        for x in xs[i : j + 1]:
            room[x] = min(x - left, right - x)
        i = j + 1
    for g, r in found:
        r.room = room.get(g.x, 0.0)


def _with_margin(env, state, target, goal, result, memo=None):
    """The chosen move's version with the most margin (training labels only).

    First the move itself is chosen where an enemy is near ahead
    (_choose_move). A jump is then replaced by the jump that does the same
    thing while staying farthest from enemies and landing farthest from the
    platform's edges. On the ground, a jump that passes an enemy also gets a
    better takeoff when a short run first gives it at least 4 more pixels of
    margin. Runs and holds keep the plan: a run's job is often to reach an
    edge for a takeoff.
    """
    memo = {} if memo is None else memo
    goal, result = _choose_move(env, state, target, goal, result, memo)
    if goal.mode != "jump" or not result.outcome:
        return goal
    if getattr(env, "_action_jump_direction", 0) or not env.mario["on_ground"]:
        takeoff = False  # the lesson is one immediate jump: its takeoff is fixed
    else:
        takeoff = result.enemy_gap < 48
    versions = sorted(
        _versions(env, state, target, goal, result, memo), key=lambda v: _rank(*v), reverse=True
    )
    chosen = None
    hop = _is_hop(env, result)
    for g, r in versions:
        if g == goal:
            chosen = (g, result)
            break
        confirmed = trial(env, state, g, target)
        if _works(confirmed, result, hop):
            chosen = (g, confirmed)
            break
    if chosen is None:
        return goal
    if takeoff:
        run = _better_takeoff(env, state, target, *chosen)
        if run is not None:
            return run
    return chosen[0]


def _enemy_ahead(env, direction) -> bool:
    """Whether a live enemy Mario could jump over or onto is within 128 pixels
    ahead (``direction``: 1 right, -1 left) at about his height."""
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    for e in env.enemies:
        if e["dead"] or e["h"] <= 0 or e.get("kind") in ("monster", "piranha_plant"):
            continue
        ahead = (e["x"] + e["w"] / 2 - cx) * direction
        if 0 < ahead <= 128 and abs(e["y"] + e["h"] - feet) <= 32:
            return True
    return False


def _move_kind(env, result) -> str:
    """What a command (its label-only trial from the current state) does to
    enemies: "stomp" (it kills one), "pass" (Mario ends on the other side of an
    enemy that stays alive) or "other"."""
    if not result.outcome:
        return "other"
    killed, sides = result.outcome[2], result.outcome[4]
    if killed:
        return "stomp"
    cx = env.mario["x"] + env.mario["w"] / 2
    alive = [e for e in env.enemies if not e["dead"]]
    now = tuple(cx < e["x"] + e["w"] / 2 for e in alive)
    return "pass" if sides != now else "other"


def _choose_move(env, state, target, goal, result, memo):
    """Choose between stomping an enemy, jumping over it and waiting first
    (training labels only). Returns (command, its result).

    It applies on the ground with a live enemy near ahead, except in lessons
    that are one given jump. Under max points (kills score), a stomp that works
    now is taught over any other move. Under speed run, a jump over is taught
    over waiting, unless the waiting command lets the same jump pass with at
    least 4 more pixels of margin; and over a stomp. Where no jump over works
    now, a short wait after which one works is taught over a stomp. Which of
    these works is decided by trials, so scenes that look alike get the same
    move.
    """
    if (
        not env.mario["on_ground"]
        or getattr(env, "_action_jump_direction", 0)
        or (result.won and goal.mode != "jump")
        or not _enemy_ahead(env, target.direction)
    ):
        return goal, result
    kind = _move_kind(env, result)
    max_points = getattr(state, "strategy", "speed_run") == "max_points"
    if kind == ("stomp" if max_points else "pass"):
        return goal, result
    wanted = "stomp" if max_points else "pass"
    working = _jumps_that(env, state, target, wanted, memo)
    for g, r in sorted(working, key=lambda v: _rank(*v), reverse=True):
        confirmed = trial(env, state, g, target)
        if not (confirmed.won or confirmed.safe and confirmed.clearance >= 8):
            continue
        if kind == "other" and not max_points:
            if _waiting_helps(env, state, target, goal, g, confirmed):
                return goal, result
        return g, confirmed
    if kind == "stomp" and not max_points:
        wait = _wait_to_pass(env, state, target)
        if wait is not None:
            return wait
    return goal, result


def _jumps_that(env, state, target, kind, memo, direction=None):
    """Jumps toward the target (label-only trials, kept in ``memo``) that work
    and do ``kind`` to an enemy ("stomp" or "pass", _move_kind)."""
    direction = target.direction if direction is None else direction
    found = []
    for other in candidates(env, target, None):
        if other.mode != "jump" or other.x == 0 or (other.x > 0) != (direction > 0):
            continue
        if other not in memo:
            memo[other] = _trial(env, state, other, target)
        r = memo[other]
        works = r.won or (r.safe and r.clearance >= 8 and (r.advanced or r.progress > 0))
        if works and _move_kind(env, r) == kind:
            found.append((other, r))
    return found


def _wait_to_pass(env, state, target):
    """A wait (holding still, or a step of 4 to 32 pixels toward the target)
    after which a jump over the enemy works: the one whose jump over then keeps
    the most margin (in 2-pixel steps; the shorter wait on ties), as (command,
    its result); or None."""
    best, best_rank = None, None
    waits = [SkillToken("hold", 0, 0)]
    waits += [SkillToken("run", d * target.direction, 0) for d in (4, 8, 16, 24, 32)]
    for order, wait in enumerate(waits):
        moved = _trial(env, state, wait, target, save=True)
        if not moved.safe or moved.snapshot is None or moved.enemy_gap < 8:
            continue
        with probe_state(env):
            restore_env_state(env, moved.snapshot)
            following = copy(state)
            following.execution, following.controller = moved.spatial, moved.controller
            following.scene = moved.spatial.previous
            passes = _jumps_that(env, following, target, "pass", {})
        if passes:
            margin = max(min(r.margin, moved.enemy_gap) for _, r in passes)
            rank = (math.floor(margin / 2), -order)
            if best_rank is None or rank > best_rank:
                best, best_rank = (wait, moved), rank
    return best


def _waiting_helps(env, state, target, waiting, jump, result) -> bool:
    """Whether, after the ``waiting`` command, a jump over the enemy (to the
    same place, or with the same reach) passes with at least 4 more pixels of
    margin than ``jump`` does now."""
    moved = _trial(env, state, waiting, target, save=True)
    if not moved.safe or moved.snapshot is None:
        return False
    start = env.mario["x"]
    with probe_state(env):
        restore_env_state(env, moved.snapshot)
        following = copy(state)
        following.execution, following.controller = moved.spatial, moved.controller
        following.scene = moved.spatial.previous
        shift = env.mario["x"] - start
        for g in dict.fromkeys((SkillToken("jump", round(jump.x - shift), jump.y), jump)):
            r = _trial(env, following, g, target)
            works = r.won or (r.safe and r.clearance >= 8)
            if works and _move_kind(env, r) == "pass" and r.margin >= result.margin + 4:
                return True
    return False


def _better_takeoff(env, state, target, jump, result):
    """A short run toward the jump after which the same jump passes the enemy
    with at least 4 more pixels of margin, or None.

    After each run it tries the jump to the same place and the jump with the
    same reach from the new takeoff."""
    side = 1 if jump.x > 0 else -1
    best, margin = None, result.margin + 4
    start = env.mario["x"]
    for travel in (4, 8, 16, 24):
        run = SkillToken("run", side * travel, 0)
        moved = _trial(env, state, run, target, save=True)
        if not moved.safe or moved.snapshot is None or moved.enemy_gap < 8:
            continue
        with probe_state(env):
            restore_env_state(env, moved.snapshot)
            following = copy(state)
            following.execution, following.controller = moved.spatial, moved.controller
            following.scene = moved.spatial.previous
            shift = env.mario["x"] - start
            same_place = SkillToken("jump", round(jump.x - shift), jump.y)
            for g in dict.fromkeys((same_place, jump)):
                r = _trial(env, following, g, target)
                if _works(r, result) and r.margin >= margin:
                    best, margin = run, r.margin
    return best


def descent_destination(env, state, target):
    """Certify a supported edge preparation and its following downward jump.

    A contact correction can end a run before its requested point. That is a
    usable preparation only if Mario remains supported and a replayed next
    jump finishes the objective. Never label an interrupted run on progress
    alone or substitute a policy decision for the missing destination.
    """
    m = env.mario
    support = m.get("_platform")
    if (
        not m["on_ground"]
        or support is None
        or support.get("moving")
        or not getattr(env, "_action_jump_direction", 0)
        or target.kind != "gap"
        or target.platform_index is None
        or any(not e["dead"] for e in env.enemies)
    ):
        return None
    source, far = support["rect"], env.platforms[target.platform_index]["rect"]
    if far.top <= source.top or env.platforms[target.platform_index].get("moving"):
        return None
    half = m["w"] / 2
    # Two pixels of body overlap still support Mario at either takeoff edge.
    takeoff = source.right + half - 2 if target.direction > 0 else source.left - half + 2
    goal = SkillToken("run", round(takeoff - m["x"] - half), 0)
    result = _trial(env, state, goal, target, save=True, visual=state.visual_observer is not None)
    if result.snapshot is None or result.progress <= 0 or result.clearance < 8:
        return None
    with probe_state(env):
        restore_env_state(env, result.snapshot)
        m = env.mario
        overlap = min(m["x"] + m["w"], source.right) - max(m["x"], source.left)
        if not m["on_ground"] or overlap < 2 or not route_available(env):
            return None
        following = copy(state)
        following.execution, following.controller = result.spatial, result.controller
        following.scene = result.spatial.previous
        lo, hi = far.left + half + 2, far.right - half - 2
        for x in (hi if target.direction < 0 else lo, (lo + hi) / 2, lo, hi):
            jump = SkillToken("jump", round(x - m["x"] - half), round(far.top - source.top))
            outcome = trial(env, following, jump, target)
            if outcome.won:
                return goal
    return None


def closes_on_enemy(env, target, goal, result):
    """A stopped approach must gain on a retreating stomp target.

    A button-route waypoint can move only as far as the enemy moves while
    Mario accelerates and brakes. Such commands are individually safe but
    repeat indefinitely without bringing the enemy into jumping range.
    Measure relative progress through the real controller before teaching it.
    """
    if (
        goal.mode != "run"
        or target.kind != "stomp"
        or target.enemy_index is None
        or not env.mario["on_ground"]
        or result.advanced
    ):
        return True
    enemy = env.enemies[target.enemy_index]
    if enemy["dead"] or enemy["speed"] <= 0 or enemy["direction"] != target.direction:
        return True
    return result.progress >= max(2.0, 0.1 * result.frames)


def mount_destination(env, state, target):
    """Certify the takeoff location as well as the raised landing destination."""
    m = env.mario
    if not m["on_ground"] or target.platform_index is None or target.kind != "mount":
        return None
    rect = env.platforms[target.platform_index]["rect"]
    feet, center, half = m["y"] + m["h"], m["x"] + m["w"] / 2, m["w"] / 2
    if rect.top >= feet - 2 or rect.top < feet - 88:
        return None
    side = target.direction
    lo, hi = rect.left + half + 2, rect.right - half - 2
    # Landings across the platform, the middle first: the one farthest from
    # its edges that works is taught.
    landings = list(
        dict.fromkeys(
            (
                (lo + hi) / 2,
                lo + (hi - lo) * 0.25,
                lo + (hi - lo) * 0.75,
                lo if side > 0 else hi,
                hi if side > 0 else lo,
            )
        )
    )

    def landing_jumps(cx, here):
        found = []
        for x in landings:
            goal = SkillToken("jump", round(x - cx), round(rect.top - feet))
            if abs(goal.x) > 128:
                continue
            result = _trial(env, here, goal, target)
            if result.won or (result.safe and result.advanced):
                found.append((_rank(goal, result), goal))
        return sorted(found, key=lambda f: f[0], reverse=True)

    for _, goal in landing_jumps(center, state):
        result = trial(env, state, goal, target)
        if result.won or (result.safe and result.advanced):
            return goal
    support = env.mario.get("_platform")
    if support is None or support.get("moving"):
        return None
    source = support["rect"]
    # Ground waypoints must be supported and allow a jump from rest, including
    # the clearance needed to rise beside an overhead ledge before moving over it.
    edge = rect.left if side > 0 else rect.right
    takeoffs = [edge - side * (half + margin) for margin in (4, 12, 20, 28)]
    takeoffs += [source.right - half - 2 if side > 0 else source.left + half + 2]
    # The takeoff whose following jump lands farthest from the platform's edges.
    best, best_rank = None, None
    for x in dict.fromkeys(takeoffs):
        if (
            not source.left + half + 2 <= x <= source.right - half - 2
            or abs(x - center) < 2
            or abs(x - center) > 128
        ):
            continue
        goal = SkillToken("run", round(x - center), 0)
        result = trial(env, state, goal, target, save=True)
        if not result.safe or result.snapshot is None:
            continue
        with probe_state(env):
            restore_env_state(env, result.snapshot)
            following = copy(state)
            following.execution, following.controller = result.spatial, result.controller
            following.scene = result.spatial.previous
            cx = env.mario["x"] + env.mario["w"] / 2
            for rank, jump in landing_jumps(cx, following):
                confirmed = trial(env, following, jump, target)
                if confirmed.won or (confirmed.safe and confirmed.advanced):
                    if best_rank is None or rank > best_rank:
                        best, best_rank = goal, rank
                    break
    return best


def _waiting(env, state):
    from .teacher_tokens import schedule_stance

    return schedule_stance(env, state)[0] == "hold_area"


def teacher_target(env):
    """Keep uncollected strategy rewards ahead of the terminal destination."""
    from .local_traversal import LocalObjective

    target = training_target(env)
    if env.points() >= env._strategy_objective.get("points", 0):
        return target
    if (
        env._tactic_index != len(env._tactics) - 1
        or tactic_schedule.next_route_platform(env) is not None
    ):
        return target
    m = env.mario
    center = m["x"] + m["w"] / 2
    coins = [c["rect"] for c in env.coins if not c["collected"]]
    if coins:
        coin = min(coins, key=lambda r: abs(r.centerx - center) + abs(r.bottom - (m["y"] + m["h"])))
        supports = [
            (i, p["rect"])
            for i, p in enumerate(env.platforms)
            if p["rect"].left <= coin.centerx <= p["rect"].right
            and 0 <= p["rect"].top - coin.bottom <= 4
        ]
        if supports:
            i, r = supports[0]
            side = 1 if coin.centerx >= center else -1
            if m["y"] + m["h"] < r.top - 2 or m["y"] + m["h"] > r.top + 2:
                return LocalObjective("mount", r.left, r.right, r.top, i, direction=side)
            return LocalObjective("coin", coin.centerx - 2, coin.centerx + 2, r.top, direction=side)
    enemies = [
        (i, e) for i, e in enumerate(env.enemies) if not e["dead"] and e.get("stompable", True)
    ]
    if enemies:
        i, e = min(enemies, key=lambda pair: abs(pair[1]["x"] - center))
        return LocalObjective(
            "stomp",
            e["x"],
            e["x"] + e["w"],
            e["y"],
            enemy_index=i,
            direction=1 if e["x"] >= center else -1,
        )
    return target


def route_available(env):
    """Do not scroll an unfinished route or required reward off the left edge."""
    edge = env.camera_x
    half = env.mario["w"] / 2
    if env.goal.right <= edge:
        return False
    for offset, segment in enumerate(env._tactics[env._tactic_index :]):
        route = segment.get("route", ())
        if offset == 0:
            route = route[env._route_done :]
        if any(env.platforms[i]["rect"].right - half < edge + half for i in route):
            return False
        end = segment.get("end", {})
        if segment["direction"] < 0 and end.get("reach_x", edge) < edge:
            return False
    needed = env._strategy_objective.get("points", 0) - env.points()
    if needed > 0:
        available = sum(not c["collected"] and c["rect"].right > edge for c in env.coins)
        available += sum(
            not e["dead"]
            and e.get("stompable", True)
            and e.get("patrol_max", e["x"]) + e["w"] > edge
            for e in env.enemies
        )
        if available < needed:
            return False
    return True


def boxed_with_enemy(env, clearance):
    m = env.mario
    if not m["on_ground"] or clearance >= 64:
        return False
    return any(
        p["rect"].left < m["x"] + m["w"]
        and p["rect"].right > m["x"]
        and 0 < m["y"] - p["rect"].bottom < 48
        for p in env.platforms
    )


def on_plant_mouth(env):
    """A temporarily hidden plant is not a persistent supported destination."""
    m = env.mario
    return any(
        e.get("kind") == "piranha_plant"
        and abs(m["y"] + m["h"] - e["pipe_top"]) < 2
        and m["x"] + m["w"] > e["x"] - 2
        and m["x"] < e["x"] + e["w"] + 2
        for e in env.enemies
    )


def plant_stop_safe(env, spatial, executor):
    """A pipe landing needs room to brake outside the mouth of its plant."""
    m = env.mario
    if not any(
        e.get("kind") == "piranha_plant"
        and abs(m["y"] + m["h"] - e["pipe_top"]) < 2
        and abs(m["x"] - e["x"]) < 48
        for e in env.enemies
    ):
        return True
    if on_plant_mouth(env):
        return False
    with probe_state(env):
        feedback = deepcopy(spatial)
        control = SMBExecutor(last_button=executor.last_button)
        scene = scene_from_labels(env.scene_labels())
        plan = feedback.begin(SkillToken("hold", 0, 0), scene)
        control.start(plan, flight=feedback.flight, travel=feedback.travel)
        for frame in range(6):
            if frame:
                control.end("hold_recheck")
                plan = feedback.begin(SkillToken("hold", 0, 0), scene)
                control.start(plan, flight=feedback.flight, travel=feedback.travel)
            button = control.press(scene)
            feedback.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            if done or truncated or on_plant_mouth(env):
                return bool(env._goal_credited)
            scene = scene_from_labels(env.scene_labels())
            feedback.observe(scene)
    return True


def plant_destination(env, state):
    """Time the actual spatial crossing, including its stopped takeoff waypoint."""
    cached = state.notes.get("plant_destination")
    if cached is not None and cached[0] == env.steps:
        return cached[1]
    target = teacher_target(env)
    m = env.mario
    cx, feet, half = m["x"] + m["w"] / 2, m["y"] + m["h"], m["w"] / 2
    plant = next(
        (e for e in env.enemies if e.get("kind") == "piranha_plant" and e["x"] + e["w"] >= m["x"]),
        None,
    )
    if plant is None:
        return None
    pipe = next(
        (
            p["rect"]
            for p in env.platforms
            if p["rect"].left <= plant["x"]
            and p["rect"].right >= plant["x"] + plant["w"]
            and abs(p["rect"].top - plant["pipe_top"]) < 1
        ),
        None,
    )
    if pipe is None:
        return None
    goals = []
    far = pipe.right + half + 8
    floor = next(
        (
            p["rect"]
            for p in env.platforms
            if p["rect"].left <= far <= p["rect"].right and p["rect"].top > pipe.top
        ),
        None,
    )
    if floor is not None and abs(far - cx) <= 128:
        goals.extend(
            [
                SkillToken(
                    "jump" if m["on_ground"] else "run", round(far - cx), round(floor.top - feet)
                )
            ]
        )
        if feet <= pipe.top + 2:
            goals.append(SkillToken("run", round(far - cx), round(floor.top - feet)))
    if m["on_ground"] and cx < pipe.right - half and abs(pipe.right - half - cx) <= 128:
        goals.append(SkillToken("jump", round(pipe.right - half - cx), round(pipe.top - feet)))
    if m["on_ground"] and feet > pipe.top + 2 and abs(pipe.left + half - cx) <= 128:
        goals.append(SkillToken("jump", round(pipe.left + half - cx), round(pipe.top - feet)))
    support = m.get("_platform")
    if support is not None and feet > pipe.top + 2 and cx < pipe.left - half - 2:
        approach = min(cx + 96, pipe.left - half - 2)
        goals.append(SkillToken("run", round(approach - cx), 0))
    # Of the crossings that work, the one passing farthest from the plant.
    working = []
    for goal in goals:
        result = _trial(env, state, goal, target)
        if result.won or result.safe and result.clearance >= 8:
            working.append((_rank(goal, result), goal))
    for _, goal in sorted(working, key=lambda w: w[0], reverse=True):
        result = trial(env, state, goal, target)
        if result.won or result.safe and result.clearance >= 8:
            state.notes["plant_destination"] = (env.steps, goal)
            return goal
    hold = SkillToken("hold", 0, 0)
    if trial(env, state, hold, target).safe:
        state.notes["plant_destination"] = (env.steps, hold)
        return hold
    return None


def monster_destination(env, state):
    """Approach, wait and retreat separately; a jump must clear the monster."""
    cached = state.notes.get("monster_destination")
    if cached is not None and cached[0] == env.steps:
        return cached[1]
    segment = tactic_schedule.current(env)
    index = segment["end"].get("past_enemy")
    if index is None:
        return None
    enemy, m = dict(env.enemies[index]), dict(env.mario)
    support = m.get("_platform")
    if support is None:
        support = next(
            (
                p
                for p in env.platforms
                if p["rect"].left <= m["x"] + m["w"] / 2 <= p["rect"].right
                and abs(p["rect"].top - m["y"] - m["h"]) <= 2
            ),
            None,
        )
    if support is None:
        return None
    floor = support["rect"]
    side = segment["direction"]
    center, half = m["x"] + m["w"] / 2, m["w"] / 2
    target = teacher_target(env)

    def remember(goal):
        state.notes["monster_destination"] = (env.steps, goal)
        return goal

    def supported(offset):
        return floor.left + half + 2 <= center + offset <= floor.right - half - 2

    gap = (enemy["x"] - m["x"] - m["w"]) if side > 0 else m["x"] - enemy["x"] - enemy["w"]
    line = segment.get("keep_behind", floor.right if side > 0 else floor.left)
    near_opening = (enemy["x"] - line) * side <= 16
    if gap < 96 and near_opening:
        # Of the jumps that clear the monster, the one passing farthest from it.
        working = []
        for travel in (48, 64, 80, 96, 112, 128, 32):
            offset = side * travel
            if not supported(offset):
                continue
            goal = SkillToken("jump", offset, 0)
            result = _trial(env, state, goal, target)
            if result.won or (result.safe and result.advanced and result.clearance >= 8):
                working.append((_rank(goal, result), goal))
        for _, goal in sorted(working, key=lambda w: w[0], reverse=True):
            result = trial(env, state, goal, target)
            if result.won or (result.safe and result.advanced and result.clearance >= 8):
                return remember(goal)
    # Reveal the monster with small supported approaches. Do not spend the
    # runway needed to brake and retreat outside the low tunnel.
    sleeping = not enemy.get("awake") and enemy["x"] >= env.camera_x + env.width
    if sleeping:
        offset = side * 8
        endpoint = center + offset
        if (
            supported(offset)
            and (line - endpoint) * side >= half + 4
            and endpoint - max(env.camera_x, floor.left) >= 40
        ):
            goal = SkillToken("run", offset, 0)
            result = trial(env, state, goal, target)
            if result.safe and result.clearance >= 20:
                return remember(goal)
    coming = enemy["direction"] == -side and enemy["speed"] > 0
    if (coming and gap < 40) or (center + side * half - line) * side > -8:
        for travel in (24, 16, 32, 48, 64, 80, 96, 8):
            offset = -side * travel
            if not supported(offset) or center + offset < env.camera_x + half + 2:
                continue
            goal = SkillToken("run", offset, 0)
            result = trial(env, state, goal, target)
            if result.safe and result.clearance >= 8:
                return remember(goal)
    hold = SkillToken("hold", 0, 0)
    result = trial(env, state, hold, target)
    return remember(hold) if result.safe and result.clearance >= 8 else None
