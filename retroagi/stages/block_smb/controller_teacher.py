"""Training-only certification and repair of spatial teacher destinations.

Every candidate goes through the unmodified ground/flight controllers. Hidden
geometry is used only to propose labels and score simulator trials.

Among the commands that work, the teacher labels the one with the most room
(Result.distances): the farthest Mario stays from every enemy he does not
stomp and from every pit edge, before, during and after a jump, and, where he
comes to stand on a platform, the farthest from its edges. Commands are
compared by their closest threat, then their next closest, and so on, so a
threat they all share (the pit edge Mario starts beside) does not hide the
others. A command that only just works is not taught when a safer one does
the same.
"""

import math
from collections import defaultdict
from copy import copy, deepcopy
from dataclasses import dataclass, field
from typing import Optional

from retroagi.core.smb_agent import LandingWatch
from retroagi.core.smb_coaching import probe_state, training_target
from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.smb_trajectory import VisualTracks, aimed_goal, best_holds, hold_paths
from retroagi.core.tokens import SKILL_X, SKILL_Y, SkillToken

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
    # Every threat's closest distance over the trial (pixels): each enemy he
    # did not kill ("enemy", index), each pit edge ("pit", x, y), and the
    # nearer edge of the platform he ended on ("landing",).
    distances: dict = field(default_factory=dict)
    # Where Mario's feet ended, from where they started (pixels; x right, y down).
    end_dx: float = 0.0
    end_dy: float = 0.0
    # With record=True: Mario's body (x0, y0, x1, y1) before the first frame
    # and after every frame, in world pixels.
    path: list = field(default_factory=list)
    # What the command achieved: the goal, the platform he ended on, the
    # enemies it killed, the route mark reached and which side of each other
    # enemy he ended on. Commands with the same outcome are versions of the
    # same move.
    outcome: tuple = field(default=())
    # For a command that wins: how far Mario's middle ends from the goal
    # box's middle (pixels); 0 otherwise. Of the moves that win, the one
    # ending nearest the box's middle is taught (_rank): the lowest climb
    # onto a wide step had landed on its near edge (user, 2026-10-14).
    goal_offset: float = 0.0

    @property
    def margin(self) -> float:
        """The closest threat: enemy, pit edge or landing edge (pixels)."""
        values = [self.enemy_gap, *self.distances.values()]
        if self.edge_margin is not None:
            values.append(self.edge_margin)
        return min(values)


# Distances beyond this many pixels no longer matter when comparing commands.
FAR = 64


def _distance_key(result) -> tuple:
    """A command's threats from the closest up, in 2-pixel steps (at most FAR
    pixels): comparing these tuples prefers the command whose closest threat
    is farthest, then its next closest, and so on."""
    values = sorted(min(FAR, d) // 2 for d in result.distances.values())
    if not values:
        values = [min(FAR, result.margin) // 2]
    return tuple(int(v) for v in values[:12]) + (FAR // 2,) * max(0, 12 - len(values))


def _pit_lips(env) -> list:
    """The top corners of platforms past which Mario would fall into a pit:
    no platform he may stand on is beside or below that edge (the lesson's
    forbidden platforms count as pits; the world's ends are walls)."""
    forbidden = set(tactic_schedule.current(env).get("forbidden", ()))
    rects = [p["rect"] for p in env.platforms]
    lips = []
    for i, r in enumerate(rects):
        for x, outside in ((r.left, r.left - 1), (r.right, r.right)):
            if outside < 0 or outside >= env.world_width:
                continue
            caught = any(
                j != i
                and j not in forbidden
                and q.left <= outside < q.right
                and q.bottom > r.top - 2
                for j, q in enumerate(rects)
            )
            if not caught:
                lips.append((x, r.top))
    return lips


class _Watch:
    """The closest Mario's body comes, over a trial, to each live enemy and to
    each pit edge (pixels)."""

    def __init__(self, env):
        self.enemies: dict = {}
        self.pits: dict = {}
        self._lips = None if any(p.get("moving") for p in env.platforms) else _pit_lips(env)
        self.observe(env)

    def observe(self, env):
        _box_gaps(env, self.enemies)
        m = env.mario
        for lx, ly in self._lips if self._lips is not None else _pit_lips(env):
            dx = max(lx - (m["x"] + m["w"]), m["x"] - lx, 0)
            dy = max(ly - (m["y"] + m["h"]), m["y"] - ly, 0)
            key = ("pit", lx, ly)
            self.pits[key] = min(self.pits.get(key, 999.0), math.hypot(dx, dy))


def _box_gaps(env, gaps):
    """Lower each live enemy's closest distance to Mario's box (pixels)."""
    m = env.mario
    for i, e in enumerate(env.enemies):
        if e["dead"] or e["h"] <= 0:
            continue
        dx = max(e["x"] - (m["x"] + m["w"]), m["x"] - (e["x"] + e["w"]), 0)
        dy = max(e["y"] - (m["y"] + m["h"]), m["y"] - (e["y"] + e["h"]), 0)
        gaps[i] = min(gaps.get(i, 999.0), math.hypot(dx, dy))


def _measured(env, watch, alive, mark):
    """(enemy_gap, edge_margin, outcome, distances) at the end of a trial
    (``watch``: its _Watch)."""
    killed = frozenset(i for i in alive if env.enemies[i]["dead"])
    gaps = watch.enemies
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
    distances = {("enemy", i): g for i, g in gaps.items() if i not in killed}
    distances.update(watch.pits)
    if edge is not None:
        distances[("landing",)] = edge
    return enemy_gap, edge, outcome, distances


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


def _trial(env, state, goal, target, *, save=False, visual=False, record=False):
    def render():
        return type(env).render(env)

    with probe_state(env):
        spatial = deepcopy(state.execution) if state.execution is not None else SpatialFeedback()
        scene = state.scene or scene_from_labels(env.scene_labels())
        if state.execution is None:
            spatial.motion = deepcopy(env.motion)
            spatial.observe(scene)
        # The teacher's trials aim at walkers where the simulator moves them,
        # never by the learner's forecasts.
        spatial.tracks.exact = exact_positions(env)
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
        watch = _Watch(env)
        path = [_body(env)] if record else []
        def finished():
            enemy_gap, edge, outcome, distances = _measured(env, watch, alive, mark)
            m = env.mario
            offset = (
                abs(m["x"] + m["w"] / 2 - env.goal.centerx)
                if env._goal_credited and env.goal is not None
                else 0.0
            )
            return Result(
                safe=env._goal_credited,
                won=env._goal_credited,
                frames=env.steps - start,
                enemy_gap=enemy_gap,
                edge_margin=edge,
                outcome=outcome,
                distances=distances,
                end_dx=env.mario["x"] - x,
                end_dy=env.mario["y"] + env.mario["h"] - feet,
                path=path,
                goal_offset=offset,
            )

        for _ in range(192):
            button = executor.press(scene)
            spatial.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            watch.observe(env)
            if record:
                path.append(_body(env))
            if done or truncated:
                return finished()
            if visual or unsettled_contact:
                scene = observer.observe([render()])[0]
                unsettled_contact = scene.mario.on_something != env.mario["on_ground"]
            else:
                scene = scene_from_labels(env.scene_labels())
            spatial.observe(scene)
            seen_landing = landing.landed(scene)
            ended = (
                seen_landing or executor.finished or executor.reconsider or spatial.arrived()
            )
            stalled = spatial.stalled and (
                ended or spatial.observed_frames >= spatial.positions.maxlen
            )
            if ended or stalled:
                # Vision sees a stomp's landing a frame or two before the
                # simulator makes it (feet within STOMP_ROWS of the enemy's
                # top): play on, as the episode does, until the simulator
                # lands Mario or the stomp happens.
                for _ in range(SETTLE_FRAMES if seen_landing else 0):
                    if env.mario["on_ground"] or env.stomped:
                        break
                    _, _, done, truncated, _ = env.step(0)
                    watch.observe(env)
                    if record:
                        path.append(_body(env))
                    if done or truncated:
                        return finished()
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
                enemy_gap, edge, outcome, distances = _measured(env, watch, alive, mark)
                return Result(
                    safe=bool(
                        route_available(env)
                        and not boxed_with_enemy(env, clearance)
                        and next_action_safe(env, state, spatial, executor, scene)
                        # A run-up does not stop: the jump it carries into is
                        # judged with it (_certify), not a brake after it.
                        and (goal.carry or stop_safe(env, spatial, executor))
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
                    distances=distances,
                    end_dx=env.mario["x"] - x,
                    end_dy=env.mario["y"] + env.mario["h"] - feet,
                    path=path,
                )
        return Result(frames=192)


def _body(env):
    m = env.mario
    return (m["x"], m["y"], m["x"] + m["w"], m["y"] + m["h"])


# ── Dense search (training labels only) ───────────────────────────────────────
#
# No teacher picks a destination from a sparse list. A jump, replayed open loop
# (smb_trajectory.REPLAN_IN_FLIGHT off), is set at takeoff by its steering and
# how long the button is held. Every destination within REACH pixels, at every
# height a jump can land at, maps to one of these few dozen jumps by the
# executor's own rule (smb_trajectory.best_holds), so trying each jump once
# evaluates every destination (_jump_table). Runs, takeoffs and waits are
# screened at every pixel: a jump path measured once is moved along the floor,
# the enemies moved by their patrol rule, and the best are certified by trials.

REACH = 128
# Above this speed (pixels a frame) the jumps from here are not those Mario
# makes after a run-up or wait, which end at rest (_takeoff_search).
MOVING = 0.5
# Frames a trial plays on after vision sees a landing the simulator has not
# made yet (_trial).
SETTLE_FRAMES = 4
# Screened options certified by a trial, best first.
CERTIFY = 6
# No run shorter than this is proposed by the general search (_run_options):
# such a run creeps. A run-up or wait before a jump can be any length: the
# jump is remembered and taken at the next decision (_commit), so it is not
# re-planned a pixel at a time.
MIN_RUN = 8
# A run off a ledge onto a lower floor aims at least this far past the
# ledge's end (pixels beyond Mario's half width): a run aimed 5 pixels past
# it, Mario's body just clear, stopped 3 pixels short through vision and he
# stood on the ledge's corner.
DROP_MARGIN = 4
# Running first must give the jump this much more room than jumping now.
GAIN = 4


@dataclass
class JumpOption:
    """One jump Mario can make from where he stands: the command taught for it
    (the destination of this jump nearest where it lands; for a stomp, the
    enemy's middle), its trial with the path, and every destination that
    makes this very jump."""

    command: SkillToken
    result: Result
    members: list
    hold: int = 0
    direction: int = 0
    # The jump's predicted path with nothing in its way (world boxes, the
    # start first): what the same jump does from elsewhere on the floor.
    free_path: Optional[list] = None
    # Whether ``command`` names where the jump ends from here (_describes):
    # only then can it be taught from here; one that hits something on its
    # way may still be made from elsewhere on the floor (_takeoff_search).
    described: bool = True


def _landing_heights(env, feet):
    """Every height a jump can land at, from Mario's feet: the platforms' and
    the live enemies' tops (pixels)."""
    tops = {round(p["rect"].top - feet) for p in env.platforms}
    tops |= {round(e["y"] - feet) for e in env.enemies if not e["dead"] and e["h"] > 0}
    return sorted(t for t in tops if SKILL_Y[0] <= t <= SKILL_Y[-1])


def _plan_goal(tracks, box, x, y, scene, motion, direction):
    """The point a jump to (x, y) from Mario's feet aims at: plan_flight's
    rule (a walker under it is aimed at where it will be when the jump lands:
    smb_trajectory.aimed_goal)."""
    goal = ((box[0] + box[2]) / 2 + x, box[3] + y)
    targets = [
        t
        for t in tracks.tracks
        if t.kind in ("walker", "platform")
        and t.box[0] - 4 <= goal[0] <= t.box[2] + 4
        and abs(goal[1] - t.box[1]) <= 4
    ]
    if len(targets) != 1:
        return goal
    t = targets[0]
    offset = (goal[0] - (t.box[0] + t.box[2]) / 2, goal[1] - t.box[1])
    return aimed_goal(t, offset, tracks, scene, motion, box, direction)


def exact_positions(env):
    """VisualTracks.exact for training: where the simulator will have moved
    the walker a track follows (matched by its box, within 8 pixels), a number
    of frames from now, in the scene's pixels; None when no walker matches.
    The teacher is omniscient (user, 2026-10-14): its trials, and the agent's
    executor in training, aim stomps by this, not by the learner's forecast."""

    def exact(track, frames):
        camera = int(env.camera_x)
        best = None
        for i, e in enumerate(env.enemies):
            if e["dead"] or e["h"] <= 0:
                continue
            d = abs(e["x"] - camera - track.box[0]) + abs(e["y"] - track.box[1])
            if d <= 8 and (best is None or d < best[0]):
                best = (d, i)
        if best is None:
            return None
        boxes = _enemy_paths(env, frames)[best[1]]
        b = boxes[min(frames, len(boxes) - 1)]
        return (b[0] - camera, b[1], b[2] - camera, b[3])

    return exact


def _path_landing(path, rects):
    """Where a body following ``path`` (world boxes, the start first) first
    comes down onto a platform's top: (frame, its middle's x displacement,
    the top's height from the start's feet, the platform's index), or None."""
    for t in range(1, len(path)):
        x0, _, x1, y1 = path[t]
        before = path[t - 1][3]
        if y1 < before:
            continue
        for index, r in enumerate(rects):
            if before <= r.top <= y1 and x1 > r.left and x0 < r.right:
                middle = (x0 + x1) / 2 - (path[0][0] + path[0][2]) / 2
                return t, middle, r.top - path[0][3], index
    return None


def _nearest(members, x, y):
    return min(members, key=lambda g: (abs(g.x - x) + 2 * abs(g.y - y), abs(g.x), g.x))


def _table_trial(env, state, goal, target, memo):
    key = ("recorded", goal)
    if key not in memo:
        memo[key] = memo[goal] = _trial(env, state, goal, target, record=True)
    return memo[key]


def _jump_groups(env, state, memo, direction):
    """The jumps Mario can make toward ``direction`` from here, before any
    trial: [(hold, the destinations that make it, its predicted landing,
    how high it rises, its predicted path)] (_path_landing of the executor's
    own predicted path, or None; the rise in pixels above Mario's feet; the
    path in world boxes, the start first, with nothing in its way)."""
    key = ("groups", direction)
    if key in memo:
        return memo[key]
    groups_out = memo[key] = []
    m = env.mario
    scene = state.scene or scene_from_labels(env.scene_labels())
    if (
        not m["on_ground"]
        or scene.mario.box is None
        or scene.mario.support == "air"
        or not scene.mario.on_something
    ):
        return groups_out
    spatial = state.execution
    # Before vision has seen Mario's speed (one picture cannot show it), the
    # executor takes off with no speed and plans the rest of the jump again
    # once it has (Flight.replan): the jump made is the one his true speed
    # gives, which the simulator knows (training labels only).
    known = spatial is not None and getattr(spatial, "motion_ready", False)
    motion = copy(spatial.motion if known else env.motion)
    tracks = getattr(spatial, "tracks", None) or VisualTracks()
    box = scene.mario.box
    start = _body(env)
    rects = [p["rect"] for p in env.platforms]
    heights = _landing_heights(env, start[3])
    xs = [0] if direction == 0 else range(direction, direction * (REACH + 1), direction)
    destinations = [(x, y) for y in heights for x in xs]
    if not destinations:
        return groups_out
    paths = hold_paths(scene, motion, box, direction)
    goals = [_plan_goal(tracks, box, x, y, scene, motion, direction) for x, y in destinations]
    groups = defaultdict(list)
    for (x, y), (hold, _, _) in zip(
        destinations, best_holds(scene, motion, box, goals, direction, paths)
    ):
        groups[hold].append(SkillToken("jump", x, y))
    for hold, members in sorted(groups.items()):
        world = [tuple(start[k] + b[k] - box[k] for k in range(4)) for b in paths[hold - 1]]
        rise = start[3] - min(b[3] for b in world) if world else 0
        free = [start, *world]
        groups_out.append((hold, members, _path_landing(free, rects), rise, free))
    return groups_out


def _jump_table(
    env,
    state,
    target,
    memo,
    directions=(-1, 0, 1),
    lands_on=None,
    rise=None,
    described_only=True,
):
    """Every jump Mario can make from here, each tried once: [JumpOption].

    Every destination within REACH pixels (each pixel in x) at every height a
    jump can land at maps to its jump by the executor's own takeoff rule; the
    jumps nothing maps to cannot be commanded. Empty unless Mario stands.
    ``lands_on``: only the jumps predicted (before trying them) to land on
    one of these platforms (indices); ``rise``: only those rising at least
    this many pixels; for searches that need no other. With
    ``described_only`` (the default), only the jumps a destination names
    from here (JumpOption.described).
    """
    table = []
    for direction in directions:
        for hold, members, landing, height, free in _jump_groups(env, state, memo, direction):
            if lands_on is not None and (landing is None or landing[3] not in lands_on):
                continue
            if rise is not None and height < rise:
                continue
            key = ("option", direction, hold)
            if key not in memo:
                first = (
                    _nearest(members, landing[1], landing[2])
                    if landing
                    else members[len(members) // 2]
                )
                result = _table_trial(env, state, first, target, memo)
                command = _aim(env, result, members)
                if command != first:
                    aimed = _table_trial(env, state, command, target, memo)
                    if aimed.outcome == result.outcome:
                        result = aimed
                    else:
                        command = first
                memo[key] = JumpOption(
                    command,
                    result,
                    members,
                    hold,
                    direction,
                    free,
                    _describes(env, command, result),
                )
            if memo[key].described or not described_only:
                table.append(memo[key])
    return table


def _landing_point(env, result) -> tuple:
    """Where a jump's trial ends, from Mario's feet (pixels): for a stomp, the
    point on the top of the enemy it came down on (the trial ends at the
    kill, inside the enemy's box by as far as Mario fell that frame)."""
    killed = result.outcome[2] if result.outcome else ()
    if killed:
        feet = env.mario["y"] + env.mario["h"]
        return result.end_dx, min(env.enemies[i]["y"] for i in killed) - feet
    return result.end_dx, result.end_dy


def _named(members, dx, dy) -> bool:
    """Whether one of a jump's destinations names a landing at (dx, dy), as
    closely as _describes requires."""
    return any(abs(g.x - dx) <= 4 and abs(g.y - dy) <= 2 for g in members)


def _describes(env, command, result) -> bool:
    """Whether ``command`` names where its jump ends: within 4 pixels in x and
    2 in height of the landing (for a stomp, the point on the enemy). A jump
    that no such command makes is not taught: its label would name a point
    it does not reach (a climb labelled as a jump to the floor's height)."""
    x, y = _landing_point(env, result)
    return abs(command.x - x) <= 4 and abs(command.y - y) <= 2


def _aim(env, result, members):
    """The destination taught for a jump: of the destinations that make it,
    the one nearest where it lands (for a stomp, where it lands on the enemy,
    which a walking enemy has moved to)."""
    return _nearest(members, *_landing_point(env, result))


def _enemy_paths(env, frames):
    """Each live enemy's box (world pixels) for each of the next ``frames``
    frames by its patrol rule (env._update_enemy): {index: [box]}."""
    out = {}
    for i, e in enumerate(env.enemies):
        if e["dead"] or e["h"] <= 0:
            continue
        x, d = e["x"], e["direction"]
        moves = e.get("kind") != "piranha_plant" and (e.get("kind") != "monster" or e.get("awake"))
        boxes = []
        for _ in range(frames + 1):
            boxes.append((x, e["y"], x + e["w"], e["y"] + e["h"]))
            if moves:
                x += e["speed"] * d
                if x <= e["patrol_min"]:
                    x, d = e["patrol_min"], 1
                elif x >= e["patrol_max"]:
                    x, d = e["patrol_max"], -1
        out[i] = boxes
    return out


def _run_frames(motion, distance):
    """Frames a run of ``distance`` pixels takes from ``motion`` (accelerating,
    then braking to stop there), by the NES motion model."""
    from retroagi.core.smb_ground_control import stopping_distance

    motion = copy(motion)
    direction = 1 if distance > 0 else -1
    moved, frames = 0.0, 0
    while frames < 300:
        left = abs(distance) - moved
        brake = left <= abs(stopping_distance(motion)) + 1
        if brake and motion.x_speed == 0:
            break
        dx, _, _ = motion.advance(
            direction=0 if brake else direction, jump=False, grounded=True, y=0, run=True
        )
        moved += dx * direction
        frames += 1
    return frames


def _screen(env, path, shift, delay, enemies, lips, rects, solid=()):
    """A recorded path (world boxes, the start first) moved ``shift`` pixels
    along x and started ``delay`` frames later: (distances, landing platform
    index, killed enemies, where it lands from its start: (middle's x, the
    top's height)), or None when it hits an enemy, passes through one of
    the ``solid`` rects (a predicted path with nothing in its way, which the
    real jump would bump into) or ends without landing. Distances use the
    keys of Result.distances."""
    distances, killed = {}, set()
    before = path[0][3]
    for t, (x0, y0, x1, y1) in enumerate(path):
        x0, x1 = x0 + shift, x1 + shift
        landing = t and y1 >= before
        for r in solid:
            inside = x1 > r.left and x0 < r.right and y1 > r.top and y0 < r.bottom
            if inside and not (landing and before <= r.top):
                return None
        for lx, ly in lips:
            d = math.hypot(max(lx - x1, x0 - lx, 0), max(ly - y1, y0 - ly, 0))
            if d < distances.get(("pit", lx, ly), 999.0):
                distances[("pit", lx, ly)] = d
        for i, boxes in enemies.items():
            if i in killed:
                continue
            ex0, ey0, ex1, ey1 = boxes[min(delay + t, len(boxes) - 1)]
            dx, dy = max(ex0 - x1, x0 - ex1, 0), max(ey0 - y1, y0 - ey1, 0)
            if dx == 0 and dy == 0:
                if t and y1 >= before and before <= ey0 + 2:
                    killed.add(i)  # coming down onto it: a stomp
                    continue
                return None
            d = math.hypot(dx, dy)
            if d < distances.get(("enemy", i), 999.0):
                distances[("enemy", i)] = d
        if t and y1 >= before:
            for index, r in enumerate(rects):
                if before <= r.top <= y1 and x1 > r.left and x0 < r.right:
                    distances[("landing",)] = min(x0 - r.left, r.right - x1)
                    for i in killed:
                        distances.pop(("enemy", i), None)
                    middle = (x0 + x1) / 2 - shift - (path[0][0] + path[0][2]) / 2
                    return distances, index, killed, (middle, r.top - path[0][3])
        before = y1
    return None


def _run_path(start, travel, frames):
    """Mario's body along a run of ``travel`` pixels over ``frames`` frames
    (moving evenly), the start first."""
    frames = max(1, frames)
    return [
        (start[0] + travel * t / frames, start[1], start[2] + travel * t / frames, start[3])
        for t in range(frames + 1)
    ]


def _combine(*parts):
    """Each threat's closest distance over several steps."""
    out = {}
    for distances in parts:
        for key, value in distances.items():
            out[key] = min(out.get(key, 999.0), value)
    return out


# Room from every threat beyond this many pixels is enough: among moves with
# enough room, the lowest jump is taught (then the shortest run-up), not the
# highest, which keeps the most room (user, 2026-10-13: a jump should not be
# higher than it needs to be).
ENOUGH_ROOM = 16
# From the edges of the surface Mario lands on, less is enough: a pipe or
# step top is 20 to 40 pixels wide, and landing nearer its middle is no
# reason to step back first (Mario is 10 pixels wide).
ENOUGH_LANDING = 6


def _enough(key, d) -> float:
    """A distance as far as it matters: capped at ENOUGH_ROOM, or
    ENOUGH_LANDING for the landing surface's edges."""
    return min(ENOUGH_LANDING if key == ("landing",) else ENOUGH_ROOM, d)


def _room_key(distances) -> tuple:
    """_distance_key with every distance capped where it is enough (_enough):
    the closest first, in 2-pixel steps."""
    values = sorted(_enough(k, d) // 2 for k, d in distances.items())
    return tuple(int(v) for v in values[:12]) + (ENOUGH_ROOM // 2,) * max(0, 12 - len(values))


def _rise(path) -> float:
    """How far a path (world boxes, the start first) rises above its start's
    feet (pixels); 0 for none."""
    return path[0][3] - min(b[3] for b in path) if len(path) else 0.0


def _carry_run(motion, shift):
    """A run-up of ``shift`` pixels from ``motion`` that keeps its speed
    (SkillToken.carry): (its frames, Mario's motion as it ends)."""
    motion = copy(motion)
    direction = 1 if shift > 0 else -1
    moved, frames = 0.0, 0
    while moved < abs(shift) and frames < 300:
        dx, _, _ = motion.advance(
            direction=direction, jump=False, grounded=True, y=0, run=direction > 0
        )
        moved += dx * direction
        frames += 1
    return frames, motion


def _takeoff_search(
    env, state, target, memo, direction, travels, wanted, *, waits=(), rise=None, run_waits=()
):
    """Every jump toward ``direction``: from here (the jump table's), after a
    run-up of each travel in ``travels`` toward ``direction`` (pixels; at
    least MIN_RUN), and after each wait of ``waits`` frames in place;
    screened against the pits and the enemies moved by their patrol rule.
    With ``run_waits``, the run-ups are also screened started after each of
    those waits in place (for _pit_wait: screened only, never certified;
    their entries have both a travel and a wait).

    A run-up keeps its speed (SkillToken.carry): the jumps after it are
    predicted from where it ends, at the speed it ends with (_carry_run,
    smb_trajectory.hold_paths). Only run-ups toward the jump are searched:
    running away to come back gains nothing a run-up keeps. A wait ends at
    rest; none is searched while Mario is moving.

    ``wanted(option, landing index, killed)``: the jumps that do what is
    needed. Returns [(room key, travel, wait, option, screened distances)]:
    the most room up to ENOUGH_ROOM first, then the lowest jump, then the
    shortest travel and wait. For a run-up the option is the jump's hold at
    that speed (certified from where the run-up really ends: _certify).
    """
    table = _jump_table(env, state, target, memo, (direction,), rise=rise, described_only=False)
    m = env.mario
    support = m.get("_platform")
    scene = state.scene or scene_from_labels(env.scene_labels())
    if support is None or scene.mario.box is None:
        return []
    spatial = state.execution
    known = spatial is not None and getattr(spatial, "motion_ready", False)
    motion = copy(spatial.motion if known else env.motion)
    if abs(motion.x_speed) / 16 > MOVING:
        waits, run_waits = (), ()
    start = _body(env)
    box = scene.mario.box
    rects = [p["rect"] for p in env.platforms]
    # What a moved jump path must not pass through (moving platforms move).
    solid = [p["rect"] for p in env.platforms if not p.get("moving")]
    lips = _pit_lips(env)
    horizon = 400
    enemies = _enemy_paths(env, horizon)
    floor = support["rect"]
    found = []

    def add(option, path, frames, travel, wait, walked, blocked=()):
        if not len(path) or (rise is not None and _rise(path) < rise):
            return
        screened = _screen(env, path, 0, frames, enemies, lips, rects, blocked)
        if screened is None:
            return
        distances, landing, killed, (dx, _) = screened
        if option.result is None:
            # A run-up's jump, not yet tried: its outcome as screened (where
            # it lands, what it kills, and which side of every other enemy
            # Mario ends on, that enemy moved on to then), as _measured has it.
            end = _path_landing(path, rects)
            t = frames + (end[0] if end else len(path) - 1)
            middle = (path[0][0] + path[0][2]) / 2 + dx
            sides = []
            for i, e in enumerate(env.enemies):
                if e["dead"] or i in killed:
                    continue
                boxes = enemies.get(i)
                b = boxes[min(t, len(boxes) - 1)] if boxes else (e["x"], 0, e["x"] + e["w"], 0)
                sides.append(middle < (b[0] + b[2]) / 2)
            option.result = Result(outcome=(False, landing, frozenset(killed), None, tuple(sides)))
        if not wanted(option, landing, killed):
            return
        total = _combine(walked, distances)
        found.append((_room_key(total), -_rise(path), -travel, -wait, travel, wait, option, total))

    # From here: each jump's trial path (it may hit things on its way), when
    # its destination names it.
    for option in table:
        if option.described and option.result.path:
            add(option, option.result.path, 0, 0, 0, {})
    # After a run-up that keeps its speed.
    for travel in sorted({t for t in travels if t >= MIN_RUN}):
        shift = travel * direction
        if not (floor.left <= start[0] + shift and start[2] + shift <= floor.right):
            continue
        frames, arrival = _carry_run(motion, shift)
        if frames >= horizon:
            continue
        run = _run_path(start, shift, frames)
        takeoff = (start[0] + shift, start[1], start[2] + shift, start[3])
        moved_box = (box[0] + shift, box[1], box[2] + shift, box[3])
        paths = []
        for hold, predicted in enumerate(hold_paths(scene, arrival, moved_box, direction), 1):
            path = [takeoff, *[tuple(takeoff[k] + b[k] - moved_box[k] for k in range(4)) for b in predicted]]
            end = _path_landing(path, rects)
            if end is not None and abs(end[1]) > REACH:
                # No destination the teacher commands (within REACH) makes a
                # jump landing this far (_same_jump): a run-up into it could
                # not be certified, and used up the tries.
                continue
            paths.append((hold, path))
        for wait in (0, *run_waits):
            if wait + frames >= horizon:
                continue
            walked = _screen_walk([start] * wait + run, enemies, lips)
            if walked is None:
                continue
            for hold, path in paths:
                option = JumpOption(None, None, [], hold, direction, path, False)
                add(option, path, wait + frames, travel, wait, walked, solid)
    # After a wait in place: the table's jumps, started later.
    for wait in waits:
        if wait >= horizon:
            continue
        walked = _screen_walk(_run_path(start, 0, wait), enemies, lips)
        if walked is None:
            continue
        for option in table:
            if option.described and option.result.path:
                add(option, option.result.path, wait, 0, wait, walked)
    found.sort(key=lambda f: f[:4], reverse=True)
    _middle_of_equal_runs(found)
    # Last, a step that stops (a plain run, toward the jump or away from it),
    # then a jump from rest: for when no running jump works, as onto a tall
    # pipe Mario stands too close to for a run-up to rise in time, or from
    # under an overhead ledge he must first step out from (the jumps' free
    # paths moved along the floor, as from rest). Toward the jump first. A
    # step of any length: the jump after it is remembered (_commit), so it does
    # not creep (on a ledge 33 pixels wide, only a 3-pixel step to its end
    # took Mario onto the platform above).
    backs = []
    for travel in sorted({t for t in travels if t}, key=lambda t: (t < 0, abs(t))):
        shift = travel * direction
        if not (floor.left <= start[0] + shift and start[2] + shift <= floor.right):
            continue
        frames = _run_frames(env.motion, shift)
        walked = _screen_walk(_run_path(start, shift, frames), enemies, lips)
        if walked is None or frames >= horizon:
            continue
        for option in table:
            if not option.free_path:
                continue
            screened = _screen(env, option.free_path, shift, frames, enemies, lips, rects, solid)
            if screened is None:
                continue
            distances, landing, killed, (dx, dy) = screened
            if not wanted(option, landing, killed) or (
                not killed and not _named(option.members, dx, dy)
            ):
                continue
            total = _combine(walked, distances)
            backs.append(
                (_room_key(total), -_rise(option.free_path), -(travel < 0), -abs(travel), travel, 0, option, total)
            )
    backs.sort(key=lambda f: f[:4], reverse=True)
    return [(f[0], f[4], f[5], f[6], f[7]) for f in found + backs]


def _middle_of_equal_runs(found):
    """Where run-ups of several lengths keep the same, most room, put first
    the one in the middle of the unbroken range of them (beginning at the
    shortest), as equal landing destinations are: the shortest alone taught
    a run back of exactly MIN_RUN in most scenes where the room stopped
    growing after it. ``found``: _takeoff_search's entries, sorted."""
    if not found or not found[0][4] or found[0][5]:
        return
    best, start = found[0][:2], found[0][4]
    equal = {f[4] for f in found if f[:2] == best and not f[5]}
    side = 1 if start > 0 else -1
    end = start
    while end + side in equal:
        end += side
    middle = start + (end - start) // 2
    first = next(i for i, f in enumerate(found) if f[:2] == best and f[4] == middle and not f[5])
    found.insert(0, found.pop(first))


def _screen_walk(path, enemies, lips):
    """Distances along a walk (no landing), or None when it meets an enemy."""
    distances = {}
    for t, (x0, y0, x1, y1) in enumerate(path):
        for lx, ly in lips:
            d = math.hypot(max(lx - x1, x0 - lx, 0), max(ly - y1, y0 - ly, 0))
            distances[("pit", lx, ly)] = min(distances.get(("pit", lx, ly), 999.0), d)
        for i, boxes in enemies.items():
            ex0, ey0, ex1, ey1 = boxes[min(t, len(boxes) - 1)]
            dx, dy = max(ex0 - x1, x0 - ex1, 0), max(ey0 - y1, y0 - ey1, 0)
            if dx == 0 and dy == 0:
                return None
            d = math.hypot(dx, dy)
            distances[("enemy", i)] = min(distances.get(("enemy", i), 999.0), d)
    return distances


def _run_options(env, state, target, memo):
    """Runs to every pixel Mario can walk to within REACH pixels, screened by
    where they end (in the goal first, then the most progress toward the
    target per frame, then the most room), the best CERTIFY tried:
    [(command, result)]."""
    if not env.mario["on_ground"]:
        return _air_options(env, state, target, memo)
    out = []
    for goal in _screened_runs(env, state, target)[:CERTIFY]:
        if goal not in memo:
            memo[goal] = _trial(env, state, goal, target)
        out.append((goal, memo[goal]))
    return out


def _screened_runs(env, state, target):
    """Every run _run_options screens (Mario standing), best first, untried:
    [command]."""
    m = env.mario
    support = m.get("_platform")
    if support is None:
        return []
    start = _body(env)
    cx, feet, half = (start[0] + start[2]) / 2, start[3], m["w"] / 2
    floor = support["rect"]
    lips = _pit_lips(env)
    enemies = _enemy_paths(env, 300)
    forbidden = set(tactic_schedule.current(env).get("forbidden", ()))
    before = distance(env, target)
    screened = []
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        # The floor he stands on, and lower floors within 64 pixels of its
        # ends that he can walk off onto.
        if i in forbidden or p.get("moving"):
            continue
        if r is not floor and (
            r.top <= floor.top + 2 or r.left >= floor.right + 64 or r.right <= floor.left - 64
        ):
            continue
        for x in range(
            round(max(r.left + half, cx - REACH)), round(min(r.right - half, cx + REACH)) + 1
        ):
            dx = round(x - cx)
            if abs(dx) < MIN_RUN:
                continue
            on_floor = r is floor
            if not on_floor and floor.left - half - DROP_MARGIN < x < floor.right + half + DROP_MARGIN:
                # Under the floor he stands on: a lower floor is walked onto
                # off its ends only, DROP_MARGIN past them. (Aimed under a
                # raised block, every run stopped at once; with hops never
                # taught, nothing was left.)
                continue
            if on_floor:
                frames = _run_frames(env.motion, dx)
                walk = _screen_walk(_run_path(start, dx, frames), enemies, lips)
                if walk is None:
                    continue
            else:
                frames = abs(dx)  # an estimate: the fall is screened by its trial
                walk = {}
            body = (x - half, r.top - m["h"], x + half, r.top)
            goal_in = env.goal is not None and env.goal.colliderect(
                type(env.goal)(body[0], body[1], body[2] - body[0], body[3] - body[1])
            )
            progress = before - _distance_at(env, target, x, r.top)
            room = min(
                [d for d in walk.values()] + [x - half - r.left, r.right - x - half], default=999.0
            )
            screened.append(
                (
                    (goal_in, progress / max(1, frames), min(64, room) // 2),
                    SkillToken("run", dx, round(r.top - feet)),
                )
            )
    screened.sort(key=lambda s: s[0], reverse=True)
    return [goal for _, goal in screened]


def _air_options(env, state, target, memo):
    """In the air, steering to every pixel within 32 of where Mario is, at the
    height of the floor below that point, each tried: [(command, result)]."""
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    out = []
    for dx in range(-32, 33):
        x = cx + dx
        below = [
            p["rect"].top
            for p in env.platforms
            if p["rect"].left <= x <= p["rect"].right and p["rect"].top >= feet - 2
        ]
        if not below:
            continue
        goal = SkillToken("run", dx, max(SKILL_Y[0], min(SKILL_Y[-1], round(min(below) - feet))))
        if goal not in memo:
            memo[goal] = _trial(env, state, goal, target)
        out.append((goal, memo[goal]))
    return out


def _distance_at(env, target, x, feet):
    """distance() for Mario's feet at (x, feet)."""
    m = env.mario
    saved = m["x"], m["y"]
    m["x"], m["y"] = x - m["w"] / 2, feet - m["h"]
    try:
        return distance(env, target)
    finally:
        m["x"], m["y"] = saved


def destination(env, state, proposal):
    if tactic_schedule.current(env)["kind"] == "monster" and env.mario["on_ground"]:
        return monster_destination(env, state)
    if tactic_schedule.current(env)["kind"] == "plant":
        plant = plant_destination(env, state)
        if plant is not None:
            return plant
    target = teacher_target(env)
    airborne = not env.mario["on_ground"]
    committed = _committed(env, state, target)
    if committed is not None:
        return committed
    waited = _planned_wait(env, state)
    if waited is not None:
        return waited
    if target != training_target(env):
        proposal = None
    if (
        proposal is not None
        and proposal.mode == "run"
        and not airborne
        and abs(proposal.x) < 8
        and abs(proposal.y) < 8
    ):
        # A creep: the button route's run-up before a jump, which a run that
        # brakes on arrival never completes (it would be asked again, a few
        # pixels shorter each time). The candidates are ranked instead.
        proposal = None
    mount = mount_destination(env, state, target)
    if mount is not None:
        return mount
    seg = tactic_schedule.current(env)
    if (
        _waiting(env, state)
        and seg["kind"] == "plain"
        and seg.get("area") is not None
        and env.mario["on_ground"]
    ):
        # Holding an area: hold, or jump over a walker coming through it;
        # nothing else (no route's moves, no progress).
        return _dodge_in_area(env, state, target) or SkillToken("hold", 0, 0)
    # Under speed run a stomp is the last resort: any other move that wins
    # or advances is taught first (the strategy picks stomp or pass;
    # _choose_move). Under max points the ranking alone decides.
    speed_run = getattr(state, "strategy", "speed_run") != "max_points"
    # The moves found that fail when started a little later (_tolerant), in
    # the order found, each with the run-up plan it committed (or None).
    fragile = []

    def taught(command):
        """``command`` when it tolerates a later start; else None (the next
        move found is tried), the move and its plan kept in ``fragile``."""
        if _tolerant(env, state, target, command):
            return command
        notes = getattr(state, "notes", {})
        fragile.append((command, notes.pop("planned_jump", None)))
        return None

    deferred = None
    if proposal is not None:
        result = trial(env, state, proposal, target)
        if (
            _nameable(env, state, proposal, result)
            and not (speed_run and _move_kind(env, result) == "stomp")
            and not (proposal.mode == "jump" and _mere_hop(env, result))
        ):
            if result.won:
                command = taught(_with_margin(env, state, target, proposal, result))
                if command is not None:
                    return command
            elif (
                result.safe
                and result.clearance >= 8
                and closes_on_enemy(env, target, proposal, result)
                and (proposal.mode != "hold" or _waiting(env, state))
                and not _overshoots_goal(env, result)
            ):
                # A proposal that only progresses is taught when no move
                # from here wins (below): the route's waypoint 16 pixels on
                # was taught before a run that reached the goal (user).
                deferred = (proposal, result)
    # Rank every jump Mario can make and the screened runs (_jump_table,
    # _run_options: every destination considered) on label-only trials;
    # confirm in order through the full trial (which replays jumps through
    # vision when it is available). The trials are kept for the margin search.
    winners, progressing, memo = [], [], {}
    options = [(o.command, o.result) for o in _jump_table(env, state, target, memo)]
    options += _run_options(env, state, target, memo)
    for goal, result in options:
        if not _nameable(env, state, goal, result):
            continue
        # The proposal stays a candidate: it may have been passed over only
        # for being a stomp under speed run or a mere hop, ranked last here.
        last = speed_run and _move_kind(env, result) == "stomp"
        # A move away from the target's side goes against the tactic: it
        # comes after every move toward it, stomps included (a jump back off
        # a ledge reached the route's floor first, under the ledge, where a
        # walker then trapped Mario).
        back = goal.mode != "hold" and goal.x * target.direction < 0
        if result.won:
            winners.append(((back, last), _rank(goal, result), goal))
        elif (
            result.safe
            and result.clearance >= 8
            and closes_on_enemy(env, target, goal, result)
            and (result.advanced or result.progress > 0 or airborne)
            and not _overshoots_goal(env, result)
        ):
            value = (100 if result.advanced else result.progress) / max(1, result.frames)
            if goal.mode == "jump" and _mere_hop(env, result):
                # A hop that clears nothing is never taught (user,
                # 2026-10-13): a run makes the same progress.
                continue
            progressing.append(((back, last, False), (value, result.margin), goal))
    winners = [((back, last, False), rank, goal) for (back, last), rank, goal in winners]
    if deferred is not None and not winners:
        command = taught(_with_margin(env, state, target, *deferred))
        if command is not None:
            return command
    ordered = [
        (kind, goal)
        for backward in (False, True)
        for stomps in (False, True)
        for kind, entries, hops in (
            ("win", winners, False),
            ("advance", progressing, False),
            ("advance", progressing, True),
        )
        for _, _, goal in sorted(
            (e for e in entries if e[0] == (backward, stomps, hops)),
            key=lambda e: e[1],
            reverse=True,
        )
    ]
    for kind, goal in ordered:
        if len(fragile) > FRAGILE_TRIES:
            break
        result = trial(env, state, goal, target)
        if not _nameable(env, state, goal, result):
            continue
        if (kind == "win" and result.won) or (
            kind == "advance"
            and result.safe
            and result.clearance >= 8
            and closes_on_enemy(env, target, goal, result)
            and (result.advanced or result.progress > 0 or airborne)
            and not _overshoots_goal(env, result)
        ):
            command = taught(_with_margin(env, state, target, goal, result, memo))
            if command is not None:
                return command
    if fragile:
        # No move found that tolerates a later start: the first one found.
        command, plan = fragile[0]
        if plan is not None:
            state.notes["planned_jump"] = plan
        return command
    run_up = _run_up_fallback(env, state, target, memo, speed_run)
    if run_up is not None:
        return run_up
    descent = descent_destination(env, state, target)
    if descent is not None:
        return descent
    hold = SkillToken("hold", 0, 0)
    # Holding helps only while something moves: with only standing enemies
    # it changes nothing, and was taught until the episode stalled.
    if any(p.get("moving") for p in env.platforms) or any(
        not e["dead"] and (e.get("speed", 0) or e.get("kind") in ("piranha_plant", "monster"))
        for e in env.enemies
    ):
        if trial(env, state, hold, target).safe:
            return hold
    return None


def _overshoots_goal(env, result) -> bool:
    """Whether a move ends Mario past the goal box, on its floor, from short
    of it, in the layout's last tactic with its points collected: no progress
    toward the goal. (Passed over for being fragile, a jump into a goal a
    walker was heading for gave way to a longer one past it, against the
    world's end, where the walker trapped Mario.)"""
    goal = env.goal
    if (
        goal is None
        or env._goal_on_stomp
        or env._tactic_index != len(env._tactics) - 1
        or env.points() < env._strategy_objective.get("points", 0)
    ):
        return False
    m = env.mario
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    if goal.left <= cx <= goal.right or abs(feet + result.end_dy - goal.bottom) > 2:
        return False
    side = 1 if goal.centerx > cx else -1
    far = goal.right if side > 0 else goal.left
    return (cx + result.end_dx - far) * side > 0


def _nameable(env, state, goal, result) -> bool:
    """Whether a command names an action (teacher_actions.teacher_action): a
    verb aimed at an object the scene reports. Only such a command is taught;
    one aimed at a point in the air, say, gives the skill nothing to learn."""
    from .teacher_actions import teacher_action

    return teacher_action(env, state, goal, result) is not None


def _exact_key(result) -> tuple:
    """The threats' exact distances from the closest up (at most FAR pixels):
    the tie-break after _distance_key, so that among commands with equal room
    in 2-pixel steps the one with the most room wins, not the shortest."""
    values = sorted(min(FAR, d) for d in result.distances.values())
    return tuple(values[:12]) + (float(FAR),) * max(0, 12 - len(values))


def _rank(goal, result):
    """Larger is better among versions of one move: enough room from every
    threat (_room_key: up to ENOUGH_ROOM, the closest first), then the
    move that ends nearest the goal box's middle (of moves that win; in
    4-pixel steps), then the destination nearest where it lands (for a stomp,
    the enemy's middle), then the lowest jump, then the most room, then the
    quickest."""
    return (
        _room_key(result.distances) if result.distances else _distance_key(result),
        -(getattr(result, "goal_offset", 0.0) // 4),
        -getattr(result, "aim_error", 0.0),
        -_rise(result.path),
        _exact_key(result),
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


def _mere_hop(env, result) -> bool:
    """A jump that lands back where Mario stands, passing and stomping
    nothing: only a little progress, which a run makes as well. (The button
    route proposes such hops toward an enemy, and per frame a short hop
    progresses faster than a short run, which brakes.)"""
    return bool(result.outcome) and _is_hop(env, result) and _move_kind(env, result) == "other"


def _versions(env, state, target, goal, result, memo=None):
    """Every jump that makes the same move as ``goal`` (_works: the same
    landing platform, enemies killed, route mark and sides), of every jump
    Mario can make from here (_jump_table), with its trial."""
    memo = {} if memo is None else memo
    hop = _is_hop(env, result)
    direction = (goal.x > 0) - (goal.x < 0)
    found = [(goal, result)]
    support, killed = (result.outcome[1], result.outcome[2]) if result.outcome else (None, None)
    lands_on = {support} if support is not None and not killed else None
    for option in _jump_table(env, state, target, memo, (direction,), lands_on=lands_on):
        if option.command != goal and _works(option.result, result, hop):
            found.append((option.command, option.result))
    for g, r in found:
        # Aim where it lands (for a stomp, on the enemy where it has walked to).
        r.aim_error = abs(g.x - r.end_dx)
    return found


def _with_margin(env, state, target, goal, result, memo=None):
    """The chosen move's version with the most margin (training labels only).

    First the move itself is chosen where an enemy is near ahead
    (_choose_move), and the takeoff before a pit ahead (_pit_takeoff). A jump
    is then replaced by the jump that does the same thing while keeping the
    most room from every threat (_versions: every jump Mario can make from
    here). On the ground, a jump that passes within 48 pixels of an enemy is
    taught after a run toward it when one gives the same jump at least 4 more
    pixels of room (_better_takeoff).
    """
    memo = {} if memo is None else memo
    goal, result = _choose_move(env, state, target, goal, result, memo)
    goal, result = _pit_takeoff(env, state, target, goal, result, memo)
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
        if _works(confirmed, result, hop) and _nameable(env, state, g, confirmed):
            chosen = (g, confirmed)
            break
    if chosen is None:
        return goal
    if takeoff:
        run = _better_takeoff(env, state, target, *chosen, memo)
        if run is not None:
            return run
    return chosen[0]


def _pit_ahead(env, direction):
    """The x of the pit edge the floor Mario stands on ends at, within 96
    pixels ahead (``direction``: 1 right, -1 left), or None."""
    m = env.mario
    support = m.get("_platform") if m["on_ground"] else None
    if support is None or support.get("moving"):
        return None
    r = support["rect"]
    x = r.right if direction > 0 else r.left
    if not 0 <= (x - (m["x"] + m["w"] / 2)) * direction <= 96:
        return None
    return x if (x, r.top) in _pit_lips(env) else None


def _works_now(r):
    return r.won or (r.safe and r.clearance >= 8 and (r.advanced or r.progress > 0))


# A taught move must still work when it starts this many frames later, Mario
# coasting meanwhile (_tolerant): the skill decides at a slightly different
# moment and place, after its own slightly different moves, and a move that
# works only from the exact moment the teacher chose is a choice it cannot
# reliably repeat (a hop onto a strip of floor before a walker; a creep
# before a jump). After FRAGILE_TRIES such moves, the first one is taught.
TOLERANCE_FRAMES = (2, 4)
FRAGILE_TRIES = 3


def _tolerant(env, state, target, command) -> bool:
    """Whether ``command`` still works started each of TOLERANCE_FRAMES frames
    later, aimed at the same point of the world, Mario pressing nothing
    meanwhile: it wins, or is safe with 8 pixels of room and advances (in
    the air, safe is enough, as for the move itself). A hold always does."""
    if command.mode == "hold":
        return True
    airborne = not env.mario["on_ground"]
    m = env.mario
    aim_x, aim_feet = m["x"] + command.x, m["y"] + m["h"] + command.y
    for delay in TOLERANCE_FRAMES:
        with probe_state(env):
            spatial = deepcopy(state.execution) if state.execution is not None else SpatialFeedback()
            scene = state.scene or scene_from_labels(env.scene_labels())
            if state.execution is None:
                spatial.motion = deepcopy(env.motion)
                spatial.observe(scene)
            for _ in range(delay):
                spatial.executed(0, scene)
                _, _, done, truncated, _ = env.step(0)
                if done or truncated:
                    break
                scene = scene_from_labels(env.scene_labels())
                spatial.observe(scene)
            if done or truncated:
                if not env._goal_credited:
                    return False
                continue
            later = copy(state)
            later.execution, later.controller, later.scene = spatial, None, scene
            m = env.mario
            moved = SkillToken(
                command.mode,
                round(aim_x - m["x"]),
                round(aim_feet - (m["y"] + m["h"])),
                carry=command.carry,
            )
            r = _trial(env, later, moved, target)
            if not (
                r.won
                or (r.safe and r.clearance >= 8 and (r.advanced or r.progress > 0 or airborne))
            ):
                return False
    return True


def _crossing_wanted(env, target, lip, direction):
    """Whether a screened jump ends past the pit edge at ``lip``: on the
    target's platform when the target is one."""
    rects = [p["rect"] for p in env.platforms]

    def wanted(option, landing, killed):
        if landing is None:
            return False
        if target.platform_index is not None:
            return landing == target.platform_index
        return (rects[landing].centerx - lip) * direction > 0

    return wanted


def _certify(env, state, target, direction, travel, wait, option, check):
    """Certify a screened plan by trials: from here, a run-up of ``travel``
    pixels toward ``direction`` that keeps its speed (or a hold, for a wait),
    then ``option``'s
    jump from where that leaves Mario (_same_jump), as the screen assumed. ``check(jump result)`` must hold for the jump.
    Returns (first command, its result, the jump's result, the plan) or None;
    the plan, after a run or a wait: (the jump, Mario's x where it starts,
    what it is aimed at from there: teacher_actions.jump_aim)."""
    from .teacher_actions import jump_aim

    if not travel and not wait:
        confirmed = trial(env, state, option.command, target)
        if (
            not _works_now(confirmed)
            or not check(confirmed)
            or (not confirmed.won and _overshoots_goal(env, confirmed))
        ):
            return None
        return option.command, confirmed, confirmed, None
    # A run-up keeps its speed into the jump (SkillToken.carry).
    # A run-up's jump is screened from where it ends at its speed (its option
    # has no command yet); a step that stops is followed by a table jump.
    first = (
        SkillToken("run", travel * direction, 0, carry=travel > 0 and option.command is None)
        if travel
        else SkillToken("hold", 0, 0)
    )
    moved = _trial(env, state, first, target, save=True)
    if not moved.safe or moved.snapshot is None:
        return None
    with probe_state(env):
        restore_env_state(env, moved.snapshot)
        following = copy(state)
        following.execution, following.controller = moved.spatial, moved.controller
        following.scene = moved.spatial.previous
        found = _same_jump(env, following, option)
        if found is None:
            return None
        jump, members = found
        landed = trial(env, following, jump, target)
        if not _describes(env, jump, landed):
            # Taught as the destination of this jump nearest where it really
            # lands, if one names it; else the plan cannot be taught.
            aimed = _aim(env, landed, members)
            again = trial(env, following, aimed, target) if aimed != jump else landed
            if again.outcome != landed.outcome or not _describes(env, aimed, again):
                return None
            jump, landed = aimed, again
        stop = env.mario["x"]
        aim = jump_aim(env, landed)
        over = not landed.won and _overshoots_goal(env, landed)
    if not _works_now(landed) or not check(landed) or over:
        return None
    combined = Result(**{**vars(moved), "distances": _combine(moved.distances, landed.distances)})
    return first, combined, landed, (jump, stop, aim)


def _pit_takeoff(env, state, target, goal, result, memo):
    """Where to take off over a pit ahead (training labels only): jump now, or
    first run toward it and stop anywhere up to its edge (every pixel within
    48 pixels of it), followed by any jump Mario can make from there, whichever
    keeps the most room from every threat before, during and after the jump
    (_takeoff_search; equal room prefers the shorter run). Returns (command,
    its result).

    It applies on the ground when the floor Mario stands on ends at a pit
    within 96 pixels ahead and the target lies beyond it, except in lessons
    that are one given jump.
    """
    if (
        not env.mario["on_ground"]
        or getattr(env, "_action_jump_direction", 0)
        or goal.mode == "hold"
        or not result.outcome
    ):
        return goal, result
    best = _pit_run_up(env, state, target, memo)
    if best is not None and _clear_of_enemies(best[1].distances):
        return best
    # Nothing crosses now keeping ENOUGH_ROOM from every enemy (a walker
    # pacing the landing up to the pit's edge): wait where Mario stands, with
    # room to run up, until a running jump across will (user, 2026-10-14),
    # rather than walk to the edge and jump from rest onto its lip beside
    # the walker.
    wait = _pit_wait(env, state, target, memo)
    if wait is not None:
        state.notes.pop("planned_jump", None)  # the run-up _pit_run_up planned
        return wait
    if best is not None:
        return best
    return goal, result


def _clear_of_enemies(distances) -> bool:
    """Whether every enemy stays at least ENOUGH_ROOM pixels away (``distances``:
    Result.distances, or a screen's)."""
    return all(d >= ENOUGH_ROOM for key, d in distances.items() if key[0] == "enemy")


def _pit_inputs(env, target, state=None):
    """For the pit ahead with the target beyond it: (the run-up lengths
    searched, toward it and ending within 48 pixels of its edge, and which
    landings cross it); or None. Unless the lesson is played for points, a
    crossing stomps nothing: Mario lands with room and then clears the walker
    (user, 2026-10-14; the screen took a jump onto the walker for one clear
    of it, which no trial certified)."""
    direction = target.direction
    lip = _pit_ahead(env, direction)
    if lip is None or (target.center - lip) * direction <= 0:
        return None
    m = env.mario
    front = (lip - (m["x"] + m["w"] if direction > 0 else m["x"])) * direction
    travels = [0] + [t for t in range(1, math.floor(front) + 1) if front - t <= 48]
    crosses = _crossing_wanted(env, target, lip, direction)
    if getattr(state, "strategy", "speed_run") == "max_points":
        return travels, crosses
    return travels, lambda option, landing, killed: not killed and crosses(option, landing, killed)


def _pit_run_up(env, state, target, memo):
    """The takeoff over the pit ahead (_pit_takeoff's search): jumping now, or
    after a run-up toward it ending within 48 pixels of its edge, certified;
    (command, its result) or None (no pit ahead, or nothing certifies)."""
    inputs = _pit_inputs(env, target, state)
    if inputs is None:
        return None
    travels, wanted = inputs

    def check(r):
        index = r.outcome[1] if r.outcome else None
        killed = r.outcome[2] if r.outcome else ()
        return r.won or (index is not None and wanted(None, index, killed))

    return _best_certified(env, state, target, memo, target.direction, travels, wanted, check)


# A wait before a run-up over a pit (_pit_wait) is screened every WAIT_STEP
# frames up to MAX_WAIT; at most MAX_PLANNED_WAITS are planned an episode (the
# screen can keep promising a run-up the certified search then refuses, at
# about 100 s a search).
WAIT_STEP = 8
MAX_WAIT = 240
MAX_PLANNED_WAITS = 4


def _pit_wait(env, state, target, memo):
    """When no takeoff over the pit ahead certifies now keeping ENOUGH_ROOM
    from every enemy: the shortest wait in place after which a run-up and
    jump across is screened to keep it, as (hold, its trial); or None. The
    wait is remembered (notes["planned_wait"]): the holds until it ends are
    taught without searching again (_planned_wait; searched every frame, a
    wait took minutes of teacher time), and the takeoff is then certified as
    usual."""
    inputs = _pit_inputs(env, target, state)
    notes = getattr(state, "notes", None)
    if inputs is None or notes is None or notes.get("waits_planned", 0) >= MAX_PLANNED_WAITS:
        return None
    travels, wanted = inputs
    if not any(
        not e["dead"] and (e.get("speed", 0) or e.get("kind") in ("piranha_plant", "monster"))
        for e in env.enemies
    ):
        # Nothing moves: waiting changes nothing (a standing walker 14 pixels
        # from the certified landing, against 16, was waited for until
        # MAX_PLANNED_WAITS ran out).
        return None
    waits = tuple(range(WAIT_STEP, MAX_WAIT + 1, WAIT_STEP))
    found = _takeoff_search(
        env, state, target, memo, target.direction, travels, wanted, run_waits=waits
    )
    clear = [f[2] for f in found if f[1] and f[2] and _clear_of_enemies(f[4])]
    if not clear:
        return None
    wait = min(clear)
    hold = SkillToken("hold", 0, 0)
    result = trial(env, state, hold, target)
    if not result.safe:
        return None
    notes["planned_wait"] = (env.steps + wait, env.mario["x"])
    notes["waits_planned"] = notes.get("waits_planned", 0) + 1
    return hold, result


def _planned_wait(env, state):
    """The hold of a wait _pit_wait planned, while it lasts: Mario still
    standing where it began, and holding meets no enemy until it ends; else
    None (and the plan is dropped)."""
    notes = getattr(state, "notes", None)
    plan = notes.get("planned_wait") if notes is not None else None
    if plan is None:
        return None
    until, x = plan
    m, left = env.mario, until - env.steps
    if (
        left <= 0
        or not m["on_ground"]
        or abs(m["x"] - x) > 1
        or _screen_walk(_run_path(_body(env), 0, left), _enemy_paths(env, left), _pit_lips(env))
        is None
    ):
        notes.pop("planned_wait", None)
        return None
    return SkillToken("hold", 0, 0)


def _best_certified(
    env, state, target, memo, direction, travels, wanted, check, waits=(), rise=None
):
    """The best screened plan (_takeoff_search) that trials certify, as (first
    command, its result): jumping now when it works, unless running or waiting
    first gives at least GAIN more pixels of room, counting room up to
    ENOUGH_ROOM (so a takeoff is not moved a little at a time, nor for room
    already enough). None when nothing certifies."""
    now, later = None, None
    # Tries per kind of plan: jumping now, running up (or waiting) first, and
    # a step that stops first (searched last, for when no running jump works:
    # its tries are not spent on running jumps the trials refuse).
    tried = {"now": 0, "later": 0, "step": 0}
    for _, travel, wait, option, _ in _takeoff_search(
        env, state, target, memo, direction, travels, wanted, waits=waits, rise=rise
    ):
        if now is not None and later is not None:
            break
        kind = (
            "now"
            if not travel and not wait
            else "step" if travel and option.command is not None else "later"
        )
        # A step is the last resort, and its jump from rest, onto a pipe as
        # tall as a jump rises, is screened less exactly: more of them are
        # tried.
        if tried[kind] >= {"now": CERTIFY, "later": 2 * CERTIFY, "step": 4 * CERTIFY}[kind]:
            continue
        if kind == "step" and abs(travel) < MIN_RUN and now is not None:
            # A step this short only when no jump from here works: for a few
            # pixels more room it was taught before jumps that worked as they
            # were (pipe_mount, skill_upper_gap_exit).
            continue
        if kind == "now":
            if now is None:
                tried[kind] += 1
                now = _certify(env, state, target, direction, 0, 0, option, check)
        elif later is None:
            tried[kind] += 1
            later = _certify(env, state, target, direction, travel, wait, option, check)
    def enough(certified):
        r = certified[1]
        values = [_enough(k, d) for k, d in r.distances.items()] + [_enough(None, r.enemy_gap)]
        if r.edge_margin is not None:
            values.append(_enough(("landing",), r.edge_margin))
        return min(values)

    if now is not None and (later is None or enough(later) < enough(now) + GAIN):
        return now[0], now[1]
    if later is not None:
        _commit(state, later)
        return later[0], later[1]
    return None


def _same_jump(env, state, option):
    """The command for ``option``'s jump (the same hold and steering) from
    where Mario stands now, and every destination that makes it: the screen
    moved its path along the floor, so it lands where that moved path does,
    and is commanded as the destination of this jump nearest that predicted
    landing. None when no destination makes this jump from here."""
    for hold, members, landing, _, _ in _jump_groups(env, state, {}, option.direction):
        if hold == option.hold:
            first = (
                _nearest(members, landing[1], landing[2]) if landing else members[len(members) // 2]
            )
            return first, members
    return None


def _commit(state, certified):
    """Remember the jump that follows a taught run-up or wait (certified by
    _certify), to take it at the next decision (_committed)."""
    first, _, _, plan = certified
    if plan is not None:
        state.notes["planned_jump"] = (first, *plan)


def _committed(env, state, target):
    """The jump remembered after the run-up or wait just executed (_commit),
    moved by how far Mario actually went, if it still works; else None."""
    planned = state.notes.pop("planned_jump", None)
    if planned is None or not env.mario["on_ground"]:
        return None
    first, jump, stop, _ = planned
    spatial = getattr(state, "execution", None)
    if spatial is None or spatial.destination != first:
        return None  # what was executed was not the planned run-up
    jump = _shifted(jump, env.mario["x"] - stop)
    result = trial(env, state, jump, target)
    return jump if _works_now(result) else None


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
    over waiting, unless the waiting command lets a jump over pass with at
    least 4 more pixels of margin; and over a stomp. Where no jump over works
    now, a wait after which one works is taught over a stomp. Every jump Mario
    can make is tried (_jump_table), so scenes that look alike get the same
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
            if _waiting_helps(env, state, target, goal, confirmed, memo):
                return goal, result
        return g, confirmed
    # No jump from here does it: a run-up toward the enemy that keeps its
    # speed into one that does (not a run that stops short, or a hop).
    run_up = _run_up_to_enemy(env, state, target, wanted, memo)
    if run_up is not None:
        return run_up
    if kind == "stomp" and not max_points:
        wait = _wait_to_pass(env, state, target, memo)
        if wait is not None:
            return wait
    return goal, result


def run_up_travels(reach):
    """The run-up lengths searched up to ``reach`` pixels: every 2 pixels from
    MIN_RUN to 64, every 4 beyond (a long run-up's takeoff matters less to a
    pixel)."""
    near = list(range(MIN_RUN, min(64, reach) + 1, 2))
    return near + list(range(68, reach + 1, 4))


def _run_up_to_enemy(env, state, target, kind, memo):
    """A run-up toward the nearest enemy ahead, keeping its speed into a jump
    that does ``kind`` to it ("pass" or "stomp"): the best certified
    (_best_certified; the jump is remembered for the next decision), as
    (command, its result); or None."""
    direction = target.direction
    m = env.mario
    cx = m["x"] + m["w"] / 2
    ahead = [
        (e["x"] + e["w"] / 2 - cx) * direction
        for e in env.enemies
        if not e["dead"] and e["h"] > 0 and (e["x"] + e["w"] / 2 - cx) * direction > 0
    ]
    if not ahead:
        return None
    reach = int(min(REACH, max(ahead)))

    def check(r):
        return r.won or _move_kind(env, r) == kind

    return _best_certified(
        env, state, target, memo, direction, run_up_travels(reach), _pass_wanted(env, kind), check
    )


def _run_up_fallback(env, state, target, memo, speed_run):
    """When no move from here advances: a run-up keeping its speed into a jump
    past the enemy ahead (or onto it, under max points; the other if that
    fails), or over the pit ahead; its first command, or None. (Hops toward
    them are not taught, and a run that stops short of them leaves the same
    jump to make from a standstill.)"""
    if not env.mario["on_ground"] or getattr(env, "_action_jump_direction", 0):
        return None
    if _enemy_ahead(env, target.direction):
        for kind in ("pass", "stomp") if speed_run else ("stomp", "pass"):
            found = _run_up_to_enemy(env, state, target, kind, memo)
            if found is not None:
                return found[0]
    found = _pit_run_up(env, state, target, memo)
    return found[0] if found is not None else None


def _jumps_that(env, state, target, kind, memo, direction=None):
    """Of every jump Mario can make toward ``direction`` (the target's by
    default), those that work and do ``kind`` to an enemy ("stomp" or
    "pass", _move_kind): [(command, result)]."""
    direction = target.direction if direction is None else direction
    return [
        (option.command, option.result)
        for option in _jump_table(env, state, target, memo, (direction,))
        if _works_now(option.result) and _move_kind(env, option.result) == kind
    ]


def _pass_wanted(env, kind="pass"):
    """Whether a screened jump passes (or stomps, for ``kind`` "stomp") an
    enemy: judged by the jump's own trial from here, and a stomp must kill."""

    def wanted(option, landing, killed):
        if landing is None:
            return False
        if kind == "stomp":
            return bool(killed)
        return not killed and _move_kind(env, option.result) == "pass"

    return wanted


def _wait_to_pass(env, state, target, memo):
    """A wait (holding still, or a run of any length up to 32 pixels toward
    the target) after which a jump over the enemy works: the one whose jump
    over keeps the most room, screened at every pixel and certified by trials
    (equal room: the shorter wait), as (command, its result); or None."""
    direction = target.direction
    held = _trial(env, state, SkillToken("hold", 0, 0), target)
    waits = (held.frames,) if held.safe else ()
    wanted = _pass_wanted(env)

    def check(r):
        return _move_kind(env, r) == "pass" or r.won

    for _, travel, wait, option, _ in _takeoff_search(
        env, state, target, memo, direction, list(range(1, 33)), wanted, waits=waits
    )[:CERTIFY]:
        certified = _certify(env, state, target, direction, travel, wait, option, check)
        if certified is not None and certified[1].enemy_gap >= 8:
            _commit(state, certified)
            return certified[0], certified[1]
    return None


def _waiting_helps(env, state, target, waiting, result, memo) -> bool:
    """Whether, after the ``waiting`` command (a run or a hold), a jump over
    the enemy passes with at least 4 more pixels of margin than ``result``
    (passing now): screened for every jump Mario can make, the best certified.
    When it does, that jump is remembered (_commit) and taken after the wait:
    weighed afresh at every decision, waiting could win again and again (a
    walker walking away was trailed by ever shorter runs and hops)."""
    direction = target.direction
    if waiting.mode == "run" and (waiting.x > 0) == (direction > 0) and waiting.x:
        travels, waits = [abs(waiting.x)], ()
    elif waiting.mode == "hold":
        held = _trial(env, state, waiting, target)
        travels, waits = [], ((held.frames,) if held.safe else ())
    else:
        return False

    def check(r):
        return _move_kind(env, r) == "pass" or r.won

    for _, travel, wait, option, _ in _takeoff_search(
        env, state, target, memo, direction, travels, _pass_wanted(env), waits=waits
    )[:CERTIFY]:
        certified = _certify(env, state, target, direction, travel, wait, option, check)
        if certified is not None:
            helps = certified[2].margin >= result.margin + 4
            if helps:
                _commit(state, certified)
            return helps
    return False


def _shifted(jump, shift):
    """The jump to the same place after Mario moved ``shift`` pixels (clamped
    to the command range)."""
    x = max(SKILL_X[0], min(SKILL_X[-1], round(jump.x - shift)))
    return SkillToken("jump", x, jump.y)


def _better_takeoff(env, state, target, jump, result, memo):
    """A run toward the jump (any length up to 24 pixels) after which a jump
    making the same move passes the enemy with at least 4 more pixels of
    margin: screened at every pixel and certified by trials; or None."""
    direction = 1 if jump.x > 0 else -1
    kind = _move_kind(env, result)

    def wanted(option, landing, killed):
        return landing is not None and not killed and _move_kind(env, option.result) == kind

    def check(r):
        return _move_kind(env, r) == kind or r.won

    for _, travel, wait, option, total in _takeoff_search(
        env, state, target, memo, direction, list(range(1, 25)), wanted
    )[:CERTIFY]:
        if min(total.values(), default=999.0) < result.margin + GAIN:
            break  # screened best-first: nothing further gains enough
        certified = _certify(env, state, target, direction, travel, wait, option, check)
        if certified is not None and certified[1].margin >= result.margin + GAIN:
            _commit(state, certified)
            return certified[0]
    return None


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
        # Of every jump Mario can make from there, one that finishes.
        for option in _jump_table(env, following, target, {}, (target.direction,)):
            if option.result.won and trial(env, following, option.command, target).won:
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
    """Certify the takeoff location as well as the landing destination, for a
    raised platform, or a lower one across a gap (walking off the edge does
    not reach it).

    Every jump Mario can make onto the platform, from where he stands or
    (except in lessons that are one given jump) after a run along his floor
    to any point (every pixel, forward or back: a
    takeoff under an overhead ledge must first move clear of it, and a jump
    down across a wide gap may need a run-up), is screened, and the one
    keeping the most room is certified and taught. (Without this, a jump that
    dropped below a lower target counted as progress toward it.)
    """
    m = env.mario
    if not m["on_ground"] or target.platform_index is None or target.kind != "mount":
        return None
    rect = env.platforms[target.platform_index]["rect"]
    feet, center, half = m["y"] + m["h"], m["x"] + m["w"] / 2, m["w"] / 2
    side = target.direction
    lower = rect.top > feet + 2
    if lower:
        support = m.get("_platform")
        if support is None:
            return None
        source = support["rect"]
        gap = rect.left - source.right if side > 0 else source.left - rect.right
        if gap <= 8:
            return None  # walking off the edge reaches it
    elif rect.top >= feet - 2 or rect.top < feet - 88:
        return None
    memo = {}

    def wanted(option, landing, killed):
        return landing == target.platform_index

    def check(r):
        return r.won or (
            r.safe and r.advanced and r.outcome and r.outcome[1] == target.platform_index
        )

    support = env.mario.get("_platform")
    travels = [0]
    # A lesson that is one given jump takes it from where Mario stands.
    if support is not None and not support.get("moving") and not getattr(env, "_action_jump_direction", 0):
        source = support["rect"]
        travels += [
            t
            for t in range(-48, REACH + 1)
            if t and source.left + half + 2 <= center + t * side <= source.right - half - 2
        ]
    # Only jumps rising above a raised platform's top can land on it.
    best = _best_certified(
        env,
        state,
        target,
        memo,
        side,
        travels,
        wanted,
        check,
        rise=None if lower else feet - rect.top + 2,
    )
    return best[0] if best is not None else None


# Holding an area, a walker that would reach Mario within this many frames
# is jumped over (_dodge_in_area); one closer than IMMINENT_FRAMES is jumped
# over even with less than ENOUGH_ROOM.
DODGE_FRAMES = 48
IMMINENT_FRAMES = 12


def _dodge_in_area(env, state, target):
    """Holding an area (a hold segment with an ``area``): when a walker would
    reach Mario if he held, the lowest jump that lets it pass under or behind
    him and lands back on his platform without leaving the area, at the
    moment its room from the walker is greatest: now, unless the same jumps
    started up to 20 frames later keep more (then hold), or the walker is
    about to reach him. Certified by a trial that it passes the walker
    safely. None: hold."""
    seg = tactic_schedule.current(env)
    area = seg.get("area")
    m = env.mario
    support = m.get("_platform")
    if area is None or not m["on_ground"] or support is None:
        return None
    start = _body(env)
    lips = _pit_lips(env)
    soon = _enemy_paths(env, DODGE_FRAMES)
    if _screen_walk(_run_path(start, 0, DODGE_FRAMES), soon, lips) is not None:
        return None  # holding meets no walker for a while
    imminent = _screen_walk(_run_path(start, 0, IMMINENT_FRAMES), soon, lips) is None
    anchor = env._tactic_anchor
    if env._tactic_anchor_platform is not None:
        anchor += env.platforms[env._tactic_anchor_platform]["rect"].x - env._tactic_anchor_platform_x
    index = next((i for i, p in enumerate(env.platforms) if p is support), None)
    rects = [p["rect"] for p in env.platforms]
    enemies = _enemy_paths(env, 400)
    middle = (start[0] + start[2]) / 2
    scene = state.scene or scene_from_labels(env.scene_labels())
    if scene.mario.box is None:
        return None
    box = scene.mario.box
    spatial = state.execution
    known = spatial is not None and getattr(spatial, "motion_ready", False)
    motion = copy(spatial.motion if known else env.motion)
    # Each jump's predicted path (the executor's own, every hold and way):
    # cheap enough to weigh every frame; a destination is found only for the
    # one taught (destination_for).
    groups, paths_of = [], {}
    for direction in (0, -1, 1):
        paths_of[direction] = hold_paths(scene, motion, box, direction)
        for hold, predicted in enumerate(paths_of[direction], 1):
            free = [start, *[tuple(start[k] + b[k] - box[k] for k in range(4)) for b in predicted]]
            landing = _path_landing(free, rects)
            if landing is None or any(abs(b[0] - anchor) > area for b in free[: landing[0] + 1]):
                continue
            groups.append((hold, direction, landing, _rise(free), free))

    def destination_for(hold, direction, landing):
        """A destination the executor makes this very jump for (its rule:
        smb_trajectory.best_holds), nearest the predicted landing; or None."""
        y = round(landing[2])
        xs = [0] if direction == 0 else range(round(landing[1]) - 6, round(landing[1]) + 7)
        xs = [x for x in xs if x * direction >= 0 and x in SKILL_X]
        if not xs or y not in SKILL_Y:
            return None
        tracks = getattr(spatial, "tracks", None) or VisualTracks()
        goals = [_plan_goal(tracks, box, x, y, scene, motion, direction) for x in xs]
        made = best_holds(scene, motion, box, goals, direction, paths_of[direction])
        fits = [x for x, (h, _, _) in zip(xs, made) if h == hold]
        if not fits:
            return None
        return SkillToken("jump", min(fits, key=lambda x: abs(x - landing[1])), y)

    def passing(delay):
        """[(room key, -rise, hold group)] of the jumps started ``delay``
        frames from now that pass a walker and land back in the area."""
        out = []
        for group in groups:
            hold, direction, landing, rise, free = group
            screened = _screen(env, free, 0, delay, enemies, lips, rects)
            if screened is None:
                continue
            distances, landed_on, killed, (dx, _) = screened
            if killed or landed_on != index:
                continue
            passes = False
            for boxes in enemies.values():
                before = boxes[min(delay, len(boxes) - 1)]
                after = boxes[min(delay + landing[0], len(boxes) - 1)]
                if (middle < (before[0] + before[2]) / 2) != (middle + dx < (after[0] + after[2]) / 2):
                    passes = True
            if passes:
                out.append((_room_key(distances), -rise, group))
        return out

    now = sorted(passing(0), key=lambda c: c[:2], reverse=True)
    if not imminent:
        best = now[0][0] if now else None
        for delay in range(2, 21, 2):
            if _screen_walk(_run_path(start, 0, delay), enemies, lips) is None:
                break  # holding that long meets the walker
            later = passing(delay)
            if later and (best is None or max(c[0] for c in later) > best):
                return None  # a later start keeps more room: hold for now
    for _, _, (hold, direction, landing, rise, free) in now[:CERTIFY]:
        command = destination_for(hold, direction, landing)
        if command is None:
            continue
        r = trial(env, state, command, target)
        if (r.safe or r.won) and _move_kind(env, r) == "pass":
            return command
    return None


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


# The steers tried after a stomp for a safe next action (next_action_safe):
# every NEXT_STEP pixels within 32 of Mario, straight down first.
NEXT_STEP = 4


def next_action_safe(env, state, spatial, executor, scene):
    """After a stomp, whether the teacher has a safe next action to teach.

    A stomp ends when Mario's falling feet meet the enemy; what he does next
    is a new action the teacher chooses there. In the air it chooses a steer
    to a point of the floor below within 32 pixels (_air_options, each named
    as running to that floor, or stomping an enemy it comes down on). The
    stomp is safe only if one of them is: it wins, or it is safe as a taught
    action is (reaches the floor at least 8 pixels from every live enemy,
    with room to brake). Tried every NEXT_STEP pixels, straight down first.
    ``spatial``, ``executor`` and ``scene``: the stomp's at its end."""
    m = env.mario
    if not env.stomped or m["on_ground"]:
        return True
    following = copy(state)
    following.execution, following.controller = deepcopy(spatial), executor
    following.scene = scene
    target = teacher_target(env)
    side = getattr(target, "direction", 1)
    cx, feet = m["x"] + m["w"] / 2, m["y"] + m["h"]
    for dx in sorted(range(-32, 33, NEXT_STEP), key=lambda d: (abs(d), d * side < 0)):
        below = [
            p["rect"].top
            for p in env.platforms
            if p["rect"].left <= cx + dx <= p["rect"].right and p["rect"].top >= feet - 2
        ]
        if not below:
            continue
        steer = SkillToken("run", dx, max(SKILL_Y[0], min(SKILL_Y[-1], round(min(below) - feet))))
        result = _trial(env, following, steer, target)
        if result.won or (result.safe and result.clearance >= 8):
            return True
    return False


# A landing this close (pixels between middles) to a live enemy at the same
# height must leave room to brake (stop_safe).
BRAKE_WATCH = 48
# Frames of holding watched after such a landing: NES braking from a full run
# takes about 20.
BRAKE_FRAMES = 24


def stop_safe(env, spatial, executor):
    """A landing needs room to brake: holding from where Mario lands must not
    carry him into a live enemy beside him, nor onto the mouth of a pipe's
    plant. (A landing 6 pixels from a standing enemy was taught; the slide
    after it reached the enemy.) An enemy walking toward him is the next
    decision's to pass: it is held where it stands, so only his own slide is
    judged. One walking away keeps walking (held, it barred every landing
    behind a receding walker, and a stomp was taught under speed run)."""
    m = env.mario
    feet, middle = m["y"] + m["h"], m["x"] + m["w"] / 2
    plant = any(
        e.get("kind") == "piranha_plant"
        and abs(feet - e["pipe_top"]) < 2
        and abs(m["x"] - e["x"]) < 48
        for e in env.enemies
    )
    near = [
        e
        for e in env.enemies
        if m["on_ground"]
        and not e["dead"]
        and e["h"] > 0
        and e.get("kind") != "piranha_plant"
        and abs(e["y"] + e["h"] - feet) < 20
        and abs(e["x"] + e["w"] / 2 - middle) < BRAKE_WATCH
    ]
    if not plant and not near:
        return True
    if plant and on_plant_mouth(env):
        return False
    with probe_state(env):
        held = [
            e
            for e in near
            if e.get("speed", 0) * e.get("direction", 0) * (middle - e["x"] - e["w"] / 2) >= 0
        ]
        spans = [(e["x"], e["x"] + e["w"]) for e in held]
        for e in held:
            e["dead"] = True  # held where they are (the probe restores them)
        feedback = deepcopy(spatial)
        control = SMBExecutor(last_button=executor.last_button)
        scene = scene_from_labels(env.scene_labels())
        plan = feedback.begin(SkillToken("hold", 0, 0), scene)
        control.start(plan, flight=feedback.flight, travel=feedback.travel)
        for frame in range(BRAKE_FRAMES if near else 6):
            if frame:
                control.end("hold_recheck")
                plan = feedback.begin(SkillToken("hold", 0, 0), scene)
                control.start(plan, flight=feedback.flight, travel=feedback.travel)
            button = control.press(scene)
            feedback.executed(button, scene)
            _, _, done, truncated, _ = env.step(button)
            if done or truncated or (plant and on_plant_mouth(env)):
                return bool(env._goal_credited)
            m = env.mario
            if any(m["x"] < right and m["x"] + m["w"] > left for left, right in spans):
                return False
            if not plant and m["on_ground"] and abs(m["vx"]) < 1 / 16:
                break  # stopped
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
    # Every jump Mario can make that ends on the pipe's top or past it, and
    # the screened runs (every pixel he can walk to), certified by trials: of
    # the crossings that work, the one passing farthest from the plant.
    memo = {}
    options = []
    if m["on_ground"]:
        for option in _jump_table(env, state, target, memo, (target.direction,)):
            index = option.result.outcome[1] if option.result.outcome else None
            ends = env.platforms[index]["rect"] if index is not None else None
            if option.result.won or (
                ends is not None and (ends == pipe or ends.left >= pipe.right - 2)
            ):
                options.append((option.command, option.result))
    options += _run_options(env, state, target, memo)
    working = [
        (_rank(goal, result), goal)
        for goal, result in options
        if result.won or result.safe and result.clearance >= 8
    ]
    for _, goal in sorted(working, key=lambda w: w[0], reverse=True)[:CERTIFY]:
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
        # Of every jump Mario can make that clears the monster onto his floor,
        # the one passing farthest from it.
        working = []
        for option in _jump_table(env, state, target, {}, (side,)):
            result = option.result
            index = result.outcome[1] if result.outcome else None
            if not (result.won or (result.safe and result.advanced and result.clearance >= 8)):
                continue
            if not result.won and (index is None or env.platforms[index] is not support):
                continue
            working.append((_rank(option.command, result), option.command))
        for _, goal in sorted(working, key=lambda w: w[0], reverse=True)[:CERTIFY]:
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
        # Back away: every retreat of 8 to 96 pixels, screened by how far it
        # keeps from the monster as it comes (its patrol rule), the best
        # certified by trials.
        enemies = _enemy_paths(env, 300)
        lips = _pit_lips(env)
        start = _body(env)
        # Inside the low tunnel, a retreat must leave it (past the line the
        # segment keeps Mario behind): a shorter one keeps as much room from
        # the monster while it walks, and is trapped when it arrives.
        inside = (center + side * half - line) * side > 0
        screened = []
        for travel in range(8, 97):
            offset = -side * travel
            if not supported(offset) or center + offset < env.camera_x + half + 2:
                continue
            if inside and (center + offset + side * half - line) * side > 0:
                continue
            walked = _screen_walk(
                _run_path(start, offset, _run_frames(env.motion, offset)), enemies, lips
            )
            if walked is not None:
                screened.append((_distance_key(Result(distances=walked)), -travel, offset))
        for _, _, offset in sorted(screened, reverse=True)[:CERTIFY]:
            goal = SkillToken("run", offset, 0)
            result = trial(env, state, goal, target)
            if result.safe and result.clearance >= 8:
                return remember(goal)
    hold = SkillToken("hold", 0, 0)
    result = trial(env, state, hold, target)
    return remember(hold) if result.safe and result.clearance >= 8 else None
