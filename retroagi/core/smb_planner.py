"""The jump planner of the choose-and-control controller
(docs/choose-and-control-plan.md phase 1; user, 2026-11-07): the learned
layers choose a goal (land on this thing, stomp this enemy, pass it), the
planner works out how, from what perception reports.

It searches takeoff after a few frames of running (PRE_RUNS) and every
jump-button hold (1 to 32 frames), and walking on without jumping (off a
ledge, down a step), rolling each out with the NES motion
model (smb_physics) and checking it against the scene: walls (which stop
Mario's sideways motion, as the game's do: a jump pressed against a pipe
rises along it and over), blocks hit from below, landings on tops, enemies
met on their estimated paths (their velocity, turned back at walls as the
game turns walkers: enemy_paths). The plan reaching the goal with the most margin wins;
none reaching it is an answer too (None: infeasible, for the chooser to
decide again).

Everything is in world pixels (screen pixels plus the camera's position),
so the camera's scrolling during a jump changes nothing.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numba import njit

from .smb_observer import packed_lists
from .smb_physics import NES_JUMP_FRAMES, NESPlayerMotion

# Frames of running before takeoff the planner tries.
PRE_RUNS = (0, 2, 4, 6, 8, 12, 16, 20, 24, 32)
# Frames a rollout lasts at most.
HORIZON = 96
# Below this the screen ends: Mario has fallen (pixels).
SCREEN_BOTTOM = 240
# A side touching a solid this far below its top is a wall, not a ledge.
WALL_DEPTH = 2
# Feet this close to a solid's top stand on it (pixels).
STAND = 1
# Enemies are kept this much wider on every side than their boxes when a
# hit is checked (pixels): the game's boxes include their last pixel, and
# estimates are not exact. A stomp needs the box itself.
ENEMY_MARGIN = 2
# A stomp bounces Mario up at this vertical speed (the game's $FC).
STOMP_BOUNCE = -4
# A stomp needs this much sideways overlap with the enemy (pixels): in 1-1 a
# fall grazing a Goomba's top corner by one pixel was a hit, not a stomp.
STOMP_OVERLAP = 4
# A landing is safe only if no enemy reaches Mario within this many frames
# after it, as he slides on his momentum, buttons released (time to decide
# and jump again): dropping off 1-1's second pipe and sliding toward a
# Goomba left no room to jump it.
SAFE_AFTER = 12
# A pass lands this far clear of the enemy (pixels).
PASS_GAP = 4
# How far inside a thing's ends a landing is kept (pixels; Mario is 10-12
# wide): the goal region's margin. On a thing too narrow for it (a 16-pixel
# step), at least NARROW_KEEP of it: Mario's middle stays on the top.
INSET = 8
NARROW_KEEP = 3
# Buttons: SMBAction RIGHT 1, LEFT 3, RIGHT_JUMP 2, LEFT_JUMP 4, NOOP 0, JUMP 5.
BUTTON = {(1, False): 1, (-1, False): 3, (0, False): 0, (1, True): 2, (-1, True): 4, (0, True): 5}


@dataclass
class Solid:
    """Something Mario cannot pass through, in world pixels; ``ref`` names
    it as the policy's pointers do ((list, slot) into packed_lists)."""

    x0: float
    top: float
    x1: float
    bottom: float
    kind: str
    ref: tuple


@dataclass
class Enemy:
    """An enemy's box now (world pixels) and its velocity (pixels a frame);
    ``path``: its box each frame from now, when predicted (enemy_paths)."""

    box: tuple
    vx: float
    vy: float
    ref: tuple
    stompable: bool = True
    path: Optional[list] = None

    def at(self, frame: int) -> tuple:
        if self.path is not None and frame < len(self.path):
            return self.path[frame]
        b = self.box
        dx, dy = self.vx * frame, self.vy * frame
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


def _straight(e: Enemy, frames: int) -> np.ndarray:
    """``e``'s boxes for ``frames`` frames from now at its velocity."""
    k = np.arange(frames + 1, dtype=np.float64)[:, None]
    return np.asarray(e.box, np.float64)[None, :] + k * np.array([e.vx, e.vy, e.vx, e.vy])


def enemy_paths(enemies: Sequence[Enemy], solids: Sequence[Solid], frames: int) -> None:
    """Each enemy's box every frame for ``frames`` frames (Enemy.path): moving
    at its velocity, turned back when it walks into a wall (a solid rising
    above its feet), as the game turns walkers at pipes and steps."""
    if not enemies:
        return
    boxes = np.array([e.box for e in enemies], np.float64)
    velocities = np.array([(e.vx, e.vy) for e in enemies], np.float64)
    paths = _paths(boxes, velocities, _solid_array(solids), frames)
    for e, path in zip(enemies, paths):
        e.path = path


@dataclass
class Goal:
    """What the chooser asked for: "land" on ``ref`` (inside x0..x1 by
    INSET), "stomp" the enemy ``ref``, or "pass" it (land on its ``side``,
    1 right of it, -1 left, PASS_GAP clear of where it is at the landing,
    without touching it: a walker coming at Mario can pass under a jump).
    ``prefer``: where on a landing, "near", "middle" or "far"."""

    kind: str
    ref: tuple
    x0: float = 0.0
    x1: float = 0.0
    prefer: str = "middle"
    side: int = 1


@dataclass
class Outcome:
    """How a rollout ends: "landed" (on ``solid``), "hit" (``enemy``; also a
    landing an enemy reaches within SAFE_AFTER frames), "bonk" (a block from
    below), "fell", "timeout"; at ``frame``, Mario's body then. A stomp on the way bounces Mario and the rollout goes on to
    his landing; ``stomped`` is the enemy stomped."""

    end: str
    frame: int
    body: tuple
    solid: Optional[Solid] = None
    enemy: Optional[Enemy] = None
    stomped: Optional[Enemy] = None


@dataclass
class Plan:
    pre_run: int
    hold: int
    direction: int
    outcome: Outcome
    margin: float
    buttons: list = field(default_factory=list)


def _overlap(a0, a1, b0, b1) -> bool:
    return a0 < b1 and b0 < a1


def solids_of(scene, camera: float) -> list:
    """Every still thing vision reports as a solid, in world pixels: pipes
    and blocks by their boxes, the floor and steps from their tops down to
    the screen's bottom, moving surfaces thin."""
    lists = packed_lists(scene)
    out = []
    for slot, p in enumerate(lists.get("pipes", ())):
        out.append(Solid(p[0] + camera, p[1], p[2] + camera, SCREEN_BOTTOM, "pipe", ("pipes", slot)))
    for slot, b in enumerate(lists.get("blocks", ())):
        x0, y0, x1, y1 = b.box
        out.append(Solid(x0 + camera, y0, x1 + camera, y1, "block", ("blocks", slot)))
    for slot, s in enumerate(lists.get("surfaces", ())):
        x0, x1 = s.x0 + camera, s.x1 + camera
        covered = any(
            _overlap(x0, x1, o.x0, o.x1) and abs(o.top - s.top) <= 2 for o in out if o.kind in ("pipe", "block")
        )
        if covered:
            continue
        bottom = s.top + 8 if s.moving else SCREEN_BOTTOM
        out.append(Solid(x0, s.top, x1, bottom, "moving" if s.moving else "ground", ("surfaces", slot)))
    return out


# The compiled core (numba): the same physics and checks as plain Python
# loops would do, about a hundred times faster (a plan took 275 ms in
# Python, most of the controller's time; the controller is meant to be
# arithmetic of about a millisecond a frame).
_MOTION_FIELDS = (
    "x_speed", "x_accumulator", "x_fraction", "y_speed", "y_force", "y_fraction", "force_up",
    "force_down", "force", "origin_y", "previous_jump", "facing", "moving", "running_speed", "running_timer",
)
ENDS = ("landed", "hit", "bonk", "fell", "timeout")
_LANDED, _HIT, _BONK, _FELL, _TIMEOUT = range(5)


def _motion_array(motion: NESPlayerMotion) -> np.ndarray:
    return np.array([int(getattr(motion, f)) for f in _MOTION_FIELDS], np.int64)


def _solid_array(solids: Sequence[Solid]) -> np.ndarray:
    return np.array([(s.x0, s.top, s.x1, s.bottom) for s in solids], np.float64).reshape(-1, 4)


@njit(cache=True)
def _advance(m, direction, jump, grounded, y, run):
    """NESPlayerMotion.advance on its fields as an array (_MOTION_FIELDS):
    (dx, dy, jumped)."""
    m[14] = max(0, m[14] - 1)
    absolute = abs(m[0])
    jumped = jump and m[10] == 0 and grounded
    if jumped:
        index = (absolute >= 9) + (absolute >= 16) + (absolute >= 25) + (absolute >= 28)
        m[6] = (0x20, 0x20, 0x1E, 0x28, 0x28)[index]
        m[7] = (0x70, 0x70, 0x60, 0x90, 0x90)[index]
        m[8] = m[6]
        m[3] = (-4, -4, -4, -5, -5)[index]
        m[4] = 0
        m[5] = 0
        m[9] = int(y)
        grounded = False
    if grounded:
        running = direction == m[12] and (run or m[14] > 0)
        if direction == m[12] and run:
            m[14] = 10
    else:
        running = absolute >= 25
    maximum = 40 if running else 24
    friction = 0xE4 if running else (0xD0 if (m[13] != 0 or absolute >= 33) else 0x98)
    if m[11] != m[12]:
        friction *= 2
    if grounded:
        if absolute >= 28:
            m[13] = absolute
        elif direction == m[12]:
            m[13] = 0
        elif direction != 0 and absolute < 11:
            m[12] = m[11]
            m[0] = 0
            m[1] = 0
        if direction != 0:
            m[11] = direction
    if grounded or direction != 0:
        if direction != 0:
            sign = direction
        else:
            sign = -1 if m[0] > 0 else (1 if m[0] < 0 else 0)
        if sign != 0:
            total = m[1] + sign * friction
            m[1] = total % 256
            m[0] += total // 256
            m[0] = min(maximum, m[0]) if sign > 0 else max(-maximum, m[0])
    t = m[2] + m[0] * 16
    dx = t // 256
    m[2] = t % 256
    if m[0] != 0:
        m[12] = 1 if m[0] > 0 else -1
    dy = 0
    if not grounded:
        if m[3] >= 0 or (not (jump and m[10] != 0) and m[9] - int(y) >= 1):
            m[8] = m[7]
        t = m[5] + m[4]
        m[5] = t % 256
        dy = m[3] + t // 256
        t = m[4] + m[8]
        m[4] = t % 256
        m[3] += t // 256
        if m[3] >= 4 and m[4] >= 128:
            m[3] = 4
            m[4] = 0
    m[10] = 1 if jump else 0
    return dx, dy, jumped


@njit(cache=True)
def _blocked(x, width, top, feet, solids):
    for i in range(solids.shape[0]):
        if x < solids[i, 2] and solids[i, 0] < x + width and feet > solids[i, 1] + WALL_DEPTH and top < solids[i, 3]:
            return True
    return False


@njit(cache=True)
def _paths(boxes, velocities, solids, frames):
    out = np.empty((boxes.shape[0], frames + 1, 4))
    for e in range(boxes.shape[0]):
        x0, y0, x1, y1 = boxes[e, 0], boxes[e, 1], boxes[e, 2], boxes[e, 3]
        vx, vy = velocities[e, 0], velocities[e, 1]
        out[e, 0] = (x0, y0, x1, y1)
        for k in range(1, frames + 1):
            nx0, nx1 = x0 + vx, x1 + vx
            if vx != 0.0:
                for i in range(solids.shape[0]):
                    if (nx0 < solids[i, 2] and solids[i, 0] < nx1 and solids[i, 1] < y1 - WALL_DEPTH
                            and y0 < solids[i, 3]):
                        vx = -vx
                        nx0, nx1 = x0 + vx, x1 + vx
                        break
            x0, x1, y0, y1 = nx0, nx1, y0 + vy, y1 + vy
            out[e, k] = (x0, y0, x1, y1)
    return out


@njit(cache=True)
def _reached(x0, y0, x1, y1, m0, paths, stomped, frame):
    """The enemy reaching Mario, landed at the box at ``frame`` and sliding
    on (buttons released), within SAFE_AFTER frames (with ENEMY_MARGIN);
    -1 if none does."""
    m = m0.copy()
    last = paths.shape[1] - 1
    for k in range(1, SAFE_AFTER + 1):
        dx, _, _ = _advance(m, 0, False, True, y0, False)
        x0 += dx
        x1 += dx
        for e in range(paths.shape[0]):
            if e == stomped:
                continue
            b = paths[e, min(frame + k, last)]
            if (x0 < b[2] + ENEMY_MARGIN and b[0] - ENEMY_MARGIN < x1
                    and y0 < b[3] + ENEMY_MARGIN and b[1] - ENEMY_MARGIN < y1):
                return e
    return -1


@njit(cache=True)
def _rollout(m0, x, y, x1, y1, pre_run, hold, direction, solids, paths, stompable, horizon):
    """rollout on arrays: (end, frame, body x0, y0, x1, y1, solid, enemy,
    stomped), indexes -1 where none."""
    m = m0.copy()
    width, height = x1 - x, y1 - y
    airborne = False
    stomped = -1
    last = paths.shape[1] - 1
    for frame in range(horizon):
        jump = pre_run <= frame < pre_run + hold
        dx, dy, jumped = _advance(m, direction, jump, not airborne, y, True)
        airborne = airborne or jumped
        nx = x + dx
        feet = y + height
        if _blocked(nx, width, y, feet, solids):
            nx = x  # a wall: no sideways motion, as the game stops it
            m[0] = 0
            m[1] = 0
            m[2] = 0
        if not airborne:
            ny = y
            supported = False
            for i in range(solids.shape[0]):
                if nx < solids[i, 2] and solids[i, 0] < nx + width and abs(solids[i, 1] - feet) <= STAND:
                    supported = True
                    break
            if not supported:
                airborne = True  # ran off a ledge
        else:
            ny = y + dy
            if dy < 0:
                for i in range(solids.shape[0]):
                    if nx < solids[i, 2] and solids[i, 0] < nx + width and ny < solids[i, 3] <= y:
                        return _BONK, frame, nx, ny, nx + width, ny + height, i, -1, stomped
            else:
                for i in range(solids.shape[0]):
                    if nx < solids[i, 2] and solids[i, 0] < nx + width and feet <= solids[i, 1] <= ny + height:
                        top = solids[i, 1]
                        threat = _reached(nx, top - height, nx + width, top, m, paths, stomped, frame)
                        if threat >= 0:
                            return _HIT, frame, nx, top - height, nx + width, top, -1, threat, stomped
                        return _LANDED, frame, nx, top - height, nx + width, top, i, -1, stomped
        hit = -1
        for e in range(paths.shape[0]):
            if e == stomped:
                continue
            b = paths[e, min(frame + 1, last)]
            if not (nx < b[2] + ENEMY_MARGIN and b[0] - ENEMY_MARGIN < nx + width
                    and ny < b[3] + ENEMY_MARGIN and b[1] - ENEMY_MARGIN < ny + height):
                continue
            touching = nx < b[2] and b[0] < nx + width and ny < b[3] and b[1] < ny + height
            across = min(nx + width, b[2]) - max(nx, b[0])
            from_above = feet <= b[1]  # last frame's feet above its top
            if (stomped < 0 and touching and airborne and dy > 0 and stompable[e]
                    and from_above and across >= STOMP_OVERLAP):
                stomped = e
                m[3] = STOMP_BOUNCE
                m[4] = 0
                m[8] = m[7]
                break
            hit = e
            break
        if hit >= 0:
            return _HIT, frame, nx, ny, nx + width, ny + height, -1, hit, stomped
        if ny > SCREEN_BOTTOM:
            return _FELL, frame, nx, ny, nx + width, ny + height, -1, -1, stomped
        x, y = nx, ny
    return _TIMEOUT, horizon, x, y, x + width, y + height, -1, -1, stomped


def blocked(x: float, width: float, top: float, feet: float, solids: Sequence[Solid]) -> bool:
    """Whether a body at ``x`` (``width`` wide, from ``top`` to ``feet``)
    would be inside a solid's side: a wall, not a ledge it stands on."""
    return bool(_blocked(float(x), float(width), float(top), float(feet), _solid_array(solids)))


def _packed(enemies: Sequence[Enemy], frames: int) -> tuple:
    """(paths [enemies, frames + 1, 4], stompable) for the compiled core:
    each enemy's predicted path (or its straight course when none is)."""
    paths = np.zeros((len(enemies), frames + 1, 4))
    for i, e in enumerate(enemies):
        path = np.asarray(e.path, np.float64) if e.path is not None else _straight(e, frames)
        n = min(len(path), frames + 1)
        paths[i, :n] = path[:n]
        paths[i, n:] = path[n - 1]
    return paths, np.array([e.stompable for e in enemies], np.bool_)


def _outcome(raw, solids: Sequence[Solid], enemies: Sequence[Enemy]) -> Outcome:
    end, frame, x0, y0, x1, y1, solid, enemy, stomped = raw
    return Outcome(
        ENDS[end], int(frame), (float(x0), float(y0), float(x1), float(y1)),
        solid=solids[solid] if solid >= 0 else None,
        enemy=enemies[enemy] if enemy >= 0 else None,
        stomped=enemies[stomped] if stomped >= 0 else None,
    )


def rollout(motion: NESPlayerMotion, body: tuple, pre_run: int, hold: int, direction: int,
            solids: Sequence[Solid], enemies: Sequence[Enemy], horizon: int = HORIZON) -> Outcome:
    """Mario from ``body`` (world pixels), standing, running ``pre_run``
    frames then holding the jump button ``hold`` frames, steering
    ``direction`` throughout, until something ends it."""
    paths, stompable = _packed(enemies, horizon + SAFE_AFTER + 1)
    x, y, x1, y1 = (float(v) for v in body)
    raw = _rollout(_motion_array(motion), x, y, x1, y1, pre_run, hold, direction, _solid_array(solids),
                   paths, stompable, horizon)
    return _outcome(raw, solids, enemies)


def _meets(goal: Goal, outcome: Outcome, enemies: Sequence[Enemy] = ()) -> Optional[float]:
    """The margin by which ``outcome`` meets ``goal`` (larger is better);
    None if it does not."""
    middle = (outcome.body[0] + outcome.body[2]) / 2
    if goal.kind == "stomp":
        stomped = outcome.end == "landed" and outcome.stomped is not None and outcome.stomped.ref == goal.ref
        return 0.0 if stomped else None
    if goal.kind == "pass":
        enemy = next((e for e in enemies if e.ref == goal.ref), None)
        if outcome.end != "landed" or enemy is None:
            return None
        b = enemy.at(outcome.frame + 1)
        edge = outcome.body[0] if goal.side > 0 else outcome.body[2]
        beyond = (edge - b[2]) - PASS_GAP if goal.side > 0 else (b[0] - edge) - PASS_GAP
        if beyond < 0:
            return None
        return min(beyond, 64.0)
    if outcome.end != "landed" or outcome.solid.ref != goal.ref:
        return None
    inset = min(INSET, max(NARROW_KEEP, (goal.x1 - goal.x0) / 2 - NARROW_KEEP))
    left, right = goal.x0 + inset, goal.x1 - inset
    if not left - 0.5 <= middle <= right + 0.5:
        return None
    room = min(middle - goal.x0, goal.x1 - middle)
    if goal.prefer == "near":
        return room - abs(middle - left)
    if goal.prefer == "far":
        return room - abs(right - middle)
    return room


def buttons_of(pre_run: int, hold: int, direction: int, frames: int) -> list:
    """The button pressed each frame of a plan, as the rollout pressed them."""
    return [BUTTON[direction, pre_run <= f < pre_run + hold] for f in range(frames)]


def plan_jump(goal: Goal, motion: NESPlayerMotion, body: tuple, solids: Sequence[Solid],
              enemies: Sequence[Enemy], direction: int, latest_takeoff: int = HORIZON) -> Optional[Plan]:
    """The plan meeting ``goal`` with the most margin (then the soonest);
    None when no takeoff and hold (nor walking on) does: the goal is
    infeasible from here. ``latest_takeoff``: no plan taking off later
    (frames of running) than this; walking on without jumping is a plan that
    never takes off, allowed only when unbounded."""
    best = None
    frames = HORIZON + SAFE_AFTER + 1
    enemy_paths(enemies, solids, frames)
    paths, stompable = _packed(enemies, frames)
    m, packed_solids = _motion_array(motion), _solid_array(solids)
    x, y, x1, y1 = (float(v) for v in body)
    tries = [(pre_run, hold) for pre_run in PRE_RUNS if pre_run <= latest_takeoff for hold in NES_JUMP_FRAMES]
    if latest_takeoff >= HORIZON:
        tries.insert(0, (HORIZON, 0))
    for pre_run, hold in tries:
        raw = _rollout(m, x, y, x1, y1, pre_run, hold, direction, packed_solids, paths, stompable, HORIZON)
        if raw[0] != _LANDED:
            continue  # every goal ends standing
        outcome = _outcome(raw, solids, enemies)
        margin = _meets(goal, outcome, enemies)
        if margin is None:
            continue
        key = (margin, -outcome.frame)
        if best is None or key > best[0]:
            best = (key, Plan(pre_run, hold, direction, outcome, margin))
    if best is None:
        return None
    plan = best[1]
    plan.buttons = buttons_of(plan.pre_run, plan.hold, plan.direction, plan.outcome.frame + 1)
    return plan
