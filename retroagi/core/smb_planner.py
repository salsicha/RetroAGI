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

from copy import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

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


def enemy_paths(enemies: Sequence[Enemy], solids: Sequence[Solid], frames: int) -> None:
    """Each enemy's box every frame for ``frames`` frames (Enemy.path): moving
    at its velocity, turned back when it walks into a wall (a solid rising
    above its feet), as the game turns walkers at pipes and steps."""
    for e in enemies:
        x0, y0, x1, y1 = e.box
        vx, path = e.vx, [e.box]
        for _ in range(frames):
            nx0, nx1 = x0 + vx, x1 + vx
            if vx and any(_overlap(nx0, nx1, s.x0, s.x1) and s.top < y1 - WALL_DEPTH and y0 < s.bottom
                          for s in solids):
                vx = -vx
                nx0, nx1 = x0 + vx, x1 + vx
            x0, x1 = nx0, nx1
            y0, y1 = y0 + e.vy, y1 + e.vy
            path.append((x0, y0, x1, y1))
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


def blocked(x: float, width: float, top: float, feet: float, solids: Sequence[Solid]) -> bool:
    """Whether a body at ``x`` (``width`` wide, from ``top`` to ``feet``)
    would be inside a solid's side: a wall, not a ledge it stands on."""
    return any(
        _overlap(x, x + width, s.x0, s.x1) and feet > s.top + WALL_DEPTH and top < s.bottom for s in solids
    )


def _reached(body: tuple, motion: NESPlayerMotion, enemies: Sequence[Enemy], stomped: Optional[Enemy],
             frame: int) -> Optional[Enemy]:
    """The enemy reaching Mario, landed at ``body`` at ``frame`` and sliding
    on (``motion``, buttons released), within SAFE_AFTER frames (with
    ENEMY_MARGIN); None if none does."""
    m = copy(motion)
    x0, y0, x1, y1 = body
    for k in range(1, SAFE_AFTER + 1):
        dx, _, _ = m.advance(direction=0, jump=False, grounded=True, y=y0, run=False)
        x0, x1 = x0 + dx, x1 + dx
        for e in enemies:
            if e is stomped:
                continue
            b = e.at(frame + k)
            if _overlap(x0, x1, b[0] - ENEMY_MARGIN, b[2] + ENEMY_MARGIN) and _overlap(
                y0, y1, b[1] - ENEMY_MARGIN, b[3] + ENEMY_MARGIN
            ):
                return e
    return None


def rollout(motion: NESPlayerMotion, body: tuple, pre_run: int, hold: int, direction: int,
            solids: Sequence[Solid], enemies: Sequence[Enemy], horizon: int = HORIZON) -> Outcome:
    """Mario from ``body`` (world pixels), standing, running ``pre_run``
    frames then holding the jump button ``hold`` frames, steering
    ``direction`` throughout, until something ends it."""
    m = copy(motion)
    x, y, x1, y1 = body
    width, height = x1 - x, y1 - y
    airborne = False
    stomped = None
    for frame in range(horizon):
        jump = pre_run <= frame < pre_run + hold
        dx, dy, jumped = m.advance(direction=direction, jump=jump, grounded=not airborne, y=y, run=True)
        airborne = airborne or jumped
        nx = x + dx
        feet = y + height
        if blocked(nx, width, y, feet, solids):
            nx = x  # a wall: no sideways motion, as the game stops it
            m.x_speed = m.x_accumulator = m.x_fraction = 0
        if not airborne:
            ny = y
            if not any(_overlap(nx, nx + width, s.x0, s.x1) and abs(s.top - feet) <= STAND for s in solids):
                airborne = True  # ran off a ledge
        else:
            ny = y + dy
            if dy < 0:
                for s in solids:
                    if _overlap(nx, nx + width, s.x0, s.x1) and ny < s.bottom <= y:
                        return Outcome("bonk", frame, (nx, ny, nx + width, ny + height), solid=s)
            else:
                for s in solids:
                    if _overlap(nx, nx + width, s.x0, s.x1) and feet <= s.top <= ny + height:
                        body = (nx, s.top - height, nx + width, s.top)
                        threat = _reached(body, m, enemies, stomped, frame)
                        if threat is not None:
                            return Outcome("hit", frame, body, enemy=threat)
                        return Outcome("landed", frame, body, solid=s, stomped=stomped)
        now = (nx, ny, nx + width, ny + height)
        for e in enemies:
            if e is stomped:
                continue
            b = e.at(frame + 1)
            near = (b[0] - ENEMY_MARGIN, b[1] - ENEMY_MARGIN, b[2] + ENEMY_MARGIN, b[3] + ENEMY_MARGIN)
            if not (_overlap(now[0], now[2], near[0], near[2]) and _overlap(now[1], now[3], near[1], near[3])):
                continue
            touching = _overlap(now[0], now[2], b[0], b[2]) and _overlap(now[1], now[3], b[1], b[3])
            across = min(now[2], b[2]) - max(now[0], b[0])
            from_above = feet <= b[1]  # last frame's feet above its top
            if (
                stomped is None and touching and airborne and dy > 0 and e.stompable
                and from_above and across >= STOMP_OVERLAP
            ):
                stomped = e
                m.y_speed, m.y_force, m.force = STOMP_BOUNCE, 0, m.force_down
                break
            return Outcome("hit", frame, now, enemy=e)
        if ny > SCREEN_BOTTOM:
            return Outcome("fell", frame, now)
        x, y = nx, ny
    return Outcome("timeout", horizon, (x, y, x + width, y + height))


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
    enemy_paths(enemies, solids, HORIZON + SAFE_AFTER + 1)
    tries = [(pre_run, hold) for pre_run in PRE_RUNS if pre_run <= latest_takeoff for hold in NES_JUMP_FRAMES]
    if latest_takeoff >= HORIZON:
        tries.insert(0, (HORIZON, 0))
    for pre_run, hold in tries:
        outcome = rollout(motion, body, pre_run, hold, direction, solids, enemies)
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
