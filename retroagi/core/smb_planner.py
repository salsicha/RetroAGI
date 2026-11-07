"""The jump planner of the choose-and-control controller
(docs/choose-and-control-plan.md phase 1; user, 2026-11-07): the learned
layers choose a goal (land on this thing, stomp this enemy, pass it), the
planner works out how, from what perception reports.

It searches takeoff after a few frames of running (PRE_RUNS) and every
jump-button hold (1 to 32 frames), rolling each out with the NES motion
model (smb_physics) and checking it against the scene: walls, blocks hit
from below, landings on tops, enemies met on their estimated paths
(constant velocity). The plan reaching the goal with the most margin wins;
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
# Landing on an enemy's top within this of its top is a stomp (pixels).
STOMP_REACH = 4
# How far inside a thing's ends a landing is kept (pixels; Mario is 10-12
# wide): the goal region's margin.
INSET = 8
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
    """An enemy's box now (world pixels) and its velocity (pixels a frame)."""

    box: tuple
    vx: float
    vy: float
    ref: tuple
    stompable: bool = True

    def at(self, frame: int) -> tuple:
        b = self.box
        dx, dy = self.vx * frame, self.vy * frame
        return (b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy)


@dataclass
class Goal:
    """What the chooser asked for: "land" on ``ref`` (inside x0..x1 by
    INSET), "stomp" the enemy ``ref``, or "pass" it (land beyond ``x1``,
    world pixels, without touching it). ``prefer``: where on a landing,
    "near", "middle" or "far"."""

    kind: str
    ref: tuple
    x0: float = 0.0
    x1: float = 0.0
    prefer: str = "middle"


@dataclass
class Outcome:
    """How a rollout ends: "landed" (on ``solid``), "stomp" or "hit" (an
    enemy), "wall", "bonk" (a block from below), "fell", "timeout"; at
    ``frame``, Mario's body then."""

    end: str
    frame: int
    body: tuple
    solid: Optional[Solid] = None
    enemy: Optional[Enemy] = None


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


def rollout(motion: NESPlayerMotion, body: tuple, pre_run: int, hold: int, direction: int,
            solids: Sequence[Solid], enemies: Sequence[Enemy], horizon: int = HORIZON) -> Outcome:
    """Mario from ``body`` (world pixels), standing, running ``pre_run``
    frames then holding the jump button ``hold`` frames, steering
    ``direction`` throughout, until something ends it."""
    m = copy(motion)
    x, y, x1, y1 = body
    width, height = x1 - x, y1 - y
    airborne = False
    for frame in range(horizon):
        jump = pre_run <= frame < pre_run + hold
        dx, dy, jumped = m.advance(direction=direction, jump=jump, grounded=not airborne, y=y, run=True)
        airborne = airborne or jumped
        nx = x + dx
        feet = y + height
        for s in solids:
            if _overlap(nx, nx + width, s.x0, s.x1) and feet > s.top + WALL_DEPTH and y < s.bottom:
                return Outcome("wall", frame, (x, y, x + width, feet), solid=s)
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
                        return Outcome("landed", frame, (nx, s.top - height, nx + width, s.top), solid=s)
        now = (nx, ny, nx + width, ny + height)
        for e in enemies:
            b = e.at(frame + 1)
            if _overlap(now[0], now[2], b[0], b[2]) and _overlap(now[1], now[3], b[1], b[3]):
                stomp = airborne and dy > 0 and e.stompable and feet <= b[1] + STOMP_REACH
                return Outcome("stomp" if stomp else "hit", frame, now, enemy=e)
        if ny > SCREEN_BOTTOM:
            return Outcome("fell", frame, now)
        x, y = nx, ny
    return Outcome("timeout", horizon, (x, y, x + width, y + height))


def _meets(goal: Goal, outcome: Outcome) -> Optional[float]:
    """The margin by which ``outcome`` meets ``goal`` (larger is better);
    None if it does not."""
    middle = (outcome.body[0] + outcome.body[2]) / 2
    if goal.kind == "stomp":
        return 0.0 if outcome.end == "stomp" and outcome.enemy.ref == goal.ref else None
    if goal.kind == "pass":
        if outcome.end != "landed" or middle < goal.x1:
            return None
        return min(middle - goal.x1, 64.0)
    if outcome.end != "landed" or outcome.solid.ref != goal.ref:
        return None
    left, right = goal.x0 + INSET, goal.x1 - INSET
    if right - left < 0:
        left = right = (goal.x0 + goal.x1) / 2
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
              enemies: Sequence[Enemy], direction: int) -> Optional[Plan]:
    """The plan meeting ``goal`` with the most margin (then the soonest);
    None when no takeoff and hold does: the goal is infeasible from here."""
    best = None
    for pre_run in PRE_RUNS:
        for hold in NES_JUMP_FRAMES:
            outcome = rollout(motion, body, pre_run, hold, direction, solids, enemies)
            margin = _meets(goal, outcome)
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
