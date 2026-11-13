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

In the air, plan_flight corrects a jump whose landing has stopped meeting
its goal (an enemy woken by the scroll after takeoff waiting where Mario
comes down): it searches how much longer the jump button stays held and a
few frames of steering (released or braking, STEERS) before steering on
toward the goal again.

Everything is in world pixels (screen pixels plus the camera's position),
so the camera's scrolling during a jump changes nothing.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from numba import njit

from .physics import NES_JUMP_FRAMES, NESPlayerMotion

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
# Walkers: their feet up to this far below their boxes (pixels: a Goomba's
# collision box ends 4 above its feet), and how they fall off a ledge
# (pixels a frame, gained each frame and at most: the game's, fitted to
# Goombas and Koopas falling in worlds 1-1 to 5-1, and the simulator's
# under the game's rules; the 0.5 up to 4.5 written first put a falling
# Goomba 11 pixels off on average).
FEET_BELOW = 4
ENEMY_GRAVITY = 0.2
ENEMY_FALL = 3.4
# What an enemy is, for its path (enemy_paths): a walker walks and falls off
# ledges; a hopper (a Paratroopa) hops HOP_SPEED pixels a frame up each time
# it lands, under HOP_GRAVITY; a plant rises out of its pipe at PLANT_SPEED
# to PLANT_TOP above the pipe's top (its box's bottom) and stays (sinking, it
# goes down and stays: Mario on or by its pipe keeps it down); a flier (a
# Bullet Bill, anything else) flies straight through everything; a thrown
# thing (a Hammer Bro's hammer) flies through everything falling under
# THROWN_GRAVITY to at most THROWN_FALL (Block SMB's hammer, measured in the
# game: sim/env.HAMMER_GRAVITY and HAMMER_FALL); a pacer (a red Koopa) is a
# walker that turns back where the floor under its middle would end; an egg
# (Lakitu's) falls straight down under EGG_GRAVITY to at most EGG_FALL until
# its feet (EGG_FEET below its box) meet a top, then walks at its vx, the
# walk a Spiny starts on landing (Block SMB's: sim/env.EGG_GRAVITY,
# EGG_FALL); a leaper (a leaping Cheep Cheep) flies through everything
# falling under LEAPER_GRAVITY (sim/env.CHEEP_GRAVITY).
ENEMY_KINDS = ("walker", "hopper", "plant", "flier", "thrown", "pacer", "egg", "leaper")
THROWN_GRAVITY, THROWN_FALL = 0.0625, 4.5
EGG_GRAVITY, EGG_FALL, EGG_FEET = 0.125, 3.25, 4.0
LEAPER_GRAVITY = 0.0524
HOP_SPEED, HOP_GRAVITY, HOPPER_FEET = -3.25, 0.12, 3
PLANT_SPEED, PLANT_TOP = 0.5, 4
# A plant's box is at most this far below its pipe's top while it is seen
# (its head showing: pixels).
PLANT_DEEP = 24
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
# A jump starts only while Mario would still stand on something this many
# pixels further the way he moves (standing still, where he is); else the
# rollout says "fell" (no plan): he would run off before the jump took.
# Vision placed 1-1's second pipe's end 2 pixels too far, a run-up and jump
# from its last pixels was planned, and Mario dropped off unjumped behind a
# Goomba. (Measured on both sides, standing at the back of a step vision
# began a pixel late refused every jump up the 2368 staircase.)
TAKEOFF_KEEP = 4
# _reached's answer for a landing Mario slides off into a pit.
SLID_OFF = -2
# In the air with no landing that safe, one no enemy reaches within this
# many frames is taken (plan_flight): Mario can jump again as he lands.
LAST_RESORT_SAFE = 4
# A pass lands this far clear of the enemy (pixels).
PASS_GAP = 4
# Plans that still meet their goal from this many pixels further back are
# preferred (plan_jump): on a 16-pixel step across 1-1's second staircase
# pit, the soonest landing was a 3-frame hop that a few pixels of vision's
# error put against the step's side.
ROBUST_SHIFT = 3
# How far inside a thing's ends a landing is kept (pixels; Mario is 10-12
# wide): the goal region's margin. On a thing too narrow for it (a 16-pixel
# step), at least NARROW_KEEP of it: Mario's middle stays on the top.
INSET = 8
NARROW_KEEP = 3
# Frames of steering (released, or braking against the jump's direction) an
# in-flight correction tries before steering toward the goal again.
STEERS = (2, 4, 8, 12, 16, 24, 32, HORIZON)
# Further frames of the jump button an in-flight correction tries, while
# the button is still held.
FLIGHT_HOLDS = (0, 2, 4, 8, 12, 16, 24, 32)
# A gap in the floor this narrow (pixels) with an enemy standing in it,
# feet at most HIDDEN_FEET above the floor's top (or 2 below: vision's
# boxes include their last pixel row), is the enemy hiding the floor
# (floor_behind).
HIDDEN_GAP = 20
HIDDEN_FEET = 8
# The kinds of solid that are blocks (a question block not yet hit is one).
BLOCK_KINDS = ("block", "question")
# Buttons: SMBAction RIGHT 1, LEFT 3, RIGHT_JUMP 2, LEFT_JUMP 4, NOOP 0, JUMP 5.
BUTTON = {(1, False): 1, (-1, False): 3, (0, False): 0, (1, True): 2, (-1, True): 4, (0, True): 5}


@dataclass
class Solid:
    """Something Mario cannot pass through, in world pixels; ``ref`` names
    it as the policy's pointers do ((list, slot) into packed_lists); a lift
    moves ``vx``, ``vy`` pixels a frame (the planner keeps it at that speed:
    _solid_paths), carrying Mario standing on it."""

    x0: float
    top: float
    x1: float
    bottom: float
    kind: str
    ref: tuple
    vx: float = 0.0
    vy: float = 0.0


@dataclass
class Enemy:
    """An enemy's box now (world pixels) and its velocity (pixels a frame);
    ``path``: its box each frame from now, when predicted (a dynamics
    model's paths). For a model identified from what is seen
    (framework/observed_dynamics): ``track``, its places as seen, oldest
    first and ending now ((x, y) of one point of it each frame; None: not
    kept), and ``type``, what the state source calls it (any label: things
    of one type are taken to move alike)."""

    box: tuple
    vx: float
    vy: float
    ref: tuple
    stompable: bool = True
    path: Optional[list] = None
    kind: str = "walker"  # one of ENEMY_KINDS
    track: Optional[tuple] = None
    type: object = None

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
    """Each enemy's box every frame for ``frames`` frames (Enemy.path), by its
    kind (ENEMY_KINDS): moving at its velocity, turned back when it walks
    into a wall (a solid rising above its feet), as the game turns walkers
    at pipes and steps, and falling once it walks off what it stands on (a
    Goomba off a brick row); hopping; rising from its pipe; flying."""
    if not enemies:
        return
    boxes = np.array([e.box for e in enemies], np.float64)
    velocities = np.array([(e.vx, e.vy) for e in enemies], np.float64)
    kinds = np.array([ENEMY_KINDS.index(e.kind) for e in enemies], np.int64)
    # (the screen's edge stops Mario only: walkers walk off the screen)
    paths = _paths(boxes, velocities, kinds, _solid_array([s for s in solids if s.kind != "edge"]), frames)
    for e, path in zip(enemies, paths):
        e.path = path


class RuleDynamics:
    """The planner's model of how the game's things move, written by hand
    from the game: each by its kind (enemy_paths). Any object with the same
    ``paths`` can stand in for it (framework/interfaces.Dynamics): a model
    identified from what is seen, or a learned one."""

    paths = staticmethod(enemy_paths)

    def observe(self, things: Sequence, solids: Sequence) -> None:
        pass  # (the rules are fixed: nothing seen changes them)

    def __deepcopy__(self, memo) -> "RuleDynamics":
        return self  # (and there is nothing learned to copy)


RULES = RuleDynamics()


@dataclass
class Goal:
    """What the chooser asked for: "land" on the thing ``ref`` whose top is
    ``top``, inside x0..x1 by INSET (matched by where it is: the slots
    renumber as the camera scrolls; ``top`` None: on whatever lies there,
    a staircase's steps), "bump" the block over x0..x1 whose bottom is
    ``top`` from below, "collect" the item ``ref``, "stomp" the enemy ``ref``, or "pass" it
    (land on its ``side``, 1 right of it, -1 left, PASS_GAP clear of where
    it is at the landing, without touching it: a walker coming at Mario can
    pass under a jump). ``prefer``: where on a landing, "near", "middle" or
    "far"."""

    kind: str
    ref: tuple
    x0: float = 0.0
    x1: float = 0.0
    prefer: str = "middle"
    side: int = 1
    top: Optional[float] = None


@dataclass
class Outcome:
    """How a rollout ends: "landed" (on ``solid``), "hit" (``enemy``; also a
    landing an enemy reaches within SAFE_AFTER frames), "fell" (also a jump
    started within TAKEOFF_KEEP of an end), "timeout"; at ``frame``, Mario's
    body then. A stomp on the way bounces Mario, and a block hit from below
    stops his rise, and the rollout goes on to his landing; ``stomped`` is
    the enemy stomped, ``bumped`` the block hit, ``collected`` the item he
    touched."""

    end: str
    frame: int
    body: tuple
    solid: Optional[Solid] = None
    enemy: Optional[Enemy] = None
    stomped: Optional[Enemy] = None
    collected: Optional[Enemy] = None
    bumped: Optional[Solid] = None


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


def screen_edge(camera: float) -> Solid:
    """The screen's left edge as a wall (world pixels): Mario cannot walk
    back past it, the camera never scrolling back."""
    return Solid(camera - 64, -SCREEN_BOTTOM, camera, SCREEN_BOTTOM, "edge", ("edge", 0))


def floor_behind(solids: Sequence[Solid], enemies: Sequence[Enemy]) -> list:
    """``solids`` with each gap in the floor that enemies standing in it
    hide (no wider than HIDDEN_GAP for each, their middles inside it, their
    feet at the floor's height) joined: in 1-1 a Goomba drawn over the ground
    split it, a pit to the chooser (and a pair of them, side by side, a pit
    two Goombas wide)."""
    grounds = sorted((s for s in solids if s.kind == "ground"), key=lambda s: s.x0)
    joined = []
    for s in grounds:
        last = joined[-1] if joined else None
        hiding = 0 if last is None else sum(
            last.x1 <= (e.box[0] + e.box[2]) / 2 <= s.x0 and -2 <= last.top - e.box[3] <= HIDDEN_FEET
            for e in enemies
        )
        if hiding and abs(last.top - s.top) <= 2 and 0 < s.x0 - last.x1 <= HIDDEN_GAP * hiding:
            joined[-1] = Solid(last.x0, last.top, s.x1, last.bottom, "ground", last.ref)
        else:
            joined.append(s)
    return [s for s in solids if s.kind != "ground"] + joined


# The compiled core (numba): the same physics and checks as plain Python
# loops would do, about a hundred times faster (a plan took 275 ms in
# Python, most of the controller's time; the controller is meant to be
# arithmetic of about a millisecond a frame).
_MOTION_FIELDS = (
    "x_speed", "x_accumulator", "x_fraction", "y_speed", "y_force", "y_fraction", "force_up",
    "force_down", "force", "origin_y", "previous_jump", "facing", "moving", "running_speed", "running_timer",
)
ENDS = ("landed", "hit", "fell", "timeout")
_LANDED, _HIT, _FELL, _TIMEOUT = range(4)


def _motion_array(motion: NESPlayerMotion) -> np.ndarray:
    return np.array([int(getattr(motion, f)) for f in _MOTION_FIELDS], np.int64)


def _solid_array(solids: Sequence[Solid]) -> np.ndarray:
    return np.array([(s.x0, s.top, s.x1, s.bottom) for s in solids], np.float64).reshape(-1, 4)


def _solid_paths(solids: Sequence[Solid], frames: int) -> np.ndarray:
    """Each solid's box every frame for ``frames`` frames from now [frames +
    1, solids, 4]: a lift at its velocity, the rest where they are."""
    now = _solid_array(solids)
    speed = np.array([(s.vx, s.vy, s.vx, s.vy) for s in solids], np.float64).reshape(-1, 4)
    return now[None] + np.arange(frames + 1, dtype=np.float64)[:, None, None] * speed[None]


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
def _blocked(x, start, width, top, feet, solids):
    """Whether a body moved from ``start`` to ``x`` goes deeper into a solid's
    side (a wall, not a ledge it stands on); one already overlapping a wall
    may move out (Mario pressed against a pipe overlaps it by a few pixels:
    the game tests walls at points inside his box; taken for blocked every
    way, he stood by 5-1's tall pipe until the time ran out)."""
    for i in range(solids.shape[0]):
        if x < solids[i, 2] and solids[i, 0] < x + width and feet > solids[i, 1] + WALL_DEPTH and top < solids[i, 3]:
            now = min(x + width, solids[i, 2]) - max(x, solids[i, 0])
            before = min(start + width, solids[i, 2]) - max(start, solids[i, 0])
            if now > before:
                return True
    return False


@njit(cache=True)
def _floor_under(x0, x1, feet, solids):
    """The top of a solid under x0..x1 that ``feet`` stand on; -1 if none."""
    for i in range(solids.shape[0]):
        if x0 < solids[i, 2] and solids[i, 0] < x1 and abs(solids[i, 1] - feet) <= 0.5:
            return solids[i, 1]
    return -1.0


@njit(cache=True)
def _paths(boxes, velocities, kinds, solids, frames):
    out = np.empty((boxes.shape[0], frames + 1, 4))
    for e in range(boxes.shape[0]):
        x0, y0, x1, y1 = boxes[e, 0], boxes[e, 1], boxes[e, 2], boxes[e, 3]
        vx, vy = velocities[e, 0], velocities[e, 1]
        out[e, 0] = (x0, y0, x1, y1)
        if kinds[e] == 3:  # flier
            for k in range(1, frames + 1):
                out[e, k] = (x0 + vx * k, y0 + vy * k, x1 + vx * k, y1 + vy * k)
            continue
        if kinds[e] == 6:  # egg
            landed = False
            for k in range(1, frames + 1):
                if landed:
                    x0, x1 = x0 + vx, x1 + vx
                else:
                    vy = min(vy + EGG_GRAVITY, EGG_FALL)
                    feet, dy = y1 + EGG_FEET, vy
                    if vy > 0:
                        for i in range(solids.shape[0]):
                            if x0 < solids[i, 2] and solids[i, 0] < x1 and feet <= solids[i, 1] <= feet + vy:
                                dy, landed = solids[i, 1] - feet, True
                                break
                    y0, y1 = y0 + dy, y1 + dy
                out[e, k] = (x0, y0, x1, y1)
            continue
        if kinds[e] == 7:  # leaper
            for k in range(1, frames + 1):
                vy += LEAPER_GRAVITY
                x0, x1, y0, y1 = x0 + vx, x1 + vx, y0 + vy, y1 + vy
                out[e, k] = (x0, y0, x1, y1)
            continue
        if kinds[e] == 4:  # thrown
            for k in range(1, frames + 1):
                vy = min(vy + THROWN_GRAVITY, THROWN_FALL)
                x0, x1, y0, y1 = x0 + vx, x1 + vx, y0 + vy, y1 + vy
                out[e, k] = (x0, y0, x1, y1)
            continue
        if kinds[e] == 2:
            # plant: its pipe is the solid under its middle whose top is
            # nearest its box (above it while only its head shows: its box,
            # 14 below its head, is still in the pipe)
            middle, pipe = (x0 + x1) / 2, -1.0
            for i in range(solids.shape[0]):
                if solids[i, 0] <= middle <= solids[i, 2] and solids[i, 1] >= y0 - PLANT_DEEP and (
                        pipe < 0 or abs(solids[i, 1] - y1) < abs(pipe - y1)):
                    pipe = solids[i, 1]
            height = y1 - y0
            top = (pipe - PLANT_TOP - height) if pipe >= 0 else y0
            bottom = pipe + 1.0 if pipe >= 0 else y0
            for k in range(1, frames + 1):
                if vy > 0:
                    y0 = min(y0 + PLANT_SPEED, max(bottom, y0))
                else:
                    y0 = max(y0 - PLANT_SPEED, min(top, y0))
                out[e, k] = (x0, y0, x1, y0 + height)
            continue
        # A walker standing on something (its feet up to FEET_BELOW under its
        # box) falls once it walks off it; one already in the air keeps its
        # own motion.
        offset = -3.0
        for i in range(solids.shape[0]):
            if (x0 < solids[i, 2] and solids[i, 0] < x1
                    and -2.0 <= solids[i, 1] - y1 <= FEET_BELOW + 2.0):
                offset = solids[i, 1] - y1
                vy = 0.0
                break
        for k in range(1, frames + 1):
            nx0, nx1 = x0 + vx, x1 + vx
            if vx != 0.0:
                for i in range(solids.shape[0]):
                    if (nx0 < solids[i, 2] and solids[i, 0] < nx1 and solids[i, 1] < y1 - WALL_DEPTH
                            and y0 < solids[i, 3]):
                        vx = -vx
                        nx0, nx1 = x0 + vx, x1 + vx
                        break
            if kinds[e] == 5 and offset >= -2.0 and vx != 0.0:
                # pacer: turned back where the floor under its middle would end
                middle = (nx0 + nx1) / 2
                if _floor_under(middle - 0.5, middle + 0.5, y1 + offset, solids) < 0:
                    vx = -vx
                    nx0, nx1 = x0 + vx, x1 + vx
            if kinds[e] == 1:  # hopper: up each time it stands, then down under its own gravity
                if offset >= -2.0 and vy >= 0.0 and _floor_under(nx0, nx1, y1 + offset, solids) >= 0:
                    vy = HOP_SPEED
                vy += HOP_GRAVITY
                if offset < -2.0:
                    offset = float(HOPPER_FEET)  # in the air: a Koopa's feet below its box
                feet = y1 + offset
                dy = vy
                if vy > 0:
                    for i in range(solids.shape[0]):
                        if nx0 < solids[i, 2] and solids[i, 0] < nx1 and feet <= solids[i, 1] <= feet + vy:
                            dy = solids[i, 1] - feet
                            vy = 0.0
                            break
                x0, x1, y0, y1 = nx0, nx1, y0 + dy, y1 + dy
                out[e, k] = (x0, y0, x1, y1)
                continue
            if offset >= -2.0 and _floor_under(nx0, nx1, y1 + offset, solids) < 0:
                vy = min(vy + ENEMY_GRAVITY, ENEMY_FALL)
                feet, landed = y1 + offset + vy, -1.0
                for i in range(solids.shape[0]):
                    if (nx0 < solids[i, 2] and solids[i, 0] < nx1
                            and y1 + offset <= solids[i, 1] <= feet):
                        landed = solids[i, 1]
                        break
                dy = vy if landed < 0 else landed - (y1 + offset)
                if landed >= 0:
                    vy = 0.0
            elif offset >= -2.0:
                dy = 0.0
            else:
                dy = vy
            x0, x1, y0, y1 = nx0, nx1, y0 + dy, y1 + dy
            out[e, k] = (x0, y0, x1, y1)
    return out


@njit(cache=True)
def _reached(x0, y0, x1, y1, m0, paths, stomped, frame, safe_after, solids, margin):
    """The enemy reaching Mario, landed at the box at ``frame`` and sliding
    on (buttons released), within ``safe_after`` frames (enemies ``margin``
    wider on every side),
    those in the bit mask ``stomped`` aside; -1 if none does, SLID_OFF if
    he slides off what he landed on with nothing below to catch him (a
    landing on the last pixels before a pit at speed: he was falling before
    anything could be chosen), or so near its end, still sliding, that no
    jump could start (_footing: on the far end of 1-1's last staircase top,
    every jump was refused and he slid into the pit)."""
    m = m0.copy()
    last = paths.shape[1] - 1
    for k in range(1, safe_after + 1):
        dx, _, _ = _advance(m, 0, False, True, y0, False)
        if _blocked(x0 + dx, x0, x1 - x0, y0, y1, solids):
            dx = 0.0  # stopped by a wall (the next step up)
            m[0] = 0
            m[1] = 0
            m[2] = 0
        x0 += dx
        x1 += dx
        supported, below = _footing(x0, x1 - x0, y1, solids, (m[0] > 0) - (m[0] < 0)), False
        for i in range(solids.shape[0]):
            if x0 < solids[i, 2] and solids[i, 0] < x1 and solids[i, 1] > y1 + STAND:
                below = True
        if not supported:
            if not below:
                return SLID_OFF
            break  # off onto something lower: no longer sliding here
        for e in range(paths.shape[0]):
            if (stomped >> e) & 1:
                continue
            b = paths[e, min(frame + k, last)]
            if x0 < b[2] + margin and b[0] - margin < x1 and y0 < b[3] + margin and b[1] - margin < y1:
                return e
    return -1


@njit(cache=True)
def _touch(x0, y0, x1, y1, items, frame, got):
    """``got``, or the first of ``items`` (paths) the body touches at
    ``frame`` if none yet."""
    if got >= 0:
        return got
    last = items.shape[1] - 1
    for k in range(items.shape[0]):
        b = items[k, min(frame + 1, last)]
        if x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1:
            return k
    return -1


@njit(cache=True)
def _footing(x, width, feet, solids, moving):
    """Whether a body at ``x``, moved TAKEOFF_KEEP the way it is ``moving``
    (the sign of its speed), still stands on a solid at its feet, or a wall
    there holds it (the next step up a staircase: taken for an end, every
    step was a dead end)."""
    x += TAKEOFF_KEEP * moving
    for i in range(solids.shape[0]):
        if x < solids[i, 2] and solids[i, 0] < x + width and (
                abs(solids[i, 1] - feet) <= STAND or solids[i, 1] < feet - WALL_DEPTH < solids[i, 3]):
            return True
    return False


@njit(cache=True)
def _rollout(m0, x, y, x1, y1, pre_run, hold, direction, solid_paths, paths, stompable, horizon,
             airborne, steer, steer_direction, safe_after, items, margin):
    """rollout on arrays (``solid_paths``: each solid's box by frame, a
    lift carrying Mario standing on it; from the air when ``airborne``, steering
    ``steer_direction`` the first ``steer`` frames): (end, frame, body x0,
    y0, x1, y1, solid, enemy, stomped, got, bumped), indexes -1 where none;
    ``stomped`` is the first enemy stomped (a bounce can stomp the next one
    too: the ones stomped are a bit mask, ``down``); ``got`` the first of
    ``items`` (paths, as enemies') Mario touched; ``bumped`` the first solid
    hit from below (his rise stops there, as the game's: he falls on and
    the landing counts, a bump next to a woken Goomba included)."""
    m = m0.copy()
    width, height = x1 - x, y1 - y
    stomped = -1
    got = -1
    bumped = -1
    down = 0
    last = paths.shape[1] - 1
    final = solid_paths.shape[0] - 1
    for frame in range(horizon):
        solids = solid_paths[min(frame, final)]
        after = solid_paths[min(frame + 1, final)]
        jump = pre_run <= frame < pre_run + hold
        moving = (m[0] > 0) - (m[0] < 0)
        if jump and frame == pre_run and not airborne and not _footing(x, width, y + height, solids, moving):
            return _FELL, frame, x, y, x + width, y + height, -1, -1, stomped, got, bumped  # off the end before it takes
        d = steer_direction if frame < steer else direction
        dx, dy, jumped = _advance(m, d, jump, not airborne, y, True)
        airborne = airborne or jumped
        nx = x + dx
        feet = y + height
        if _blocked(nx, x, width, y, feet, solids):
            nx = x  # a wall: no sideways motion, as the game stops it
            m[0] = 0
            m[1] = 0
            m[2] = 0
        got = _touch(nx, y, nx + width, y + height, items, frame, got)
        if not airborne:
            ny = y
            supported = False
            for i in range(solids.shape[0]):
                if nx < solids[i, 2] and solids[i, 0] < nx + width and abs(solids[i, 1] - feet) <= STAND:
                    supported = True
                    nx += after[i, 0] - solids[i, 0]  # (carried by what he stands on)
                    ny = y + after[i, 1] - solids[i, 1]
                    break
            if not supported:
                airborne = True  # ran off a ledge
        else:
            ny = y + dy
            if dy < 0:
                for i in range(solids.shape[0]):
                    if nx < solids[i, 2] and solids[i, 0] < nx + width and ny < solids[i, 3] <= y:
                        if bumped < 0:
                            bumped = i
                        ny = solids[i, 3]
                        m[3] = 0  # (the game's vertical contact: his rise stops)
                        m[4] = 0
                        break
            else:
                for i in range(after.shape[0]):
                    if (nx < after[i, 2] and after[i, 0] < nx + width
                            and min(feet, feet + solids[i, 1] - after[i, 1]) <= after[i, 1] <= ny + height):
                        top = after[i, 1]
                        got = _touch(nx, top - height, nx + width, top, items, frame, got)
                        threat = _reached(nx, top - height, nx + width, top, m, paths, down, frame, safe_after,
                                          after, margin)
                        if threat == SLID_OFF:
                            return _FELL, frame, nx, top - height, nx + width, top, -1, -1, stomped, got, bumped
                        if threat >= 0:
                            return _HIT, frame, nx, top - height, nx + width, top, -1, threat, stomped, got, bumped
                        return _LANDED, frame, nx, top - height, nx + width, top, i, -1, stomped, got, bumped
        got = _touch(nx, ny, nx + width, ny + height, items, frame, got)
        hit = -1
        for e in range(paths.shape[0]):
            if (down >> e) & 1:
                continue
            b = paths[e, min(frame + 1, last)]
            if not (nx < b[2] + margin and b[0] - margin < nx + width
                    and ny < b[3] + margin and b[1] - margin < ny + height):
                continue
            touching = nx < b[2] and b[0] < nx + width and ny < b[3] and b[1] < ny + height
            across = min(nx + width, b[2]) - max(nx, b[0])
            from_above = feet <= b[1]  # last frame's feet above its top
            if (touching and airborne and dy > 0 and stompable[e]
                    and from_above and across >= STOMP_OVERLAP):
                if stomped < 0:
                    stomped = e
                down |= 1 << e
                m[3] = STOMP_BOUNCE
                m[4] = 0
                m[8] = m[7]
                break
            hit = e
            break
        if hit >= 0:
            return _HIT, frame, nx, ny, nx + width, ny + height, -1, hit, stomped, got, bumped
        if ny > SCREEN_BOTTOM:
            return _FELL, frame, nx, ny, nx + width, ny + height, -1, -1, stomped, got, bumped
        x, y = nx, ny
    return _TIMEOUT, horizon, x, y, x + width, y + height, -1, -1, stomped, got, bumped


def blocked(x: float, start: float, width: float, top: float, feet: float, solids: Sequence[Solid]) -> bool:
    """Whether a body moved from ``start`` to ``x`` (``width`` wide, from
    ``top`` to ``feet``) would go deeper into a solid's side: a wall, not a
    ledge it stands on (_blocked)."""
    return bool(_blocked(float(x), float(start), float(width), float(top), float(feet), _solid_array(solids)))


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


def _outcome(raw, solids: Sequence[Solid], enemies: Sequence[Enemy], items: Sequence[Enemy] = ()) -> Outcome:
    end, frame, x0, y0, x1, y1, solid, enemy, stomped, got, bumped = raw
    return Outcome(
        ENDS[end], int(frame), (float(x0), float(y0), float(x1), float(y1)),
        solid=solids[solid] if solid >= 0 else None,
        enemy=enemies[enemy] if enemy >= 0 else None,
        stomped=enemies[stomped] if stomped >= 0 else None,
        collected=items[got] if got >= 0 else None,
        bumped=solids[bumped] if bumped >= 0 else None,
    )


def _item_paths(items: Sequence[Enemy], solids: Sequence[Solid], frames: int, dynamics) -> np.ndarray:
    """The items' predicted paths (they move as walkers do) for the compiled
    core."""
    dynamics.paths(items, solids, frames)
    return _packed(items, frames)[0]


def rollout(motion: NESPlayerMotion, body: tuple, pre_run: int, hold: int, direction: int,
            solids: Sequence[Solid], enemies: Sequence[Enemy], horizon: int = HORIZON,
            airborne: bool = False, steer: int = 0, steer_direction: int = 0,
            items: Sequence[Enemy] = (), dynamics=RULES, berth: float = ENEMY_MARGIN,
            safe_after: int = SAFE_AFTER) -> Outcome:
    """Mario from ``body`` (world pixels), standing (or in the air,
    ``airborne``), running ``pre_run`` frames then holding the jump button
    ``hold`` frames, steering ``steer_direction`` the first ``steer`` frames
    and ``direction`` after, until something ends it; enemies kept
    ``berth`` wider, a landing safe once none reaches him within
    ``safe_after`` frames."""
    paths, stompable = _packed(enemies, horizon + safe_after + 1)
    x, y, x1, y1 = (float(v) for v in body)
    raw = _rollout(_motion_array(motion), x, y, x1, y1, pre_run, hold, direction, _solid_paths(solids, horizon),
                   paths, stompable, horizon, airborne, steer, steer_direction, safe_after,
                   _item_paths(items, solids, horizon + safe_after + 1, dynamics), float(berth))
    return _outcome(raw, solids, enemies, items)


def _meets(goal: Goal, outcome: Outcome, enemies: Sequence[Enemy] = ()) -> Optional[float]:
    """The margin by which ``outcome`` meets ``goal`` (larger is better);
    None if it does not."""
    middle = (outcome.body[0] + outcome.body[2]) / 2
    if goal.kind == "bump":
        # the block hit from below, and a safe landing after, the nearer under
        # it the better (landing on past it, Mario slid to a Goomba ahead)
        block = outcome.bumped
        hit = outcome.end == "landed" and block is not None and _overlap(block.x0, block.x1, goal.x0, goal.x1) \
            and abs(block.bottom - goal.top) <= STAND + 1
        return -abs(middle - (goal.x0 + goal.x1) / 2) if hit else None
    if goal.kind == "collect":
        got = outcome.end == "landed" and outcome.collected is not None and outcome.collected.ref == goal.ref
        return 0.0 if got else None
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
    solid = outcome.solid
    if (outcome.end != "landed" or not _overlap(solid.x0, solid.x1, goal.x0, goal.x1)
            or (goal.top is not None and abs(solid.top - goal.top) > STAND + 1)):
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


def buttons_of(pre_run: int, hold: int, direction: int, frames: int, steer: int = 0,
               steer_direction: int = 0) -> list:
    """The button pressed each frame of a plan, as the rollout pressed them."""
    return [BUTTON[steer_direction if f < steer else direction, pre_run <= f < pre_run + hold]
            for f in range(frames)]


def plan_jump(goal: Optional[Goal], motion: NESPlayerMotion, body: tuple, solids: Sequence[Solid],
              enemies: Sequence[Enemy], direction: int, latest_takeoff: int = HORIZON,
              items: Sequence[Enemy] = (), dynamics=RULES, berth: float = ENEMY_MARGIN,
              safe_after: int = SAFE_AFTER) -> Optional[Plan]:
    """The plan meeting ``goal`` (None: any safe landing), robust to a seen
    position's error (ROBUST_SHIFT) if any is, with the most margin (then
    the soonest);
    None when no takeoff and hold (nor walking on) does: the goal is
    infeasible from here. ``latest_takeoff``: no plan taking off later
    (frames of running) than this; walking on without jumping is a plan that
    never takes off, allowed only when unbounded. ``dynamics``: the model
    of how the enemies and items move; ``berth`` and ``safe_after`` as in
    rollout (how carefully: the choice's care, controller.CARE)."""
    best = None
    frames = HORIZON + safe_after + 1
    dynamics.paths(enemies, solids, frames)
    paths, stompable = _packed(enemies, frames)
    item_paths = _item_paths(items, solids, frames, dynamics)
    m, packed_solids = _motion_array(motion), _solid_paths(solids, HORIZON)
    x, y, x1, y1 = (float(v) for v in body)
    tries = [(pre_run, hold) for pre_run in PRE_RUNS if pre_run <= latest_takeoff for hold in NES_JUMP_FRAMES]
    if latest_takeoff >= HORIZON:
        tries.insert(0, (HORIZON, 0))
    for pre_run, hold in tries:
        raw = _rollout(m, x, y, x1, y1, pre_run, hold, direction, packed_solids, paths, stompable, HORIZON,
                       False, 0, 0, safe_after, item_paths, float(berth))
        if raw[0] != _LANDED:  # every goal ends standing
            continue
        outcome = _outcome(raw, solids, enemies, items)
        margin = 0.0 if goal is None else _meets(goal, outcome, enemies)
        if margin is None:
            continue
        # Robust: it still meets the goal from ROBUST_SHIFT pixels further
        # back (the error of a seen position).
        back = ROBUST_SHIFT * direction
        again = _rollout(m, x - back, y, x1 - back, y1, pre_run, hold, direction, packed_solids, paths,
                         stompable, HORIZON, False, 0, 0, safe_after, item_paths, float(berth))
        robust = again[0] == _LANDED and (
            goal is None or _meets(goal, _outcome(again, solids, enemies, items), enemies) is not None)
        key = (robust, margin, -outcome.frame)
        if best is None or key > best[0]:
            best = (key, Plan(pre_run, hold, direction, outcome, margin))
    if best is None:
        return None
    plan = best[1]
    plan.buttons = buttons_of(plan.pre_run, plan.hold, plan.direction, plan.outcome.frame + 1)
    return plan


def plan_flight(goal: Optional[Goal], motion: NESPlayerMotion, body: tuple, solids: Sequence[Solid],
                enemies: Sequence[Enemy], direction: int, hold_left: int, aim: float,
                items: Sequence[Enemy] = (), dynamics=RULES, berth: float = ENEMY_MARGIN,
                safe_after: int = SAFE_AFTER) -> Optional[Plan]:
    """In the air: the jump's rest, ``hold_left`` more frames of the jump
    button steering ``direction``, if it still meets ``goal`` (None: any
    safe landing); else the correction meeting it with the most margin: the
    button released sooner (or held as long as the jump allows, while it is
    still held) and STEERS frames released or braking first. Failing all,
    the safe landing nearest ``aim`` (the planned landing's middle); with no
    safe landing, the one nearest ``aim`` that no enemy reaches within
    LAST_RESORT_SAFE frames (time to jump again on landing: better than a
    plan known to hit); None when nothing lands even so. ``berth`` and
    ``safe_after`` as in rollout."""
    frames = HORIZON + safe_after + 1
    dynamics.paths(enemies, solids, frames)
    paths, stompable = _packed(enemies, frames)
    item_paths = _item_paths(items, solids, frames, dynamics)
    m, packed_solids = _motion_array(motion), _solid_paths(solids, HORIZON)
    x, y, x1, y1 = (float(v) for v in body)
    holds = FLIGHT_HOLDS if motion.previous_jump else (0,)
    tries = [(hold_left, 0, 0)] + [(h, 0, 0) for h in holds if h != hold_left] + [
        (h, steer, d) for h in holds for steer in STEERS for d in (0, -direction)
    ]
    best, fallback = None, None
    for hold, steer, d in tries:
        raw = _rollout(m, x, y, x1, y1, 0, hold, direction, packed_solids, paths, stompable, HORIZON,
                       True, steer, d, safe_after, item_paths, float(berth))
        if raw[0] != _LANDED:
            continue
        outcome = _outcome(raw, solids, enemies, items)
        plan = Plan(0, hold, direction, outcome, 0.0,
                    buttons_of(0, hold, direction, outcome.frame + 1, steer, d))
        margin = 0.0 if goal is None else _meets(goal, outcome, enemies)
        if margin is not None:
            if (hold, steer) == (hold_left, 0):
                plan.margin = margin
                return plan  # the jump as planned still does
            if best is None or (margin, -outcome.frame) > (best.margin, -best.outcome.frame):
                plan.margin = margin
                best = plan
        near = -abs((outcome.body[0] + outcome.body[2]) / 2 - aim)
        if fallback is None or near > fallback[0]:
            fallback = (near, plan)
    if best is not None:
        return best
    if fallback is not None:
        return fallback[1]
    for hold, steer, d in tries:
        raw = _rollout(m, x, y, x1, y1, 0, hold, direction, packed_solids, paths, stompable, HORIZON,
                       True, steer, d, LAST_RESORT_SAFE, item_paths, float(berth))
        if raw[0] != _LANDED:
            continue
        outcome = _outcome(raw, solids, enemies, items)
        near = -abs((outcome.body[0] + outcome.body[2]) / 2 - aim)
        if fallback is None or near > fallback[0]:
            fallback = (near, Plan(0, hold, direction, outcome, 0.0,
                                   buttons_of(0, hold, direction, outcome.frame + 1, steer, d)))
    return fallback[1] if fallback is not None else None
