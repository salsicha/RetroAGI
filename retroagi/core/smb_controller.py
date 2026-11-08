"""The controller of the choose-and-control agent
(docs/choose-and-control-plan.md phase 1; user, 2026-11-07): it carries
out one choice at a time, every frame, from the state it is given, and
reports how the choice ended (an event) so the chooser decides again.

A choice is a verb, the thing it is about and a coarse modifier:
- "jump" to land on a still thing (``prefer``: near, middle or far), to
  stomp an enemy, or to pass it: planned by smb_planner (takeoff after some
  running, the jump-button hold) and played frame by frame, re-planned
  while Mario is still on the ground, checked every REPLAN_EVERY frames in
  the air and corrected when its landing no longer meets the goal
  (smb_planner.plan_flight), done when he stands again;
- "run" to a level x: running there and braking so as to stop at it (the
  motion model's stopping distance), or keeping speed (``keep_speed``);
- "wait": standing until an enemy is clear of a level x (``until_clear``),
  a hit is due, or at most ``frames``.

Events: "landed", "stomped", "arrived", "waited", "infeasible", "hit",
"blocked", "threat" (a run meeting an enemy soon: meets_soon). The state
each frame (State) is the scene's solids and enemies in world pixels,
Mario's body and his motion: from vision and the
executor's estimates in play (perceived), or from game memory to test the
controller alone.
"""

from copy import copy
from dataclasses import dataclass, field
from typing import Optional

from .smb_physics import NESPlayerMotion
from .smb_planner import (
    BUTTON,
    HORIZON,
    Enemy,
    Goal,
    Plan,
    blocked,
    floor_behind,
    plan_flight,
    plan_jump,
    screen_edge,
    solids_of,
)

# Frames of a planned jump after which the chooser decides again even if
# Mario has not landed (a plan gone wrong).
PLAN_SLACK = 24
# Re-plan before takeoff, and check the flight after it, every this many
# frames (the scene moves on).
REPLAN_EVERY = 4
# A run ends this close to its x (pixels).
ARRIVE = 2
# A wait ends when a coming enemy is this close (pixels) whatever else.
TOO_CLOSE = 24
# Frames a run may last before the chooser decides again.
RUN_FRAMES = 240
# Perceived enemies seen for fewer frames than this are taken to walk as
# walkers do, WALK pixels a frame toward Mario.
MEASURED = 4
WALK = 0.5
# Mario's body as the game collides it, inside the sprite vision draws: this
# wide about the drawn box's middle (pixels), its top this far below the
# drawn top. The drawn 16-pixel width stood him on a 16-pixel step past its
# edge, in the planner, while he slid off it.
BODY_WIDTH = 10
HEAD_ROOM = 4
# Enemies seen starting in this many columns at the screen's left edge are
# left out: there vision read cut-off sprites as enemies walking toward
# Mario, and he jumped back over a pit at them (the game hides the first 8
# columns; a real enemy there is half gone and leaving).
LEFT_COLUMNS = 16
# Enemies seen narrower or shorter than this (pixels) are left out unless
# the screen's right edge cuts them (from RIGHT_EDGE on): fragments 2 to 8
# pixels wide in 1-1's second pit were read as enemies walking toward Mario
# (a Goomba is 16 by 16, 8 tall when stomped flat).
MIN_ENEMY = 6
RIGHT_EDGE = 240
# An enemy ahead at Mario's level (feet within LEVEL pixels) meeting him
# within this many frames at their closing speed ends a run ("threat"): a
# jump's length, time to choose and make one. A run of 96 pixels blind to
# enemies ran Mario at full speed into a pair of Goombas too close to jump.
THREAT_FRAMES = 48
LEVEL = 24
# A step down a staircase (pixels: a block).
STEP = 16


@dataclass
class Choice:
    verb: str  # "run", "jump", "wait"
    goal: Optional[Goal] = None  # jump
    x: float = 0.0  # run: the level x to reach
    keep_speed: bool = False
    until_clear: Optional[tuple] = None  # wait: (enemy ref, level x)
    frames: int = 120  # wait: at most


@dataclass
class State:
    """What the controller reads each frame (world pixels)."""

    body: tuple
    motion: NESPlayerMotion
    grounded: bool
    solids: list = field(default_factory=list)
    enemies: list = field(default_factory=list)
    # Whether Mario's motion is known well enough to plan from (perceived:
    # the executor has measured his speed; always, from game memory).
    ready: bool = True


def perceived(feedback, scene) -> Optional[State]:
    """The controller's state from what vision shows (``scene``) and the
    executor's estimates (``feedback``, smb_spatial_feedback.SpatialFeedback,
    already observing the scene), in world pixels (screen pixels plus its
    camera position):
    - Mario's body is his collision box inside the drawn one (BODY_WIDTH,
      HEAD_ROOM); his motion is the estimate, ready once it has measured his
      speed; he stands when vision says so or when his feet rest still on a
      solid's top (vision's flag alone left him "in the air" in a third of
      the frames beside 1-1's fourth pipe);
    - the solids are vision's, none inside Mario's drawn box, floor gaps
      hidden by enemies joined, and the screen's left edge, a wall;
    - the enemies are its tracks seen this frame, none starting in the
      LEFT_COLUMNS and none smaller than MIN_ENEMY away from the right edge,
      with their averaged velocities (seen for fewer than
      MEASURED frames: a walker's WALK pixels a frame toward Mario).
    None while vision does not show Mario."""
    if scene.mario.box is None:
        return None
    camera = feedback.camera_position
    box = scene.mario.box
    middle = (box[0] + box[2]) / 2 + camera
    body = (middle - BODY_WIDTH / 2, box[1] + HEAD_ROOM, middle + BODY_WIDTH / 2, box[3])
    enemies = []
    for t in feedback.tracks.tracks:
        if t.kind == "platform" or t.missed or t.box[0] < LEFT_COLUMNS:
            continue
        small = min(t.box[2] - t.box[0], t.box[3] - t.box[1]) < MIN_ENEMY
        if small and t.box[2] < RIGHT_EDGE:
            continue
        world = (t.box[0] + camera, t.box[1], t.box[2] + camera, t.box[3])
        if len(t.samples) >= MEASURED:
            vx, vy = t.velocity
        else:
            vx, vy = (-WALK if world[0] > body[0] else WALK), 0.0
        enemies.append(Enemy(world, vx, vy, ("enemies", t.identity), stompable=t.kind == "walker"))
    # Nothing solid lies inside Mario's own drawn box: at the screen's left
    # edge, parts of his cut-off sprite read as a 3-pixel pipe.
    drawn = (box[0] + camera, box[1], box[2] + camera, box[3])
    solids = [
        s for s in floor_behind(solids_of(scene, camera), enemies)
        if not (drawn[0] - 1 <= s.x0 and s.x1 <= drawn[2] + 1 and s.top < drawn[3] and drawn[1] < s.bottom)
    ] + [screen_edge(camera)]
    resting = abs(feedback.feet_velocity) < 1 and any(
        s.x0 < body[2] and body[0] < s.x1 and abs(s.top - body[3]) <= 2 for s in solids
    )
    motion = copy(feedback.motion)
    motion.origin_y += HEAD_ROOM  # the jump's origin, measured at the drawn top
    return State(
        body=body,
        motion=motion,
        grounded=(scene.mario.on_something and scene.mario.support != "air") or resting,
        solids=solids,
        enemies=enemies,
        ready=feedback.motion_ready,
    )


def on_the_way(state: State, enemy: Enemy) -> bool:
    """Whether ``enemy`` is on Mario's way: its feet within LEVEL of his, or
    of a floor he reaches walking toward it down steps (each next one
    adjacent and at most STEP lower: Goombas at the foot of a staircase,
    ignored while he ran down it, were met at full speed)."""
    body, feet = state.body, state.body[3]
    if abs(enemy.box[3] - feet) < LEVEL:
        return True
    if enemy.box[3] < feet:
        return False
    toward = 1 if (enemy.box[0] + enemy.box[2]) / 2 >= (body[0] + body[2]) / 2 else -1
    floor = next((s for s in state.solids if s.x0 < body[2] and body[0] < s.x1 and abs(s.top - feet) <= 2), None)
    while floor is not None:
        if abs(enemy.box[3] - floor.top) < LEVEL:
            return True
        edge = floor.x1 + 2 if toward > 0 else floor.x0 - 2
        floor = next((s for s in state.solids if s.x0 <= edge < s.x1 and 0 < s.top - floor.top <= STEP
                      and s.kind not in ("block", "edge")), None)
    return False


def meets_soon(state: State, enemy: Enemy, frames: int = THREAT_FRAMES) -> bool:
    """Whether ``enemy``, on Mario's way (on_the_way), meets him within
    ``frames`` at their present closing speed."""
    body = state.body
    if not on_the_way(state, enemy):
        return False
    speed = state.motion.x_speed / 16
    if (enemy.box[0] + enemy.box[2]) / 2 >= (body[0] + body[2]) / 2:
        gap, closing = enemy.box[0] - body[2], speed - enemy.vx
    else:
        gap, closing = body[0] - enemy.box[2], enemy.vx - speed
    return closing > 0 and max(gap, 0.0) <= closing * frames


def stopping_distance(motion: NESPlayerMotion, direction: int) -> float:
    """How far Mario slides, ``direction`` released now (pixels)."""
    m = copy(motion)
    total = 0.0
    for _ in range(64):
        dx, _, _ = m.advance(direction=0, jump=False, grounded=True, y=0, run=False)
        total += dx
        if m.x_speed == 0:
            break
    return abs(total)


class Controller:
    """One choice at a time; ``press(state)`` gives this frame's button and
    sets ``event`` when the choice has ended."""

    def __init__(self):
        self.choice: Optional[Choice] = None
        self.plan: Optional[Plan] = None
        self.frame = 0
        self.event: Optional[str] = None

    def start(self, choice: Choice, state: State) -> None:
        self.choice, self.plan, self.frame, self.event = choice, None, 0, None
        if choice.verb == "jump":
            self._plan(state)

    def _plan(self, state: State, latest_takeoff: int = HORIZON) -> None:
        goal = self.choice.goal
        middle = (state.body[0] + state.body[2]) / 2
        if goal.kind == "land":
            direction = 1 if (goal.x0 + goal.x1) / 2 >= middle else -1
        elif goal.kind == "pass":
            direction = goal.side
        else:
            enemy = next((e for e in state.enemies if e.ref == goal.ref), None)
            target = (enemy.box[0] + enemy.box[2]) / 2 if enemy else middle
            direction = 1 if target >= middle else -1
        self.plan = plan_jump(goal, state.motion, state.body, state.solids, state.enemies, direction,
                              latest_takeoff=latest_takeoff)
        # The plan's first button is pressed this frame (or, from start, the
        # first press).
        self.plan_frame = max(0, self.frame - 1)
        if self.plan is None:
            self.event = "infeasible"

    def press(self, state: State) -> int:
        choice = self.choice
        if choice is None or self.event is not None:
            return 0
        self.frame += 1
        if choice.verb == "jump":
            return self._jump(state)
        if choice.verb == "run":
            return self._run(state)
        return self._wait(state)

    def _jump(self, state: State) -> int:
        plan = self.plan
        step = self.frame - 1 - self.plan_frame
        taken_off = step >= plan.pre_run
        if not taken_off and state.grounded and step % REPLAN_EVERY == 0 and step:
            # Re-planned on the new state, taking off no later than the plan
            # it replaces (each re-plan postponing takeoff by its own run-up
            # ran Mario off the edge of 1-1's second pit).
            self._plan(state, latest_takeoff=plan.pre_run - step)
            if self.event:
                return 0
            plan, step = self.plan, self.frame - 1 - self.plan_frame
        if taken_off and not state.grounded and (step - plan.pre_run) % REPLAN_EVERY == 0:
            plan, step = self._check_flight(state, plan, step)
        if taken_off and state.grounded and step > plan.pre_run + 1:
            # Landed (or stomped and landed): the choice is done, whatever
            # buttons the plan had left.
            self.event = "stomped" if plan.outcome.stomped is not None else "landed"
            return 0
        if step >= len(plan.buttons):
            if state.grounded:
                self.event = "stomped" if plan.outcome.stomped is not None else "landed"
                return 0
            if step >= len(plan.buttons) + PLAN_SLACK:
                self.event = "landed"
                return 0
            return BUTTON[plan.direction, False]
        return plan.buttons[step]

    def _check_flight(self, state: State, plan: Plan, step: int) -> tuple:
        """(plan, step) after checking the rest of the flight against the
        state: the same when its landing still meets the goal, else the
        correction (an enemy woken after takeoff waits where Mario comes
        down). A goal whose enemy is gone (stomped, or off the screen)
        becomes any safe landing."""
        goal = self.choice.goal
        if goal.kind != "land" and not any(e.ref == goal.ref for e in state.enemies):
            goal = None
        rest = plan.buttons[step:]
        hold_left = next((k for k, b in enumerate(rest) if b not in (2, 4, 5)), len(rest))
        aim = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
        flight = plan_flight(goal, state.motion, state.body, state.solids, state.enemies, plan.direction,
                             hold_left, aim)
        if flight is None or flight.buttons == rest[:len(flight.buttons)]:
            return plan, step
        if flight.outcome.stomped is None:
            flight.outcome.stomped = plan.outcome.stomped  # a stomp already made counts
        self.plan, self.plan_frame = flight, self.frame - 1
        return flight, 0

    def _run(self, state: State) -> int:
        choice = self.choice
        middle = (state.body[0] + state.body[2]) / 2
        direction = 1 if choice.x >= middle else -1
        left = (choice.x - middle) * direction
        if left <= ARRIVE or self.frame > RUN_FRAMES:
            self.event = "arrived"
            return BUTTON[direction, False] if choice.keep_speed else 0
        x0, top, x1, feet = state.body
        if blocked(x0 + direction * 2, x1 - x0, top, feet, state.solids):
            self.event = "blocked"
            return 0
        ahead = [e for e in state.enemies if ((e.box[0] + e.box[2]) / 2 - middle) * direction > 0]
        if any(meets_soon(state, e) for e in ahead):
            self.event = "threat"
            return BUTTON[direction, False] if choice.keep_speed else 0
        if not choice.keep_speed and left <= stopping_distance(state.motion, direction):
            return 0  # braking: released, Mario slides to the x
        return BUTTON[direction, False]

    def _wait(self, state: State) -> int:
        choice = self.choice
        middle = (state.body[0] + state.body[2]) / 2
        coming = [e for e in state.enemies if abs((e.box[0] + e.box[2]) / 2 - middle) < TOO_CLOSE]
        if coming or self.frame > choice.frames:  # (frames waited, the frame it ends not among them)
            self.event = "waited"
            return 0
        if choice.until_clear is not None:
            ref, x = choice.until_clear
            enemy = next((e for e in state.enemies if e.ref == ref), None)
            if enemy is None or (enemy.box[2] < x if enemy.vx <= 0 else enemy.box[0] > x):
                self.event = "waited"
        return 0
