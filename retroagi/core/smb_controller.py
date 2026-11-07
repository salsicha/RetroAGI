"""The controller of the choose-and-control agent
(docs/choose-and-control-plan.md phase 1; user, 2026-11-07): it carries
out one choice at a time, every frame, from the state it is given, and
reports how the choice ended (an event) so the chooser decides again.

A choice is a verb, the thing it is about and a coarse modifier:
- "jump" to land on a still thing (``prefer``: near, middle or far), to
  stomp an enemy, or to pass it: planned by smb_planner (takeoff after some
  running, the jump-button hold) and played frame by frame, re-planned
  while Mario is still on the ground, done when he stands again;
- "run" to a level x: running there and braking so as to stop at it (the
  motion model's stopping distance), or keeping speed (``keep_speed``);
- "wait": standing until an enemy is clear of a level x (``until_clear``),
  a hit is due, or at most ``frames``.

Events: "landed", "stomped", "arrived", "waited", "infeasible", "hit",
"blocked". The state each frame (State) is the scene's solids and enemies
in world pixels, Mario's body and his motion: from vision and the
executor's estimates in play (perceived), or from game memory to test the
controller alone.
"""

from copy import copy
from dataclasses import dataclass, field
from typing import Optional

from .smb_physics import NESPlayerMotion
from .smb_planner import BUTTON, Enemy, Goal, Plan, blocked, plan_jump, solids_of

# Frames of a planned jump after which the chooser decides again even if
# Mario has not landed (a plan gone wrong).
PLAN_SLACK = 24
# Re-plan before takeoff every this many frames (the scene moves on).
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
    already observing the scene): world pixels are screen pixels plus its
    camera position; enemies are its tracks seen this frame, with their
    averaged velocities (seen for fewer than MEASURED frames: a walker's
    WALK pixels a frame toward Mario); Mario's motion is its estimate, ready
    once it has measured his speed; he stands when vision says so or when
    his feet rest still on a solid's top (vision's flag alone left him "in
    the air" in a third of the frames beside 1-1's fourth pipe). None while
    vision does not show Mario."""
    if scene.mario.box is None:
        return None
    camera = feedback.camera_position
    box = scene.mario.box
    body = (box[0] + camera, box[1], box[2] + camera, box[3])
    enemies = []
    for t in feedback.tracks.tracks:
        if t.kind == "platform" or t.missed:
            continue
        world = (t.box[0] + camera, t.box[1], t.box[2] + camera, t.box[3])
        if len(t.samples) >= MEASURED:
            vx, vy = t.velocity
        else:
            vx, vy = (-WALK if world[0] > body[0] else WALK), 0.0
        enemies.append(Enemy(world, vx, vy, ("enemies", t.identity), stompable=t.kind == "walker"))
    solids = solids_of(scene, camera)
    resting = abs(feedback.feet_velocity) < 1 and any(
        s.x0 < body[2] and body[0] < s.x1 and abs(s.top - body[3]) <= 2 for s in solids
    )
    return State(
        body=body,
        motion=copy(feedback.motion),
        grounded=(scene.mario.on_something and scene.mario.support != "air") or resting,
        solids=solids,
        enemies=enemies,
        ready=feedback.motion_ready,
    )


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

    def _plan(self, state: State) -> None:
        goal = self.choice.goal
        middle = (state.body[0] + state.body[2]) / 2
        target = (goal.x0 + goal.x1) / 2 if goal.kind == "land" else goal.x1 if goal.kind == "pass" else None
        if goal.kind == "stomp":
            enemy = next((e for e in state.enemies if e.ref == goal.ref), None)
            target = (enemy.box[0] + enemy.box[2]) / 2 if enemy else middle
        direction = 1 if target >= middle else -1
        self.plan = plan_jump(goal, state.motion, state.body, state.solids, state.enemies, direction)
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
            self._plan(state)
            if self.event:
                return 0
            plan, step = self.plan, self.frame - 1 - self.plan_frame
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
        if not choice.keep_speed and left <= stopping_distance(state.motion, direction):
            return 0  # braking: released, Mario slides to the x
        return BUTTON[direction, False]

    def _wait(self, state: State) -> int:
        choice = self.choice
        middle = (state.body[0] + state.body[2]) / 2
        coming = [e for e in state.enemies if abs((e.box[0] + e.box[2]) / 2 - middle) < TOO_CLOSE]
        if coming or self.frame >= choice.frames:
            self.event = "waited"
            return 0
        if choice.until_clear is not None:
            ref, x = choice.until_clear
            enemy = next((e for e in state.enemies if e.ref == ref), None)
            if enemy is None or (enemy.box[2] < x if enemy.vx <= 0 else enemy.box[0] > x):
                self.event = "waited"
        return 0
