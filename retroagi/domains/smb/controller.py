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
  motion model's stopping distance), or keeping speed (``keep_speed``),
  never into a pit (it ends at the edge: pit_ahead);
- "wait": standing, braking to a stop (_brake), until an enemy is clear of
  a level x (``until_clear``), a hit is due, or at most ``frames``.

Events: "landed", "stomped", "arrived", "waited", "infeasible", "hit",
"blocked", "threat" (a run meeting an enemy soon: meets_soon), "edge" (a
run stopped before a pit). The state
each frame (State) is the scene's solids and enemies in world pixels,
Mario's body and his motion: from vision and the
executor's estimates in play (perceived), or from game memory to test the
controller alone.
"""

from copy import copy, deepcopy
from dataclasses import dataclass, field, replace
from typing import Optional

from .physics import NESPlayerMotion
from .planner import (
    BLOCK_KINDS,
    BUTTON,
    ENEMY_MARGIN,
    HORIZON,
    RULES,
    SAFE_AFTER,
    WALL_DEPTH,
    Enemy,
    Goal,
    Plan,
    blocked,
    plan_flight,
    plan_jump,
    rollout,
    standing_spring,
)
from .planner import SPRING_HORIZON
from .physics import SPRING_THROW

# Frames of a planned jump after which the chooser decides again even if
# Mario has not landed (a plan gone wrong).
PLAN_SLACK = 24
# A jump has landed once Mario has stood this many frames in a row (vision
# said "standing" for one frame mid-jump over 1-1's second pit; the jump
# ended, its buttons released, and he fell in).
LANDING_FRAMES = 2
# Re-plan before takeoff, and check the flight after it, every this many
# frames (the scene moves on).
REPLAN_EVERY = 4
# A run ends this close to its x (pixels).
ARRIVE = 2
# A wait brakes while Mario's speed is at least this (sixteenths of a pixel a
# frame: below it the game's skid turns him round); a run ends this many
# pixels before Mario would slide over a pit (pit_ahead).
BRAKE_SPEED, EDGE_KEEP = 11, 4
# (a solid's top this far above his feet still counts as under them)
STAND_SLACK = 2
# A gap in what he runs on this wide or less (one tile) is crossed walking or
# running when its far side is at most a step (WALL_DEPTH) above his feet;
# met at a higher side he falls in, if he is narrower than it (the game's,
# scripts/dev/gap_rule.py).
GAP_CROSS = 16
# A wait ends when a coming enemy is this close (pixels) whatever else.
TOO_CLOSE = 24
# Frames a run may last before the chooser decides again.
RUN_FRAMES = 240
# An enemy ahead at Mario's level (feet within LEVEL pixels) meeting him
# within this many frames at their closing speed ends a run ("threat"): a
# jump's length, time to choose and make one. A run of 96 pixels blind to
# enemies ran Mario at full speed into a pair of Goombas too close to jump.
THREAT_FRAMES = 48
LEVEL = 24


@dataclass(frozen=True)
class Care:
    """How carefully a choice is carried out: enemies kept ``berth`` pixels
    wider on every side when a jump's path is checked, a landing safe only
    if no enemy reaches Mario within ``clear`` frames after it, and a run
    ended by an enemy meeting Mario within ``threat`` frames."""

    name: str
    berth: float
    clear: int
    threat: int


# The ways of carrying a choice out, each a variant of every jump and run
# (choices.py): the outcome network predicts what each leads to and the
# chooser picks among them as among any choices (user, 2026-11-12: "make
# them part of each choice"). "usual" is the controller's hand-set way
# before then.
CARE = (Care("bold", 0.0, 6, 32), Care("usual", ENEMY_MARGIN, SAFE_AFTER, THREAT_FRAMES), Care("careful", 4.0, 20, 64))
USUAL = 1


@dataclass
class Choice:
    verb: str  # "run", "jump", "wait"
    goal: Optional[Goal] = None  # jump
    x: float = 0.0  # run: the level x to reach
    keep_speed: bool = False
    until_clear: Optional[tuple] = None  # wait: (enemy ref, level x)
    frames: int = 120  # wait: at most
    care: int = USUAL  # jump, run: how carefully (CARE)


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
    # Things to collect (a mushroom out of a question block), as enemies'
    # boxes and velocities.
    items: list = field(default_factory=list)
    # Lava seen in pits (Solids of kind "lava", which nothing stands on: only
    # seen, never planned on; from Block SMB's record, not yet from vision).
    lava: list = field(default_factory=list)

    @property
    def decidable(self) -> bool:
        """Whether a choice can be made now: Mario stands and his motion is
        known (a plan made in the air would start from a standing jump)."""
        return self.grounded and self.ready


def on_the_way(state: State, enemy: Enemy) -> bool:
    """Whether ``enemy`` is on Mario's way: its feet within LEVEL of his, or
    of a floor he reaches walking toward it, down steps or off an edge onto
    anything lower (Goombas at the foot of a staircase, ignored while he ran
    down it, were met at full speed; a Goomba on the floor beyond a pipe,
    ignored as he ran off it, was landed on)."""
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
        floor = next((s for s in state.solids if s.x0 <= edge < s.x1 and s.top > floor.top
                      and s.kind not in (*BLOCK_KINDS, "edge")), None)
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


def falls_on_run(state: State, direction: int, care: "Care", dynamics=RULES) -> bool:
    """Whether something above Mario's level (a hammer coming down, a
    Spiny's egg) meets him, running on ``direction``, within the care's
    threat frames, each along its own foreseen path (``dynamics``) and kept
    the care's berth wider. (Run back under a hammer falling just behind
    him: on_the_way looks at his level only.)"""
    frames = care.threat
    x0, top, x1, feet = state.body
    reach = (4.0 + 4.5) * frames  # (his top speed and a thrown thing's)
    above = [copy(e) for e in state.enemies
             if e.box[3] <= feet - LEVEL and abs((e.box[0] + e.box[2]) / 2 - (x0 + x1) / 2) <= reach]
    if not above:
        return False
    dynamics.paths(above, state.solids, frames)
    motion, dx = copy(state.motion), 0.0
    for k in range(1, frames + 1):
        step, _, _ = motion.advance(direction=direction, jump=False, grounded=True, y=top, run=direction > 0)
        dx += step
        for e in above:
            b = e.at(k)
            if (x0 + dx < b[2] + care.berth and b[0] - care.berth < x1 + dx
                    and top < b[3] + care.berth and b[1] - care.berth < feet):
                return True
    return False


def _thrown_clear(state: State, care: "Care", dynamics=RULES) -> bool:
    """Whether a wait is safe as far as a springboard goes: standing on
    one, waiting is no wait (it presses him down and throws him up), and the
    throw, braked until it comes (_brake) and released after, must come down
    clear of every enemy (along its foreseen path, the care's berth) and of
    every pit, as a jump's landing must. (Waits on springboards, rated safe,
    threw Mario into a Bullet Bill, a pit and Lakitu.) True off one."""
    spring = standing_spring(state.body, state.solids)
    if spring is None:
        return True
    speed = state.motion.x_speed
    brake = (-1 if speed > 0 else 1) if abs(speed) >= BRAKE_SPEED else 0
    squash = spring.spring if spring.spring is not None and spring.spring >= 0 else -1
    enemies = [copy(e) for e in state.enemies]
    dynamics.paths(enemies, state.solids, SPRING_HORIZON + care.clear + 1)
    outcome = rollout(state.motion, state.body, 0, 0, 0, state.solids, enemies, horizon=SPRING_HORIZON,
                      steer=SPRING_THROW - squash, steer_direction=brake, berth=care.berth, safe_after=care.clear)
    return outcome.end not in ("hit", "fell")


def _brake(state: State) -> int:
    """Waiting Mario stops where he is: the button against his motion while
    it is BRAKE_SPEED or more (slower, the game turns him round instead),
    else none. (Released, he slid off a 5-1 pillar into Goombas.)"""
    speed = state.motion.x_speed
    if state.grounded and abs(speed) >= BRAKE_SPEED:
        return BUTTON[-1 if speed > 0 else 1, False]
    return 0


def pit_ahead(state: State, direction: int) -> bool:
    """Whether Mario, released now, would slide (EDGE_KEEP more) over a pit:
    on the way his front goes, a stretch with no solid reaching down to his
    feet (the floor, a lower one, or a pipe or step: a wall stops him; taken
    for a pit, every run at 1-1's first pipe stopped short of it) that he
    would fall into: as wide as he is or wider, and not a gap of GAP_CROSS
    with its far side at most a step above his feet. (Asked only where he
    would stop, a run went on at a step up beyond a one-tile gap, taking the
    step for a wall that stops him; he fell into the gap against it.) A
    lower one beyond its gap holds him only if, carried on at his speed as
    he falls, he comes down over it."""
    x0, _, x1, feet = state.body
    start = x1 if direction > 0 else x0
    end = (start + direction * (stopping_distance(state.motion, direction) + EDGE_KEEP)) * direction
    spans = [(min(s.x0 * direction, s.x1 * direction), max(s.x0 * direction, s.x1 * direction), s.top)
             for s in state.solids if s.bottom >= feet - STAND_SLACK and s.kind != "edge"]
    at = start * direction  # (along the run: how far it is known to hold him)
    while True:
        on = [a1 for a0, a1, _ in spans if a0 <= at < a1]
        while on:  # (on along what holds him, one solid to the next)
            at = max(on)
            on = [a1 for a0, a1, _ in spans if a0 <= at < a1]
        if at >= end:
            return False
        beyond = [(a0, top) for a0, _, top in spans if a0 > at]
        if not beyond:
            return True
        far, top = min(beyond)
        if far > end:
            return True  # (where he would stop, nothing under him)
        gap = far - at
        if top > feet + STAND_SLACK:
            # A lower one: he slides off and falls to it, carried on at his
            # speed; caught only if he comes down over it (at full speed off
            # a ledge he flew past a short platform below into the pit).
            fall = _carried(state.motion, direction, at + (x1 - x0) - start, top - feet)
            if fall is None:
                return False  # (he stops before he is off)
            carried, landed = fall
            back = at + carried
            lower = [a1 for a0, a1, t in spans if t == top and a0 == far]
            if back + (x1 - x0) <= far or back >= max(lower):
                return True
            # (down on it at the speed he fell with: his slide from there)
            end = back + (x1 - x0) + stopping_distance(landed, direction) + EDGE_KEEP
            feet = top
            spans = [span for span in spans if span[2] >= feet - STAND_SLACK]
            at = max(back, far)
            continue
        if gap >= x1 - x0 and (gap > GAP_CROSS or top < feet - WALL_DEPTH):
            return True
        at = far


def _pressed(state: State, direction: int) -> State:
    """The state a frame on, Mario running ``direction`` on the ground."""
    motion = copy(state.motion)
    dx, _, _ = motion.advance(direction=direction, jump=False, grounded=True, y=state.body[1], run=direction > 0)
    x0, top, x1, feet = state.body
    return replace(state, body=(x0 + dx, top, x1 + dx, feet), motion=motion)


def _carried(motion: NESPlayerMotion, direction: int, slide: float, drop: float) -> Optional[tuple]:
    """How far Mario, released now, has gone on when he has come down
    ``drop`` pixels after sliding ``slide`` pixels off an edge (pixels, from
    where he left it), and his motion then; None if he stops before."""
    m = copy(motion)
    slid = 0.0
    while slid < slide:
        dx, _, _ = m.advance(direction=0, jump=False, grounded=True, y=0, run=False)
        slid += abs(dx)
        if m.x_speed == 0:
            return None
    m.y_speed = m.y_force = 0
    m.force = m.force_down
    carried, fallen, y = 0.0, 0.0, 0
    for _ in range(240):
        dx, dy, _ = m.advance(direction=0, jump=False, grounded=False, y=y, run=False)
        carried += abs(dx)
        fallen += dy
        y += dy
        if fallen >= drop:
            break
    m.y_speed = m.y_force = 0
    return carried, m


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
    sets ``event`` when the choice has ended. ``dynamics``: its model of how
    the other things move (planner.RULES, written by hand, by default)."""

    def __init__(self, dynamics=RULES):
        self.dynamics = dynamics
        self.choice: Optional[Choice] = None
        self.plan: Optional[Plan] = None
        self.frame = 0
        self.event: Optional[str] = None

    def observe(self, state: State) -> None:
        self.dynamics.observe(state.enemies + state.items, state.solids)

    def another(self) -> "Controller":
        """A new controller whose dynamics model knows what this one's has
        learned (a copy: what either learns from here on is its own)."""
        return Controller(deepcopy(self.dynamics))

    def start(self, choice: Choice, state: State) -> None:
        self.choice, self.plan, self.frame, self.event = choice, None, 0, None
        self.stood = 0  # frames in a row Mario has stood
        self.flown = False  # Mario seen in the air since the jump took off
        self.touched = False  # and seen on something since
        if choice.verb == "jump":
            self._plan(state)
        elif choice.verb == "wait" and state.grounded and not _thrown_clear(state, CARE[choice.care], self.dynamics):
            self.event = "infeasible"

    def _plan(self, state: State, latest_takeoff: int = HORIZON) -> None:
        goal = self.choice.goal
        middle = (state.body[0] + state.body[2]) / 2
        if goal.kind in ("land", "bump"):
            direction = 1 if (goal.x0 + goal.x1) / 2 >= middle else -1
        elif goal.kind == "pass":
            direction = goal.side
        else:
            thing = next((e for e in state.enemies + state.items if e.ref == goal.ref), None)
            target = (thing.box[0] + thing.box[2]) / 2 if thing else middle
            direction = 1 if target >= middle else -1
        care = CARE[self.choice.care]
        self.plan = plan_jump(goal, state.motion, state.body, state.solids, state.enemies, direction,
                              latest_takeoff=latest_takeoff, items=state.items, dynamics=self.dynamics,
                              berth=care.berth, safe_after=care.clear)
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
                # The goal cannot be met any more: rather than run on at what
                # is ahead, any safe landing taking off as soon (in 5-1 the
                # pass over two Koopas, vision's one, failed in the run-up,
                # and Mario coasted into them).
                care = CARE[self.choice.care]
                fallback = plan_jump(None, state.motion, state.body, state.solids, state.enemies,
                                     plan.direction, latest_takeoff=plan.pre_run - step, items=state.items,
                                     dynamics=self.dynamics, berth=care.berth, safe_after=care.clear)
                if fallback is None:
                    return 0
                self.plan, self.plan_frame, self.event = fallback, max(0, self.frame - 1), None
            plan, step = self.plan, self.frame - 1 - self.plan_frame
        if taken_off and not state.grounded and (step - plan.pre_run) % REPLAN_EVERY == 0:
            plan, step = self._check_flight(state, plan, step)
        self.stood = self.stood + 1 if state.grounded else 0
        # A jump lands only once Mario has been seen in the air: vision still
        # called him standing in the first frames after takeoff, the run-up's
        # standing frames counted on, and a jump across a 5-1 pit "landed"
        # at its top (a hop never seen in the air ends by PLAN_SLACK).
        self.flown = self.flown or (taken_off and not state.grounded)
        self.touched = self.touched or (self.flown and state.grounded)
        if taken_off and self.flown and self.stood >= LANDING_FRAMES and step > plan.pre_run + 1:
            # Landed (or stomped and landed): the choice is done, whatever
            # buttons the plan had left.
            self.event = "stomped" if plan.outcome.stomped is not None else "landed"
            return 0
        if step >= len(plan.buttons):
            if self.flown and self.stood >= LANDING_FRAMES:
                self.event = "stomped" if plan.outcome.stomped is not None else "landed"
                return 0
            if step >= len(plan.buttons) + PLAN_SLACK:
                self.event = "landed"
                return 0
            if self.touched:
                # Down on something, the plan's buttons done: released, as
                # the planner foresees him sliding on (planner._reached).
                # Steering on, Mario landed on a pipe's last pixel slid off
                # it past the ground beside it and into a pit.
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
        if goal.kind == "bump" or (goal.kind != "land" and not any(
                e.ref == goal.ref for e in state.enemies + state.items)):
            goal = None  # (after a bump, or its thing gone: any safe landing)
        rest = plan.buttons[step:]
        hold_left = next((k for k, b in enumerate(rest) if b not in (2, 4, 5)), len(rest))
        aim = (plan.outcome.body[0] + plan.outcome.body[2]) / 2
        care = CARE[self.choice.care]
        flight = plan_flight(goal, state.motion, state.body, state.solids, state.enemies, plan.direction,
                             hold_left, aim, items=state.items, dynamics=self.dynamics, berth=care.berth,
                             safe_after=care.clear)
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
        if blocked(x0 + direction * 2, x0, x1 - x0, top, feet, state.solids):
            self.event = "blocked"
            return 0
        ahead = [e for e in state.enemies if ((e.box[0] + e.box[2]) / 2 - middle) * direction > 0]
        care = CARE[choice.care]
        if (any(meets_soon(state, e, care.threat) for e in ahead)
                or falls_on_run(state, direction, care, self.dynamics)):
            self.event = "threat"
            return BUTTON[direction, False] if choice.keep_speed else 0
        if state.grounded and (pit_ahead(state, direction) or pit_ahead(_pressed(state, direction), direction)):
            # A run never takes Mario into a pit: it ends where he can still
            # stop (in 5-1 a run back slid him into the pit behind), this
            # frame's press counted: just turned back by the last run's end
            # at an edge, Mario skids and would stop short, but the next run
            # chosen pressed on, turning him round again, a frame each time,
            # and he went over.
            self.event = "edge"
            return BUTTON[-direction, False]
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
        return _brake(state)
