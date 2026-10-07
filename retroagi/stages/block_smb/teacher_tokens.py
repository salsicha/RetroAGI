"""Teachers for the layered agent in Block SMB. Training only.

At each decision point of a training episode these say what each layer
should emit: the strategy, the tactic token and the action plan. They read
the simulator's own state, which is allowed only for teaching. A policy never
receives anything from here, except, while the action layer is being trained,
the explicit tactic token.

- strategy: the one the layout is played for (a strategy family names it;
  every other layout is a speed run), toward the side the goal was on when
  the episode began;
- tactic: one of tokens.TACTICS, the way Mario is going here. It is worked
  out in two steps:
  1. the layout's schedule stance (tactic_schedule): what its current
     segment says, changed inside moving-platform, plant and monster
     segments by their teachers' rules (hold the area while waiting for or
     riding a moving platform, while waiting for a plant to go back into its
     pipe, or while a monster is far enough; retreat while backing away from
     it; advance otherwise);
  2. the next move under that stance. Holding the area is hold_ground: stay
     at the current spot relative to the supporting platform. A scheduled
     retreat is retreating. Under advance or alternate route the move aims
     at the segment's next route platform, else the nearest obstacle toward
     the goal: a platform higher than Mario's feet is climbed, a lower one
     (or a goal on lower ground) descended to, and anything else (walking,
     jumping a gap, getting past or stomping an enemy, the moving platform's
     approach, boarding and leaving) is advancing.
  Forward is right, the level's way: a move to the left is retreat,
  climb_backward or descend_backward, never advance. A move lasts until
  Mario lands: in the air he keeps the tactic he left the ground with;
- action: the first segment of the coached route from the current state
  (policy_recovery.coached_suffix), as an action and a frame count on the
  executor's menu; for a jump, also every certified hold (safe_jump_holds).
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from retroagi.core.actions import SMB_JUMP_ACTIONS, SMBAction
from retroagi.core.smb_executor import FRAME_COUNTS, HOLD_GROUND, ActionPlan
from retroagi.core.tokens import (
    DEFAULT_STRATEGY,
    SkillToken,
    StrategyToken,
    TacticToken,
    tactic_token,
)

from . import tactic_schedule
from .monte_carlo import block_smb_monte_carlo_metadata

# A platform this many pixels above (below) Mario's feet is climbed (descended to).
STEP = 2


@dataclass
class TeacherState:
    """What a teacher carries through one episode (training only)."""

    family: str
    direction: int = 1  # the side the goal was on when the episode began
    strategy: str = DEFAULT_STRATEGY.kind  # the strategy the layout is played for
    bridge_opening: bool = True
    phase: str = ""
    phase_step: int = -1  # the frame ``phase`` was worked out for
    observer: object = None  # EnemyObservationHistory, fed every frame by observe_frame
    history: object = None  # its latest features
    notes: dict = field(default_factory=dict)

    def observe_frame(self, env) -> None:
        """Call once per frame: the plant and enemy teachers need the history."""
        from retroagi.core.smb_enemy_history import EnemyObservationHistory

        if self.observer is None:
            self.observer = EnemyObservationHistory()
        self.history = self.observer.observe(env, env.steps)


def _bridge_phase(env, state: TeacherState) -> str:
    """The moving-platform teacher's phase now ("" when none is being crossed).

    Worked out once per frame and kept in ``state``.
    """
    from .bridge_traversal import bridge_phase
    from .hierarchy import bridge_training_active

    if state.phase_step == env.steps:
        return state.phase
    if env._bridge_jump_task:
        phase = "finish" if env._goal_credited else ("exit" if env._bridge_boarded else "board")
    elif bridge_training_active(env):
        phase = bridge_phase(env, state.bridge_opening)
        if state.phase == "exit" and phase not in ("finish",):
            phase = "exit"  # once committed to leaving the bridge, keep leaving
        if phase != "wait":
            state.bridge_opening = False
    else:
        phase = ""
    state.phase, state.phase_step = phase, env.steps
    return phase


def _bridge_jump_now(env) -> bool:
    """A bridge jump task's teacher jumps from here now (as coached_suffix does)."""
    from retroagi.core.smb_coaching import training_target

    from .bridge_curriculum import bridge_jump_allowed, bridge_takeoff_window
    from .local_traversal import safe_jump_holds

    if not env.mario["on_ground"]:
        return True
    direction = training_target(env).direction
    if not safe_jump_holds(env, training_target(env), direction):
        return False
    now, later = bridge_takeoff_window(
        env, lambda: safe_jump_holds(env, training_target(env), direction)
    )
    return bridge_jump_allowed(now, later)


def schedule_stance(env, state: TeacherState) -> tuple[str, int]:
    """The layout's schedule stance here (tactic_schedule.SCHEDULE_STANCES) and
    the way it goes: its current segment's, by the segment's rule."""
    from .monster import monster_choice
    from .piranha_tactics import tactical_choice, timed_plant

    seg = tactic_schedule.current(env)
    direction = seg["direction"]
    stance = seg["stance"]
    if seg["kind"] == "bridge":
        phase = _bridge_phase(env, state)
        if env._bridge_jump_task:
            stance = "advance" if phase == "finish" or _bridge_jump_now(env) else "hold_area"
        else:
            stance = "hold_area" if phase in ("wait", "ride") else "advance"
    elif seg["kind"] == "plant":
        plant = timed_plant(env)
        stance = "advance"
        if (
            plant is not None
            and env.mario["on_ground"]
            and env.mario["x"] < plant["x"] + plant["w"]
        ):
            stance = tactical_choice(env, state.history)[0]
    elif seg["kind"] == "monster":
        choice = monster_choice(env)
        stance = choice[0] if choice is not None else "advance"
    if seg["kind"] != "plain" and stance == "retreat":
        direction = -direction
    return stance, direction


def _move(env, state: TeacherState, stance: str, direction: int) -> tuple[str, int]:
    """The next move under a schedule stance: walk, climb or descend, and its way."""
    from retroagi.core.smb_coaching import training_target

    from .monster import active_monster

    seg = tactic_schedule.current(env)
    if stance == "hold_area":
        return "hold", 0
    if stance == "retreat" and (seg["stance"] == "retreat" or seg["kind"] != "plain"):
        # Backing out of a dead end, away from a monster or a plant.
        return "walk", direction
    phase = _bridge_phase(env, state) if seg["kind"] == "bridge" else ""
    if phase and not env._bridge_jump_task:
        # Walking up to, onto, along and off a moving platform.
        return "walk", direction
    if phase in ("board", "exit"):
        return "walk", int(training_target(env).direction)
    if active_monster(env) is not None:
        return "walk", direction
    # A move lasts until Mario lands: in the air he keeps the move he left the
    # ground with (a jump, a climb, a drop).
    held = state.notes.get("ground_move")
    if env.mario["on_ground"] or held is None:
        objective = training_target(env)
        way = int(objective.direction) if objective.direction in (-1, 1) else direction
        state.notes["ground_move"] = (_objective_move(env, objective), way)
    return state.notes["ground_move"]


def _objective_move(env, objective) -> str:
    """How an objective is reached from where Mario stands: a platform higher
    than his feet is climbed, a lower one (or a goal on lower ground)
    descended to; anything else is walked or jumped to."""
    feet = env.mario["y"] + env.mario["h"]
    if objective.kind == "mount" or objective.kind in ("finish", "retreat"):
        if objective.kind == "mount" and objective.top < feet - STEP:
            return "climb"
        if objective.top > feet + STEP:
            return "descend"
    return "walk"


def tactic_of_move(move: str, direction: int) -> str:
    """A move and its way as a tactic name: forward is right, backward left."""
    if move == "hold":
        return "hold_ground"
    if move in ("climb", "descend"):
        return f"{move}_{'forward' if direction > 0 else 'backward'}"
    return "advance" if direction > 0 else "retreat"


def teacher_tactic(env, state: TeacherState) -> TacticToken:
    """The tactic here (see the module notes)."""
    stance, direction = schedule_stance(env, state)
    return tactic_token(tactic_of_move(*_move(env, state, stance, direction)))


def teacher_strategy(state: TeacherState) -> StrategyToken:
    return StrategyToken(state.strategy, state.direction)


def _first_plan(route: list[int]) -> ActionPlan:
    """The first stretch of one button action in a per-frame route, as that action
    and its length (at most the longest frame count, 32 frames)."""
    action = int(route[0])
    run = 1
    while run < len(route) and int(route[run]) == action:
        run += 1
    return ActionPlan(action, min(run, FRAME_COUNTS[-1]))


def _fingerprint(env) -> int:
    """Everything a frame can change in the simulator, hashed: equal fingerprints
    are the same state, so the same route continues from both."""
    from .env_state import snapshot_env_state

    snap = snapshot_env_state(env)
    snap["platforms"] = [
        (tuple(p["rect"]), p.get("move_x"), p.get("move_dir")) for p in snap["platforms"]
    ]
    snap["coins"] = [coin["collected"] for coin in snap["coins"]]
    snap["power_ups"] = [item["collected"] for item in snap["power_ups"]]
    snap["goal"] = tuple(snap["goal"]) if snap["goal"] is not None else None
    return hash(repr(snap))


def _remembered_route(env, state: TeacherState) -> Optional[list[int]]:
    """The rest of the last coached route, when the episode has followed it to here."""
    if not state.notes.get("route"):
        return None
    start, route, fingerprints = state.notes["route"]
    k = env.steps - start
    if 0 <= k < len(route) and fingerprints[k] == _fingerprint(env):
        return route[k:]
    return None


def _remember_route(env, state: TeacherState, route: list[int]) -> None:
    """Keep a route with the state before each of its frames (replayed, then restored)."""
    from retroagi.core.smb_coaching import probe_state

    fingerprints = []
    with probe_state(env):
        for action in route:
            fingerprints.append(_fingerprint(env))
            env.step(action)
    state.notes["route"] = (env.steps, list(route), fingerprints)


def teacher_plan(
    env, state: TeacherState, *, certify_holds: bool = True
) -> tuple[Optional[ActionPlan], tuple[int, ...]]:
    """The coached route's next action and frame count, and every certified jump hold.

    Returns (None, ()) when no coached route reaches the goal from here. While
    the episode follows the last route frame for frame (the simulator is
    deterministic), the rest of that route is used instead of a new search.
    """
    from retroagi.core.smb_coaching import probe_state, training_target

    from .local_traversal import safe_jump_holds
    from .policy_recovery import coached_suffix

    route = _remembered_route(env, state)
    if route is None and env.steps == 0 and state.notes.get("initial_route"):
        route = list(state.notes["initial_route"])
        _remember_route(env, state, route)
    if route is None:
        # The coached route plays itself forward in the simulator; put it back after.
        with probe_state(env):
            route = coached_suffix(
                env,
                max_frames=env.frame_budget,
                observation_history=copy.deepcopy(state.observer) if state.observer else None,
            )
        if not route:
            state.notes.pop("route", None)
            return None, ()
        _remember_route(env, state, route)
    plan = _first_plan(route)
    if plan.action == SMBAction.NOOP and teacher_tactic(env, state).stance == "hold_ground":
        plan = ActionPlan(HOLD_GROUND, plan.frames)
    holds: tuple[int, ...] = ()
    if certify_holds and plan.action in SMB_JUMP_ACTIONS:
        direction = -1 if plan.action == SMBAction.LEFT_JUMP else 1
        if getattr(env, "_action_jump_direction", 0):
            from .policy_recovery import single_jump_routes

            holds = tuple(h for h in single_jump_routes(env) if h > 0)
        else:
            holds = tuple(
                safe_jump_holds(env, training_target(env), direction, plant_history=state.history)
            )
    return plan, holds


def episode_teacher(scenario) -> TeacherState:
    goal = scenario.get("goal")
    direction = -1 if goal is not None and goal[0] + goal[2] / 2 < scenario["mario"][0] else 1
    return TeacherState(
        family=str(block_smb_monte_carlo_metadata(scenario).get("family", "")),
        direction=direction,
        strategy=scenario.get("strategy") or DEFAULT_STRATEGY.kind,
        notes={
            "initial_route": block_smb_monte_carlo_metadata(scenario)
            .get("oracle", {})
            .get("actions")
        },
    )


def teacher_skill(env, state: TeacherState, plan: Optional[ActionPlan]) -> SkillToken:
    """A spatial destination for the next maneuver, from the teacher's rollout.

    A jump targets its landing/stomp, not its takeoff or a hidden object ID.
    Runs end at an objective, support transition or bounded travel waypoint.
    An upcoming jump includes its approach in the same destination. The probe
    is restored exactly; only the spatial command reaches the executor.
    """
    from retroagi.core.smb_coaching import probe_state

    if plan is not None and (
        plan.action == HOLD_GROUND or (plan.action == SMBAction.NOOP and env.mario["on_ground"])
    ):
        return SkillToken("hold", 0, 0)
    start_x = env.mario["x"] + env.mario["w"] / 2
    start_y = env.mario["y"] + env.mario["h"]
    route = _remembered_route(env, state) if plan is not None else None
    mode = "jump" if plan is not None and plan.action in SMB_JUMP_ACTIONS else "run"
    if route:
        from .tactic_schedule import _support_index

        with probe_state(env):
            airborne = not env.mario["on_ground"]
            mark = (env._tactic_index, env._route_done)
            support = _support_index(env)
            platforms = [p["rect"].copy() for p in env.platforms]
            riding = (
                support is not None
                and env.platforms[support].get("moving")
                and env.mario["x"] >= platforms[support].left + 4
            )
            for index, action in enumerate(route[:160]):
                # Label a whole maneuver, including its approach, rather than
                # an acceleration fragment whose endpoint assumes momentum.
                # A ride/wait starts a new command. Passive platform carry
                # must not become a ground waypoint that chases world motion.
                if mode == "run" and index and action == SMBAction.NOOP and not riding:
                    break
                if action in SMB_JUMP_ACTIONS:
                    mode = "jump"
                _, _, done, _, _ = env.step(action)
                airborne = airborne or not env.mario["on_ground"]
                landed = airborne and (env.mario["on_ground"] or env.stomped)
                objective = (env._tactic_index, env._route_done) != mark
                on = _support_index(env)
                boarded = (
                    on != support
                    and on is not None
                    and env.platforms[on].get("moving")
                    and env.mario["x"] >= env.platforms[on]["rect"].left + 4
                    and env.mario["x"] + env.mario["w"] <= env.platforms[on]["rect"].right - 4
                )
                dismounted = (
                    support is not None
                    and env.platforms[support].get("moving")
                    and on is not None
                    and not env.platforms[on].get("moving")
                    and env.mario["x"] >= env.platforms[on]["rect"].left + 4
                    and env.mario["x"] + env.mario["w"] <= env.platforms[on]["rect"].right - 4
                )
                if done or landed or objective or boarded or dismounted:
                    break
                if mode == "run" and index >= 63 and not riding:
                    # A barely overlapping collision box is not a stable
                    # waypoint. Finish stepping onto the support before the
                    # executor brakes or vision chooses the wrong support.
                    if on is not None and (
                        env.mario["x"] >= env.platforms[on]["rect"].left + 4
                        and env.mario["x"] + env.mario["w"] <= env.platforms[on]["rect"].right - 4
                    ):
                        break
            x = env.mario["x"] + env.mario["w"] / 2 - start_x
            y = env.mario["y"] + env.mario["h"] - start_y
            on = _support_index(env)
            if mode == "run" and on is not None and env.platforms[on].get("moving"):
                # Express the landing/boarding point on the support as seen
                # NOW. Runtime tracks that visual support; no IDs are passed.
                x -= env.platforms[on]["rect"].left - platforms[on].left
                y -= env.platforms[on]["rect"].top - platforms[on].top
    else:
        # Unlabelled recovery state: provide a local destination so collection
        # can continue, but do not treat it as a certified skill demonstration.
        x = env.goal.centerx - start_x if env.goal is not None else 0
        y = 0
    return SkillToken(mode, max(-256, min(256, round(x))), max(-240, min(240, round(y))))
