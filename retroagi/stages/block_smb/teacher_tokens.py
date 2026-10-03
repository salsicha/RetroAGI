"""Teachers for the four-layer agent in Block SMB. Training only.

At each decision point of a training episode these say what each layer
should emit: the strategy, tactic and skill tokens, and the action plan. They
read the simulator's own state, which is allowed only for teaching. A policy
never receives anything from here, except, while one layer is being trained,
the explicit token for the layer above it.

- strategy: the one the layout is played for (a strategy course names it;
  every other layout is a speed run), toward the side the goal was on when
  the episode began;
- tactic: what the layout's current tactic segment says (tactic_schedule).
  Moving-platform, plant and monster segments change it inside by their
  teachers' rules: hold the area while waiting for (or riding) a moving
  platform, while waiting for a plant to go back into its pipe, or while a
  monster is far enough; retreat while backing away from it; advance
  otherwise;
- skill: decided by the tactic. Hold area is waiting; retreat is retreating;
  advance and alternate route take the next step of their path (the segment's
  route platform, else the nearest obstacle toward the goal): climb, descend,
  jump gap, stomp or advance. Its target is matched to the object the vision
  transformer reports for it;
- action: the first segment of the coached route from the current state
  (policy_recovery.coached_suffix), as an action and a frame count on the
  executor's menu; for a jump, also every certified hold (safe_jump_holds).
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from retroagi.core.actions import SMB_JUMP_ACTIONS, SMBAction
from retroagi.core.smb_executor import FRAME_COUNTS, ActionPlan
from retroagi.core.smb_observer import packed_lists
from retroagi.core.smb_scene_labels import SceneObservation, box_overlap
from retroagi.core.tokens import DEFAULT_STRATEGY, SkillToken, StrategyToken, TacticToken

from . import tactic_schedule
from .monte_carlo import block_smb_monte_carlo_metadata

# An objective (local_traversal / smb_objectives) under advance or alternate
# route -> its skill. Getting past an enemy is part of every skill, so it is
# advancing; a platform to stand on is climbed or descended to by its height
# (STEP), walked onto if level.
OBJECTIVE_SKILLS = {
    "gap": "jump_gap",
    "enemy": "advance",
    "stomp": "stomp",
    "retreat": "advance",  # the finish, when the goal lies to the left
    "finish": "advance",
}
# A platform this many pixels above (below) Mario's feet is climbed (descended to).
STEP = 2
# A teacher's surface matches a reported one at most this many rows apart.
SURFACE_MATCH = 3


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


def _screen(env, left: float, right: float) -> tuple[float, float]:
    camera = int(env.camera_x)
    return left - camera, right - camera


def _surface_pointer(scene, x0: float, x1: float, top: float):
    best, best_overlap = None, 0.0
    for slot, surface in enumerate(packed_lists(scene)["surfaces"]):
        if abs(surface.top - top) > SURFACE_MATCH:
            continue
        overlap = min(surface.x1, x1) - max(surface.x0, x0)
        if overlap > best_overlap:
            best, best_overlap = ("surfaces", slot), overlap
    return best


def _box_pointer(scene, name: str, box) -> Optional[tuple[str, int]]:
    best, best_overlap = None, 0.2
    for slot, item in enumerate(packed_lists(scene)[name]):
        overlap = box_overlap(getattr(item, "box", item), box)
        if overlap > best_overlap:
            best, best_overlap = (name, slot), overlap
    return best


def _enemy_screen_box(env, index: int):
    enemy = env.enemies[index]
    camera = int(env.camera_x)
    x = int(enemy["x"]) - camera
    height = enemy["h"] + enemy.get("foot_offset", 0)
    return (x, int(enemy["y"]), x + enemy["w"], int(enemy["y"]) + height)


def _enemy_pointer(env, scene, index: Optional[int]):
    if index is None or env.enemies[index]["h"] <= 0:
        return None
    return _box_pointer(scene, "enemies", _enemy_screen_box(env, index))


def _lift_pointer(env, scene):
    lifts = [p for p in env.platforms if p.get("moving")]
    if not lifts:
        return None
    rect = lifts[0]["rect"]
    left, right = _screen(env, rect.left, rect.right)
    return _box_pointer(
        scene, "moving_platforms", (left, rect.top, right, rect.bottom)
    ) or _surface_pointer(scene, left, right, rect.top)


def _bridge_phase(env, state: TeacherState) -> str:
    """The moving-platform teacher's phase now ("" when none is being crossed).

    Worked out once per frame and kept in ``state``: the tactic and the skill
    read the same phase.
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


def teacher_tactic(env, state: TeacherState) -> TacticToken:
    """The layout's tactic here: its current segment's, by the segment's rule."""
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
    return TacticToken(stance, direction)


def teacher_skill(
    env, scene: SceneObservation, state: TeacherState, tactic: TacticToken
) -> SkillToken:
    """The skill the tactic calls for here, pointing at what the vision reports."""
    from retroagi.core.smb_coaching import training_target

    from .monster import active_monster

    seg = tactic_schedule.current(env)
    focus = None  # the enemy a plant or monster segment is about
    if seg["kind"] in ("plant", "monster"):
        focus = seg["end"].get("past_enemy")
    if tactic.stance == "hold_area":
        target = _lift_pointer(env, scene) if seg["kind"] == "bridge" else None
        return SkillToken("wait", tactic.direction, target or _enemy_pointer(env, scene, focus))
    if tactic.stance == "retreat":
        return SkillToken("retreat", tactic.direction, _enemy_pointer(env, scene, focus))
    phase = _bridge_phase(env, state) if seg["kind"] == "bridge" else ""
    if phase and not env._bridge_jump_task:
        # Walking up to, onto, along and off a moving platform is advancing.
        target = _lift_pointer(env, scene) if phase in ("approach", "board") else None
        return SkillToken("advance", tactic.direction, target)
    if phase in ("board", "exit"):
        objective = training_target(env)
        left, right = _screen(env, objective.left, objective.right)
        target = (
            _lift_pointer(env, scene)
            if phase == "board"
            else _surface_pointer(scene, left, right, objective.top)
        )
        return SkillToken("jump_gap", objective.direction, target)
    if active_monster(env) is not None:
        return SkillToken("advance", tactic.direction, None)
    objective = training_target(env)
    if objective.kind == "mount":
        feet = env.mario["y"] + env.mario["h"]
        kind = (
            "climb"
            if objective.top < feet - STEP
            else ("descend" if objective.top > feet + STEP else "advance")
        )
    else:
        kind = OBJECTIVE_SKILLS.get(objective.kind, "advance")
    target = None
    if kind == "stomp" and objective.enemy_index is not None:
        target = _enemy_pointer(env, scene, objective.enemy_index)
    elif objective.kind in ("gap", "mount"):
        left, right = _screen(env, objective.left, objective.right)
        target = _surface_pointer(scene, left, right, objective.top)
    direction = int(objective.direction) if objective.direction in (-1, 1) else tactic.direction
    return SkillToken(kind, direction, target)


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
    if 0 < k < len(route) and fingerprints[k] == _fingerprint(env):
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


def teacher_plan(env, state: TeacherState) -> tuple[Optional[ActionPlan], tuple[int, ...]]:
    """The coached route's next action and frame count, and every certified jump hold.

    Returns (None, ()) when no coached route reaches the goal from here. While
    the episode follows the last route frame for frame (the simulator is
    deterministic), the rest of that route is used instead of a new search.
    """
    from retroagi.core.smb_coaching import probe_state, training_target

    from .local_traversal import safe_jump_holds
    from .policy_recovery import coached_suffix

    route = _remembered_route(env, state)
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
    holds: tuple[int, ...] = ()
    if SMBAction(plan.action) in SMB_JUMP_ACTIONS:
        direction = -1 if plan.action == SMBAction.LEFT_JUMP else 1
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
    )
