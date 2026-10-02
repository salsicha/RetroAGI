"""Teachers for the four-layer agent in Block SMB. Training only.

At each decision point of a training episode these say what each layer
should emit: the strategy, tactic and skill tokens, and the action plan. They
read the simulator's own state, which is allowed only for teaching. A policy
never receives anything from here, except, while one layer is being trained,
the explicit token for the layer above it.

They reuse the existing Block SMB teachers:

- skill: the local objective the old trainer coached (training_target, or the
  bridge phase on bridge layouts), its kind mapped onto a skill, its
  direction, whether the enemy must be stomped, and its target matched to the
  object the vision transformer reports for it;
- tactic: the old tactic labels (tactics.tactic_label), advance otherwise;
- strategy: progress, unless a strategy family names another;
- action: the first segment of the coached route from the current state
  (policy_recovery.coached_suffix), as an action and a frame count on the
  executor's menus; for a jump, also every certified hold (safe_jump_holds).
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

from retroagi.core.actions import SMB_JUMP_ACTIONS, SMBAction
from retroagi.core.smb_executor import FRAME_COUNTS, ActionPlan
from retroagi.core.smb_observer import packed_lists
from retroagi.core.smb_scene_labels import SceneObservation, box_overlap
from retroagi.core.tokens import SkillToken, StrategyToken, TacticToken

from .monte_carlo import block_smb_monte_carlo_metadata

# Objective kinds (local_traversal / smb_objectives / bridge phases) -> skills.
OBJECTIVE_SKILLS = {
    "gap": "clear_gap",
    "mount": "mount_platform",
    "enemy": "enemy_clear",
    "stomp": "enemy_clear",
    "retreat": "retreat_recover",
    "finish": "advance",
    "wait": "wait_pass",
    "ride": "wait_pass",
    "approach": "mount_platform",
    "board": "mount_platform",
    "exit": "clear_gap",
}
# Strategy families teach these strategies; every other family teaches progress.
FAMILY_STRATEGIES: dict[str, str] = {}
# A teacher's surface matches a reported one at most this many rows apart.
SURFACE_MATCH = 3


@dataclass
class TeacherState:
    """What a teacher carries through one episode (training only)."""

    family: str
    bridge_opening: bool = True
    phase: str = ""
    observer: object = None  # EnemyObservationHistory, fed every frame by observe_frame
    history: object = None  # its latest features
    notes: dict = field(default_factory=dict)

    def observe_frame(self, env) -> None:
        """Call once per frame: the plant and enemy teachers need the history."""
        from retroagi.core.smb_enemy_history import EnemyObservationHistory

        if self.observer is None:
            self.observer = EnemyObservationHistory()
        self.history = self.observer.observe(env, env.steps)


def _goal_direction(env) -> int:
    return -1 if env.goal is not None and env.goal.centerx < env.mario["x"] else 1


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


def teacher_skill(env, scene: SceneObservation, state: TeacherState) -> SkillToken:
    """The skill the old teachers coached here, pointing at what the vision reports."""
    from retroagi.core.smb_coaching import training_target

    from .bridge_traversal import bridge_phase
    from .hierarchy import bridge_training_active

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
    state.phase = phase
    if phase:
        kind = OBJECTIVE_SKILLS[phase]
        target = None
        if phase in ("wait", "ride", "approach", "board"):
            lifts = [p for p in env.platforms if p.get("moving")]
            if lifts:
                rect = lifts[0]["rect"]
                left, right = _screen(env, rect.left, rect.right)
                target = _box_pointer(
                    scene, "moving_platforms", (left, rect.top, right, rect.bottom)
                ) or _surface_pointer(scene, left, right, rect.top)
        return SkillToken(kind, _goal_direction(env), False, target)

    objective = training_target(env)
    state.phase = objective.kind
    kind = OBJECTIVE_SKILLS.get(objective.kind, "advance")
    target = None
    if objective.enemy_index is not None:
        target = _box_pointer(scene, "enemies", _enemy_screen_box(env, objective.enemy_index))
    elif objective.kind in ("gap", "mount"):
        left, right = _screen(env, objective.left, objective.right)
        target = _surface_pointer(scene, left, right, objective.top)
    return SkillToken(
        kind,
        int(objective.direction) if objective.direction in (-1, 1) else _goal_direction(env),
        objective.kind == "stomp",
        target,
    )


def teacher_tactic(env, state: TeacherState, skill: SkillToken) -> TacticToken:
    """The old tactic label for this state, or advance in the skill's direction."""
    from .piranha_tactics import tactic_label as piranha_label
    from .tactics import TACTIC_STANCES, tactic_label

    label = piranha_label(env, state.history) if state.history is not None else -1
    if label < 0:
        label = tactic_label(env, state.history, None, family=state.family, phase=state.phase)
    stance = TACTIC_STANCES[label] if label >= 0 else "advance"
    direction = -skill.direction if stance == "retreat" else skill.direction
    return TacticToken(stance, direction)


def teacher_strategy(state: TeacherState) -> StrategyToken:
    return StrategyToken(FAMILY_STRATEGIES.get(state.family, "progress"))


def _first_plan(route: list[int]) -> ActionPlan:
    """The first stretch of one button action in a per-frame route, as that action
    and its length (at most the longest frame count, 32 frames)."""
    action = int(route[0])
    run = 1
    while run < len(route) and int(route[run]) == action:
        run += 1
    return ActionPlan(action, min(run, FRAME_COUNTS[-1]))


def teacher_plan(env, state: TeacherState) -> tuple[Optional[ActionPlan], tuple[int, ...]]:
    """The coached route's next action and frame count, and every certified jump hold.

    Returns (None, ()) when no coached route reaches the goal from here.
    """
    from retroagi.core.smb_coaching import probe_state, training_target

    from .local_traversal import safe_jump_holds
    from .policy_recovery import coached_suffix

    # The coached route plays itself forward in the simulator; put it back after.
    with probe_state(env):
        route = coached_suffix(
            env,
            observation_history=copy.deepcopy(state.observer) if state.observer else None,
        )
    if not route:
        return None, ()
    plan = _first_plan(route)
    holds: tuple[int, ...] = ()
    if SMBAction(plan.action) in SMB_JUMP_ACTIONS:
        direction = -1 if plan.action == SMBAction.LEFT_JUMP else 1
        holds = tuple(
            safe_jump_holds(env, training_target(env), direction, plant_history=state.history)
        )
    return plan, holds


def episode_teacher(scenario) -> TeacherState:
    return TeacherState(family=str(block_smb_monte_carlo_metadata(scenario).get("family", "")))
