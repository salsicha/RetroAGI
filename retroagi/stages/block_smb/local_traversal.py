"""Local obstacle objectives and physics-verified jump-duration labels.

These labels supervise attempts from their actual initiation state. They never
choose the policy's action or replace the episode's final success condition.
"""

from dataclasses import dataclass

from .env_state import restore_env_state, snapshot_env_state
from .transfer_failure_families import TRANSFER_FAILURE_FAMILIES


@dataclass
class JumpReleaseState:
    """Track executor-owned landing frames while replaying executed actions.

    A jump whose button was released in the air owns no landing frames: the
    first grounded frame is a fresh decision (resolve_landing). A jump still
    held when it lands releases on landing and suppresses a new jump for one
    more frame. Stomps reset the executor, and ordinary falls own no release
    frames. Teachers and repair splices must preserve this state across their
    prefix.
    """

    remaining: int = 0
    action: int = 1
    jumping: bool = False
    airborne: bool = False
    bouncing: bool = False

    def observe(self, env, action, info):
        if self.remaining:
            self.remaining -= 1
        if info["reward_terms"]["enemy_stomp"] > 0:
            self.jumping = self.airborne = False
            self.remaining = 0
            self.bouncing = True
        if self.bouncing:
            if env.mario["on_ground"]:
                self.bouncing = False
            return
        if not self.jumping and action in (2, 4, 5):
            self.jumping = True
            self.action = {2: 1, 4: 3, 5: 0}[action]
        if self.jumping:
            self.airborne |= not env.mario["on_ground"]
            if self.airborne and env.mario["on_ground"]:
                self.remaining = 2 if action in (2, 4, 5) else 0
                self.jumping = self.airborne = False


LOCAL_TRAVERSAL_FAMILIES = frozenset(
    "tall_pipe_jump pit_leap pipe_mount enemy_hop stair_climb single_gap retreat_recovery "
    "platform_chain mixed_section full_smb_opening_proxy enemy_patrol enemy_gap "
    "chained_obstacles chained_enemy_gauntlet".split()
    + list(TRANSFER_FAILURE_FAMILIES)
    + ["tactics_obstacle_sequence", "tactics_mixed_sequence", "tactics_bridge_then_gap"]
)


@dataclass(frozen=True)
class LocalObjective:
    kind: str
    left: float
    right: float
    top: float
    platform_index: int | None = None
    enemy_index: int | None = None
    direction: int = 1

    @property
    def center(self):
        return (self.left + self.right) / 2

    def reached(self, env) -> bool:
        m = env.mario
        if self.kind == "stomp":
            return env._stomp_credited
        if self.kind in ("finish", "retreat"):
            return env._goal_credited
        if self.enemy_index is not None:
            enemy = env.enemies[self.enemy_index]
            return bool(
                enemy["dead"]
                or (
                    m["on_ground"]
                    and (
                        m["x"] >= enemy["x"] + enemy["w"]
                        if self.direction > 0
                        else m["x"] + m["w"] <= enemy["x"]
                    )
                )
            )
        if self.platform_index is not None:
            target = env.platforms[self.platform_index]
            support = m.get("_platform")
            off_route = support is not None and any(
                env.platforms[i] is support for i in _forbidden(env)
            )
            return bool(
                m["on_ground"]
                and support is not None
                and not off_route
                and (
                    support is target
                    or (
                        (
                            m["x"] >= target["rect"].right
                            if self.direction > 0
                            else m["x"] + m["w"] <= target["rect"].left
                        )
                        and (self.kind == "gap" or support["rect"].top <= target["rect"].top)
                    )
                )
            )
        return False


def _clear_path_under(env, left, right, bottom):
    m = env.mario
    if env.goal is not None and env.goal.bottom < m["y"] + m["h"] - 1:
        return False
    return bottom <= m["y"] and any(
        not p.get("moving")
        and p["rect"].left <= min(left, m["x"])
        and p["rect"].right >= max(right, m["x"] + m["w"])
        and abs(p["rect"].top - m["y"] - m["h"]) < 1
        for p in env.platforms
    )


def _forbidden(env) -> frozenset:
    """Platforms the current tactic segment forbids or avoids: never a target or landing."""
    from . import tactic_schedule

    seg = tactic_schedule.current(env)
    return frozenset((*seg.get("forbidden", ()), *seg.get("avoid", ())))


def local_objective(env) -> LocalObjective:
    """Next obstacle in the direction of the goal, based on current geometry.

    Platforms the layout's current tactic segment forbids or avoids are left out.
    """
    m = env.mario
    x, feet = m["x"], m["y"] + m["h"]
    goal = env.goal
    finish = LocalObjective("finish", goal.left, goal.right, goal.bottom)
    if goal.centerx < x:
        return _left_objective(env)
    forbidden = _forbidden(env)
    candidates = []
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        if (
            i in forbidden
            or r.right <= x
            or r.left >= goal.right
            or p.get("moving")
            or _clear_path_under(env, r.left, r.right, r.bottom)
        ):
            continue
        # A raised surface still ahead, including a pipe Mario overlaps below.
        if r.top < feet - 1 and r.right > x + m["w"]:
            candidates.append((max(x, r.left), LocalObjective("mount", r.left, r.right, r.top, i)))
    supported = [
        p["rect"]
        for p in env.platforms
        if not p.get("moving")
        and p["rect"].left < x + m["w"]
        and p["rect"].right > x
        and abs(p["rect"].top - feet) < 1
    ]
    if supported:
        edge = max(r.right for r in supported)
        landings = [
            (i, p["rect"])
            for i, p in enumerate(env.platforms)
            if not p.get("moving") and p["rect"].left >= edge and i not in forbidden
        ]
        # A raised pipe ending above continuous lower floor is a descent,
        # not a pit. Calling it a gap selects the next pipe as a landing and
        # requests an impossible pipe-to-pipe jump instead of approaching it.
        lower_floor = any(
            p["rect"].left <= edge < p["rect"].right
            and p["rect"].top > feet + 1
            and not p.get("moving")
            for p in env.platforms
        )
        if landings and not lower_floor and goal.centerx > edge:
            i, r = min(landings, key=lambda pair: pair[1].left)
            # Bound the landing target to its near edge, not the whole far floor.
            candidates.append(
                (edge, LocalObjective("gap", r.left, min(r.right, r.left + 48), r.top, i))
            )
    for i, enemy in enumerate(env.enemies):
        if (
            not enemy["dead"]
            and enemy["h"] > 0
            and enemy["x"] + enemy["w"] > x
            and enemy["x"] < goal.right
            and not _clear_path_under(
                env, enemy["x"], enemy["x"] + enemy["w"], enemy["y"] + enemy["h"]
            )
        ):
            candidates.append(
                (
                    enemy["x"],
                    LocalObjective(
                        "enemy", enemy["x"], enemy["x"] + enemy["w"] + 24, enemy["y"], enemy_index=i
                    ),
                )
            )
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else finish


def route_objective(env) -> LocalObjective | None:
    """The tactic segment's next platform to stand on (tactic_schedule), if any.

    Higher than Mario's feet it is climbed, lower it is descended to (both
    "mount": reached by standing on it); level with them, across open space,
    it is a gap, and otherwise walked onto.
    """
    from . import tactic_schedule

    index = tactic_schedule.next_route_platform(env)
    if index is None:
        return None
    rect = env.platforms[index]["rect"]
    m = env.mario
    feet = m["y"] + m["h"]
    if m["x"] + m["w"] <= rect.left:
        direction = 1
    elif m["x"] >= rect.right:
        direction = -1
    else:
        direction = tactic_schedule.current(env)["direction"]
    level = abs(rect.top - feet) <= 1
    support = m.get("_platform")
    touching = support is not None and (
        support["rect"].right >= rect.left if direction > 0 else support["rect"].left <= rect.right
    )
    if level and m["on_ground"] and not touching:
        near, far = (rect.left, min(rect.right, rect.left + 48))
        if direction < 0:
            near, far = max(rect.left, rect.right - 48), rect.right
        return LocalObjective("gap", near, far, rect.top, index, direction=direction)
    return LocalObjective("mount", rect.left, rect.right, rect.top, index, direction=direction)


def retreat_objective(env) -> LocalObjective | None:
    """A retreat segment's line to walk back to (tactic_schedule), if it is current."""
    from . import tactic_schedule

    seg = tactic_schedule.current(env)
    if seg["stance"] != "retreat" or seg["kind"] != "plain" or "reach_x" not in seg["end"]:
        return None
    x = float(seg["end"]["reach_x"])
    return LocalObjective(
        "retreat", x - 8, x + 8, env.mario["y"] + env.mario["h"], direction=seg["direction"]
    )


def _plant_pipe_index(env, plant):
    """Index of the pipe a piranha plant emerges from, if any."""
    return next(
        (
            i
            for i, p in enumerate(env.platforms)
            if not p.get("moving")
            and p["rect"].top == plant["pipe_top"]
            and p["rect"].left <= plant["x"] < p["rect"].right
        ),
        None,
    )


def plant_clearance_target(env, objective: LocalObjective) -> LocalObjective:
    """Training-only: getting past a pipe's plant is the goal, not landing on the pipe.

    Clearing pipe and plant to the floor beyond is as good as landing past the
    plant on the pipe, and far more forgiving; landing on the pipe's near lip
    is not progress. Uses the plant even while it is hidden, so this must
    supply teacher labels only, never observations (see local_objective).
    """
    m = env.mario
    for i, plant in enumerate(env.enemies):
        if plant.get("kind") != "piranha_plant" or objective.direction < 0:
            continue
        pipe = _plant_pipe_index(env, plant)
        if pipe is None or plant["x"] + plant["w"] <= m["x"]:
            continue
        rect = env.platforms[pipe]["rect"]
        on_pipe = m["on_ground"] and m.get("_platform") is env.platforms[pipe]
        if objective.platform_index == pipe or objective.enemy_index == i or on_pipe:
            return LocalObjective(
                "enemy", plant["x"], rect.right + 24, rect.top, pipe, enemy_index=i
            )
    return objective


def blocking_wall_distance(env, target: LocalObjective) -> float | None:
    """Distance to the near side of a raised surface blocking the target, if any.

    Pressed against it, a takeoff needs a run-up: the teacher backs away and
    re-certifies instead of pushing into the wall.
    """
    if target.platform_index is None or (target.kind != "mount" and target.enemy_index is None):
        return None
    rect = env.platforms[target.platform_index]["rect"]
    m = env.mario
    if rect.top >= m["y"] + m["h"] - 1:
        return None
    return rect.left - m["x"] - m["w"] if target.direction > 0 else m["x"] - rect.right


def _left_objective(env) -> LocalObjective:
    """Mirror obstacle selection without changing world coordinates or physics."""
    m = env.mario
    x, feet = m["x"], m["y"] + m["h"]
    goal = env.goal
    finish = LocalObjective("retreat", goal.left, goal.right, goal.bottom, direction=-1)
    forbidden = _forbidden(env)
    candidates = []
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        if (
            i not in forbidden
            and not p.get("moving")
            and r.left < x
            and r.right > goal.left
            and r.top < feet - 1
            and not _clear_path_under(env, r.left, r.right, r.bottom)
        ):
            candidates.append(
                (
                    max(0, x - r.right),
                    LocalObjective("mount", r.left, r.right, r.top, i, direction=-1),
                )
            )
    support = [
        p["rect"]
        for p in env.platforms
        if not p.get("moving")
        and p["rect"].left < x + m["w"]
        and p["rect"].right > x
        and abs(p["rect"].top - feet) < 1
    ]
    if support:
        edge = min(r.left for r in support)
        landings = [
            (i, p["rect"])
            for i, p in enumerate(env.platforms)
            if not p.get("moving") and p["rect"].right <= edge and i not in forbidden
        ]
        lower_floor = any(
            not p.get("moving")
            and p["rect"].left < edge <= p["rect"].right
            and p["rect"].top > feet + 1
            for p in env.platforms
        )
        if landings and not lower_floor and goal.centerx < edge:
            i, r = max(landings, key=lambda pair: pair[1].right)
            candidates.append(
                (
                    x - edge,
                    LocalObjective(
                        "gap", max(r.left, r.right - 48), r.right, r.top, i, direction=-1
                    ),
                )
            )
    for i, e in enumerate(env.enemies):
        if (
            not e["dead"]
            and e["h"] > 0
            and e["x"] < x + m["w"]
            and e["x"] + e["w"] > goal.left
            and not _clear_path_under(env, e["x"], e["x"] + e["w"], e["y"] + e["h"])
        ):
            candidates.append(
                (
                    max(0, x - e["x"] - e["w"]),
                    LocalObjective(
                        "enemy", e["x"] - 24, e["x"] + e["w"], e["y"], enemy_index=i, direction=-1
                    ),
                )
            )
    return min(candidates, key=lambda pair: pair[0])[1] if candidates else finish


def local_target_distance(env, target: LocalObjective) -> float:
    return (
        target.left - env.mario["x"] - env.mario["w"]
        if target.direction > 0
        else env.mario["x"] - target.right
    )


def stomp_probe_distance(env, target: LocalObjective, base: float = 50) -> float:
    """Search early enough for an incoming enemy's motion during a NES flight.

    This controls collision-teacher search, not policy playback. A certified
    hold still has to succeed; speed never substitutes for a collision probe.
    """
    if target.kind != "stomp" or env.motion is None or target.enemy_index is None:
        return base
    enemy = env.enemies[target.enemy_index]
    incoming = max(0.0, -target.direction * enemy["speed"] * enemy["direction"])
    return base + incoming * 64


def support_edge_distance(env, direction: int) -> float:
    support = env.mario.get("_platform")
    if support is None:
        return float("inf")
    return (
        support["rect"].right - env.mario["x"] - env.mario["w"]
        if direction > 0
        else env.mario["x"] - support["rect"].left
    )


def safe_jump_holds(
    env,
    objective: LocalObjective,
    direction: int,
    *,
    verify_recovery: bool = True,
    plant_history=None,
) -> list[int]:
    """Replay the 1–16-frame menu through landing or terminal success, then restore.

    A successful jump must clear this objective alive. The full environment
    snapshot includes goal credit and reward potentials so probes cannot leak
    credit, shaping, or platform phase into the live episode.
    """
    from .piranha import freeze_plant_envelopes
    from .piranha_tactics import timed_plant, timed_safe_holds

    plant = timed_plant(env)
    if (
        plant is not None
        and objective.enemy_index is not None
        and env.enemies[objective.enemy_index] is plant
    ):
        return timed_safe_holds(env, plant_history, direction)

    snapshot = snapshot_env_state(env)
    original_render = env.__dict__.get("render")
    env.render = lambda: None
    valid = []
    try:
        from retroagi.core.smb_physics import NES_JUMP_FRAMES

        menu = NES_JUMP_FRAMES
        for hold in menu:
            restore_env_state(env, snapshot)
            freeze_plant_envelopes(env)
            airborne = False
            bouncing = False
            for frame in range(96):
                action = (
                    (2 if direction > 0 else 4)
                    if frame < hold and not bouncing
                    else (1 if direction > 0 else 3)
                )
                _, _, done, truncated, info = env.step(action)
                airborne |= not env.mario["on_ground"]
                landed = airborne and env.mario["on_ground"]
                bouncing |= info["reward_terms"]["enemy_stomp"] > 0
                if info["death"]:
                    break
                achieved = (
                    env._goal_credited if env._single_jump_attempt else objective.reached(env)
                )
                if achieved and (landed or env._goal_credited):
                    # Nonterminal stomp contact is not a completed primitive:
                    # the executor must survive the automatic bounce. At the
                    # landing, reject an immediately trapped next-enemy state.
                    recoverable = True
                    if verify_recovery and landed and not env._goal_credited:
                        # Match the live executor before certifying the next
                        # takeoff. A stomp resets it and a jump released in
                        # the air hands back its first grounded frame; only a
                        # jump still held at landing owns release frames.
                        if not bouncing and frame < hold:
                            for _ in range(2):
                                _, _, _, _, release_info = env.step(1 if direction > 0 else 3)
                                if release_info["death"]:
                                    recoverable = False
                                    break
                        following = local_objective(env)
                        distance = local_target_distance(env, following)
                        if recoverable and following.kind == "enemy" and distance < 50:
                            recoverable = bool(
                                safe_jump_holds(env, following, direction, verify_recovery=False)
                            )
                    if recoverable:
                        valid.append(hold)
                    break
                if done or truncated or landed:
                    break
        return valid
    finally:
        restore_env_state(env, snapshot)
        if original_render is None:
            del env.__dict__["render"]
        else:
            env.render = original_render


# Beyond any single jump's horizontal reach, including an approaching enemy.
TAKEOFF_PROBE_DISTANCE = 128
# A window this wide leaves the interior hold two or three holds of margin.
ROBUST_TAKEOFF_HOLDS = 6


def takeoff_timing_actions(env, *, plant_history=None) -> list[bool] | None:
    """Where the next jump should launch from, at a grounded decision.

    Returns a six-slot allowed-action mask toward the current training target,
    or None when no label is certain. Where no hold is certified only moving
    toward the target is allowed. On the window's thin opening edge (fewer
    than ROBUST_TAKEOFF_HOLDS certified holds, and one more running frame
    certifies more) keep running; the spawn hop launched from there. Once the
    window is robust or stops widening, running and jumping are both allowed;
    on its last frame, where one more running frame closes it, only the jump
    is. Probes restore the full environment state.
    """
    from retroagi.core.smb_coaching import training_target

    if not env.mario["on_ground"]:
        return None
    objective = training_target(env)
    if objective.kind not in ("stomp", "mount", "gap", "enemy"):
        return None
    direction = objective.direction
    run, jump = (1, 2) if direction > 0 else (3, 4)

    def certified():
        target = training_target(env)
        if target.kind not in ("stomp", "mount", "gap", "enemy"):
            return []
        if local_target_distance(env, target) >= TAKEOFF_PROBE_DISTANCE:
            return []
        return safe_jump_holds(env, target, target.direction, plant_history=plant_history)

    now = certified()
    snapshot = snapshot_env_state(env)
    original_render = env.__dict__.get("render")
    env.render = lambda: None
    try:
        _, _, done, truncated, info = env.step(run)
        run_survives = not (info["death"] or truncated) and (not done or bool(env._goal_credited))
        later = certified() if now and run_survives and not done else []
    finally:
        restore_env_state(env, snapshot)
        if original_render is None:
            del env.__dict__["render"]
        else:
            env.render = original_render
    allowed = [False] * 6
    if not now:
        if not run_survives:
            return None
        allowed[run] = True
    elif not run_survives or not later:
        allowed[jump] = True
    else:
        allowed[run] = True
        allowed[jump] = len(now) >= ROBUST_TAKEOFF_HOLDS or len(now) >= len(later)
    return allowed


def terrain_oracle(scenario: dict, max_steps: int = 300) -> list[int]:
    """Generate a replayable sequence by solving successive local obstacles."""
    from .env import MarioScenarioEnv

    if any(
        isinstance(e, dict) and e.get("kind") == "piranha_plant"
        for e in scenario.get("enemies", [])
    ):
        from .piranha import plant_oracle

        return plant_oracle(scenario, max_steps)
    env = MarioScenarioEnv()
    actions = []
    hold_remaining = 0
    in_jump = False
    release = JumpReleaseState()
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for _ in range(max_steps):
            target = local_objective(env)
            if (
                (env._goal_on_stomp or env._require_stomp_before_goal)
                and not env._stomp_credited
                and target.kind == "enemy"
            ):
                from dataclasses import replace

                target = replace(target, kind="stomp")
            if in_jump and env.mario["on_ground"]:
                in_jump = False
            direction = target.direction
            if release.remaining:
                hold_remaining = 0
                action = release.action
            elif hold_remaining:
                action = 2 if direction > 0 else 4
                hold_remaining -= 1
            elif in_jump or not env.mario["on_ground"]:
                action = 1 if direction > 0 else 3
            elif getattr(env, "_bridge_then_terrain", False) and not env._bridge_crossed:
                from .bridge_traversal import bridge_phase

                action = 1 if bridge_phase(env, True) in ("approach", "board", "exit") else 0
            elif target.kind in ("finish", "retreat"):
                action = 1 if direction > 0 else 3
            else:
                distance = local_target_distance(env, target)
                valid = (
                    safe_jump_holds(env, target, direction)
                    if distance < stomp_probe_distance(env, target)
                    else []
                )
                if valid:
                    hold_remaining = valid[len(valid) // 2] - 1
                    in_jump = True
                    action = 2 if direction > 0 else 4
                else:
                    action = 1 if direction > 0 else 3
            _, _, done, truncated, info = env.step(action)
            release.observe(env, action, info)
            actions.append(action)
            if info["reward_terms"]["enemy_stomp"] > 0:
                hold_remaining = 0
                in_jump = True
            if done or truncated:
                break
        return actions
    finally:
        env.close()


def normalize_oracle_jumps(scenario: dict, actions: list[int]) -> list[int]:
    """Keep demonstrations within the executor's hold menu and bounce contract."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    result = []
    recovering = False
    held = 0
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for requested in actions:
            if requested not in (2, 4, 5):
                held = 0
            else:
                held += 1
            action = (
                {2: 1, 4: 3, 5: 0}.get(requested, requested)
                if recovering or held > 32
                else requested
            )
            _, _, done, truncated, info = env.step(action)
            result.append(action)
            if info["reward_terms"]["enemy_stomp"] > 0:
                recovering = True
            elif env.mario["on_ground"]:
                recovering = False
            if done or truncated:
                break
        return result
    finally:
        env.close()
