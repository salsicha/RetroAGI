"""Local obstacle objectives and physics-verified jump-duration labels.

These labels supervise attempts from their actual initiation state. They never
choose the policy's action or replace the episode's final success condition.
"""

from dataclasses import dataclass

from .geometry_expert import restore_env_state, snapshot_env_state

LOCAL_TRAVERSAL_FAMILIES = frozenset(
    "tall_pipe_jump pit_leap pipe_mount enemy_hop stair_climb single_gap retreat_recovery "
    "platform_chain mixed_section full_smb_opening_proxy enemy_patrol enemy_gap "
    "chained_obstacles chained_enemy_gauntlet".split()
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
            return bool(
                m["on_ground"]
                and support is not None
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


def local_objective(env) -> LocalObjective:
    """Next obstacle in the direction of the goal, based on current geometry."""
    m = env.mario
    x, feet = m["x"], m["y"] + m["h"]
    goal = env.goal
    finish = LocalObjective("finish", goal.left, goal.right, goal.bottom)
    if goal.centerx < x:
        return _left_objective(env)
    candidates = []
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        if (
            r.right <= x
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
            if not p.get("moving") and p["rect"].left >= edge
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


def _left_objective(env) -> LocalObjective:
    """Mirror obstacle selection without changing world coordinates or physics."""
    m = env.mario
    x, feet = m["x"], m["y"] + m["h"]
    goal = env.goal
    finish = LocalObjective("retreat", goal.left, goal.right, goal.bottom, direction=-1)
    candidates = []
    for i, p in enumerate(env.platforms):
        r = p["rect"]
        if (
            not p.get("moving")
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
            if not p.get("moving") and p["rect"].right <= edge
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
    env, objective: LocalObjective, direction: int, *, verify_recovery: bool = True
) -> list[int]:
    """Replay the 1–16-frame menu through landing or terminal success, then restore.

    A successful jump must clear this objective alive. The full environment
    snapshot includes goal credit and reward potentials so probes cannot leak
    credit, shaping, or platform phase into the live episode.
    """
    snapshot = snapshot_env_state(env)
    original_render = env.__dict__.get("render")
    env.render = lambda: None
    valid = []
    try:
        for hold in range(1, 17):
            restore_env_state(env, snapshot)
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
                        following = local_objective(env)
                        distance = local_target_distance(env, following)
                        if following.kind == "enemy" and distance < 50:
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


def terrain_oracle(scenario: dict, max_steps: int = 300) -> list[int]:
    """Generate a replayable sequence by solving successive local obstacles."""
    from .env import MarioScenarioEnv

    env = MarioScenarioEnv()
    actions = []
    hold_remaining = 0
    in_jump = False
    try:
        env.reset(scenario=scenario)
        env.render = lambda: None
        for _ in range(max_steps):
            target = local_objective(env)
            if in_jump and env.mario["on_ground"]:
                in_jump = False
            direction = target.direction
            if hold_remaining:
                action = 2 if direction > 0 else 4
                hold_remaining -= 1
            elif in_jump or not env.mario["on_ground"]:
                action = 1 if direction > 0 else 3
            elif target.kind in ("finish", "retreat"):
                action = 1 if direction > 0 else 3
            else:
                distance = local_target_distance(env, target)
                valid = safe_jump_holds(env, target, direction) if distance < 50 else []
                if valid:
                    hold_remaining = valid[len(valid) // 2] - 1
                    in_jump = True
                    action = 2 if direction > 0 else 4
                else:
                    action = 1 if direction > 0 else 3
            _, _, done, truncated, info = env.step(action)
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
                if recovering or held > 16
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


def traversal_metrics(episodes, cleared, objectives, finishes, timeouts, deaths):
    return {
        "episodes": episodes,
        "episodes_with_local_clear": cleared,
        "local_objectives_completed": objectives,
        "finishes_after_local_clear": finishes,
        "timeouts": timeouts,
        "deaths": deaths,
        "local_clear_rate": cleared / episodes if episodes else None,
        "finish_after_local_clear_rate": finishes / cleared if cleared else None,
        "timeout_rate": timeouts / episodes if episodes else None,
    }


TRAVERSAL_COUNT_FIELDS = (
    "episodes",
    "episodes_with_local_clear",
    "local_objectives_completed",
    "finishes_after_local_clear",
    "timeouts",
    "deaths",
)
