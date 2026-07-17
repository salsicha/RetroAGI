"""State-conditional Block SMB expert that plans by forward simulation.

The scripted teachers in this stage are open-loop, time-indexed action lists:
their labels are only correct for states on the script's exact timeline. This
expert is state-conditional by construction — from the environment's *current*
state it snapshots the full mutable state, simulates a small set of candidate
macro-plans (run, jump now with varying hold, wait then go, back up for a
run-up), scores each rollout, and returns the first action of the best plan.

Because the environment is deterministic given its state, the lookahead is
exact: the expert automatically discovers gap jumps, enemy stomps or waits,
moving-bridge timing, and leftward retreats without per-scenario tuning. It is
suitable both as a DAgger labeler for states visited by a student policy and
as a closed-loop reference policy.

The env is restored to its exact pre-planning state after every call, and
rendering is stubbed out during simulation, so planning has no side effects
and stays fast.
"""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pygame

from retroagi.stages.block_smb.env import MarioScenarioEnv

NOOP = 0
RIGHT = 1
RIGHT_JUMP = 2
LEFT = 3
LEFT_JUMP = 4
JUMP = 5

DEFAULT_RUN_HORIZON = 40
DEFAULT_JUMP_HOLDS = (4, 8, 12, 18)
DEFAULT_RUN_UPS = (3, 6, 9, 12, 16, 20, 26)
DEFAULT_ADVANCES = (4, 8, 14, 20)
DEFAULT_WAIT_LENGTHS = (8, 20, 36, 56)
DEFAULT_RETREAT_LENGTHS = (6, 14)
DEFAULT_REPLAN_INTERVAL = 4
DEFAULT_SETTLE_STEPS = 24

_GOAL_SCORE = 1_000_000.0
_DEATH_SCORE = -1_000_000.0


def snapshot_env_state(env: MarioScenarioEnv) -> dict[str, Any]:
    """Copy every field of the env that ``step`` mutates."""

    mario = dict(env.mario)
    mario["_platform"] = None  # recomputed inside every step before use
    return {
        "mario": mario,
        "platforms": [
            {**{k: v for k, v in plat.items() if k != "rect"}, "rect": plat["rect"].copy()}
            for plat in env.platforms
        ],
        "coins": [
            {"rect": coin["rect"].copy(), "collected": coin["collected"]} for coin in env.coins
        ],
        "enemies": [dict(enemy) for enemy in env.enemies],
        "camera_x": env.camera_x,
        "score": env.score,
        "steps": env.steps,
        "world_width": env.world_width,
        "_max_x_reached": env._max_x_reached,
        "_airborne_started_with_jump": env._airborne_started_with_jump,
    }


def restore_env_state(env: MarioScenarioEnv, snapshot: Mapping[str, Any]) -> None:
    """Restore a snapshot produced by :func:`snapshot_env_state`."""

    env.mario = dict(snapshot["mario"])
    env.platforms = [
        {**{k: v for k, v in plat.items() if k != "rect"}, "rect": plat["rect"].copy()}
        for plat in snapshot["platforms"]
    ]
    env.coins = [
        {"rect": coin["rect"].copy(), "collected": coin["collected"]} for coin in snapshot["coins"]
    ]
    env.enemies = [dict(enemy) for enemy in snapshot["enemies"]]
    env.camera_x = snapshot["camera_x"]
    env.score = snapshot["score"]
    env.steps = snapshot["steps"]
    env.world_width = snapshot["world_width"]
    env._max_x_reached = snapshot["_max_x_reached"]
    env._airborne_started_with_jump = snapshot["_airborne_started_with_jump"]


@dataclass(frozen=True)
class _PlanOutcome:
    reached_goal: bool
    died: bool
    steps: int
    final_distance: float
    best_safe_distance: float

    # Only the first replan_interval actions of a plan are ever executed, so a
    # plan is scored by the best *safe* (on-ground, alive) state it passes
    # through, not by its final fate: a jump that lands on the next platform
    # must beat standing still even if the simulated tail later walks off an
    # edge — the expert replans long before that tail runs.
    _DEATH_PENALTY = 25.0

    @property
    def score(self) -> float:
        if self.reached_goal:
            return _GOAL_SCORE - float(self.steps)
        score = -self.best_safe_distance
        if self.died:
            score -= self._DEATH_PENALTY
        return score


class BlockSMBGeometryExpert:
    """Deterministic lookahead expert for :class:`MarioScenarioEnv`.

    ``action(env)`` returns the next expert action for the env's current
    state. The env must have been ``reset`` and must use the default physics
    constants (the plans are simulated on the env itself, so any scenario
    geometry is supported).
    """

    def __init__(
        self,
        *,
        run_horizon: int = DEFAULT_RUN_HORIZON,
        jump_holds: tuple[int, ...] = DEFAULT_JUMP_HOLDS,
        run_ups: tuple[int, ...] = DEFAULT_RUN_UPS,
        advances: tuple[int, ...] = DEFAULT_ADVANCES,
        wait_lengths: tuple[int, ...] = DEFAULT_WAIT_LENGTHS,
        retreat_lengths: tuple[int, ...] = DEFAULT_RETREAT_LENGTHS,
        replan_interval: int = DEFAULT_REPLAN_INTERVAL,
        settle_steps: int = DEFAULT_SETTLE_STEPS,
    ):
        if run_horizon <= 0:
            raise ValueError("run_horizon must be positive")
        if replan_interval <= 0:
            raise ValueError("replan_interval must be positive")
        if settle_steps < 0:
            raise ValueError("settle_steps must be non-negative")
        self.run_horizon = int(run_horizon)
        self.jump_holds = tuple(int(hold) for hold in jump_holds)
        self.run_ups = tuple(int(run_up) for run_up in run_ups)
        self.advances = tuple(int(advance) for advance in advances)
        self.wait_lengths = tuple(int(wait) for wait in wait_lengths)
        self.retreat_lengths = tuple(int(retreat) for retreat in retreat_lengths)
        self.replan_interval = int(replan_interval)
        self.settle_steps = int(settle_steps)
        self._plan: list[int] = []

    def reset(self) -> None:
        """Drop any cached plan (call between episodes)."""

        self._plan = []

    def action(self, env: MarioScenarioEnv) -> int:
        if env.mario is None:
            raise ValueError("env must be reset before querying the geometry expert")
        if not self._plan:
            self._plan = self._best_plan(env)[: self.replan_interval]
        return self._plan.pop(0)

    def plan(self, env: MarioScenarioEnv) -> list[int]:
        """Return the full currently-best plan without consuming it."""

        return self._best_plan(env)

    # ── Planning internals ───────────────────────────────────────────────────

    def _goal_direction(self, env: MarioScenarioEnv) -> int:
        if env.goal is None:
            return 1
        goal_center = env.goal.x + env.goal.w / 2.0
        mario_center = env.mario["x"] + env.mario["w"] / 2.0
        return 1 if goal_center >= mario_center else -1

    def _candidate_plans(self, direction: int) -> list[list[int]]:
        move = RIGHT if direction > 0 else LEFT
        move_jump = RIGHT_JUMP if direction > 0 else LEFT_JUMP
        retreat = LEFT if direction > 0 else RIGHT
        horizon = self.run_horizon

        plans: list[list[int]] = [[move] * horizon]
        # Bounded advances: creep toward a hazard and stop, so incremental
        # progress survives the settle tail even when the full run would not.
        for advance in self.advances:
            plans.append([move] * advance)
        for hold in self.jump_holds:
            plans.append([move_jump] * hold + [move] * horizon)
        # Run-up jumps: keep running and launch at a future edge or obstacle.
        for run_up in self.run_ups:
            for hold in self.jump_holds[1:3]:
                plans.append([move] * run_up + [move_jump] * hold + [move] * horizon)
        for wait in self.wait_lengths:
            plans.append([NOOP] * wait + [move] * horizon)
        for wait in self.wait_lengths[:-1]:
            for hold in self.jump_holds[1:2]:
                plans.append([NOOP] * wait + [move_jump] * hold + [move] * horizon)
        for backup in self.retreat_lengths:
            for hold in self.jump_holds[1:3]:
                plans.append([retreat] * backup + [move_jump] * hold + [move] * horizon)
        # Standing still can be optimal mid-air or when boxed in; keep a pure
        # wait so the fallback ordering never forces a fatal move.
        plans.append([NOOP] * max(self.wait_lengths))
        return plans

    def _best_plan(self, env: MarioScenarioEnv) -> list[int]:
        direction = self._goal_direction(env)
        snapshot = snapshot_env_state(env)
        original_max_steps = env.max_steps
        # Keep simulations from tripping the episode timeout mid-plan.
        env.max_steps = env.steps + 10_000
        env.render = _null_render.__get__(env)  # skip drawing during simulation
        try:
            best_plan: Optional[list[int]] = None
            best_score = float("-inf")
            for plan in self._candidate_plans(direction):
                outcome = self._simulate(env, plan)
                restore_env_state(env, snapshot)
                if outcome.score > best_score:
                    best_score = outcome.score
                    best_plan = plan
                if outcome.reached_goal and outcome.steps <= self.replan_interval:
                    break  # cannot do better than reaching the goal immediately
            return list(best_plan or [NOOP])
        finally:
            del env.__dict__["render"]
            env.max_steps = original_max_steps
            restore_env_state(env, snapshot)

    def _simulate(self, env: MarioScenarioEnv, plan: list[int]) -> _PlanOutcome:
        best_safe_distance = self._goal_distance(env)
        steps = 0
        # The settle tail lets pending outcomes land: a plan that ends mid-air
        # over a pit must score as fatal, not as a survivor frozen mid-fall.
        for action in list(plan) + [NOOP] * self.settle_steps:
            _obs, _reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated:
                died = bool(info.get("death"))
                distance = self._goal_distance(env)
                if not died:
                    best_safe_distance = min(best_safe_distance, distance)
                return _PlanOutcome(
                    reached_goal=not died,
                    died=died,
                    steps=steps,
                    final_distance=distance,
                    best_safe_distance=best_safe_distance,
                )
            if env.mario["on_ground"]:
                best_safe_distance = min(best_safe_distance, self._goal_distance(env))
            if truncated:
                break
        return _PlanOutcome(
            reached_goal=False,
            died=False,
            steps=steps,
            final_distance=self._goal_distance(env),
            best_safe_distance=best_safe_distance,
        )

    @staticmethod
    def _goal_distance(env: MarioScenarioEnv) -> float:
        if env.goal is None:
            # No goal: farther right is better, mirroring the progress reward.
            return float(env.world_width - env.mario["x"])
        goal_x = env.goal.x + env.goal.w / 2.0
        goal_y = env.goal.y + env.goal.h / 2.0
        mario_x = env.mario["x"] + env.mario["w"] / 2.0
        mario_y = env.mario["y"] + env.mario["h"] / 2.0
        # Horizontal distance dominates; height matters for elevated goals.
        return abs(goal_x - mario_x) + 0.5 * abs(goal_y - mario_y)


def _null_render(self) -> None:  # noqa: ARG001 - bound as a method during planning
    return None


def evaluate_block_smb_geometry_expert(
    scenarios: Mapping[str, Mapping[str, Any]],
    *,
    max_steps: int = 400,
    replan_interval: int = DEFAULT_REPLAN_INTERVAL,
) -> dict[str, Any]:
    """Run the expert closed-loop on each scenario and report goal completion."""

    expert = BlockSMBGeometryExpert(replan_interval=replan_interval)
    results: dict[str, Any] = {}
    for name, scenario in scenarios.items():
        env = MarioScenarioEnv()
        expert.reset()
        try:
            env.reset(scenario=dict(scenario))
            env.max_steps = max_steps
            reached_goal = False
            died = False
            steps = 0
            for _ in range(max_steps):
                action = expert.action(env)
                _obs, _reward, terminated, truncated, info = env.step(action)
                steps += 1
                if terminated:
                    died = bool(info.get("death"))
                    reached_goal = not died
                    break
                if truncated:
                    break
        finally:
            env.close()
        results[name] = {"goal_reached": reached_goal, "died": died, "steps": steps}
    summary = {
        "scenarios": results,
        "success_count": sum(1 for r in results.values() if r["goal_reached"]),
        "scenario_count": len(results),
    }
    summary["success_rate"] = (
        summary["success_count"] / summary["scenario_count"] if results else 0.0
    )
    return summary


def _mario_rect(env: MarioScenarioEnv) -> pygame.Rect:
    return pygame.Rect(env.mario["x"], env.mario["y"], env.mario["w"], env.mario["h"])
