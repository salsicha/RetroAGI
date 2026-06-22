"""Success thresholds for fixed Block SMB scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class BlockSMBSuccessThreshold:
    """Deterministic evaluation threshold for one fixed scenario."""

    scenario_name: str
    min_success_rate: float
    min_mean_return: float
    min_episodes: int
    max_steps: int
    rationale: str

    def __post_init__(self) -> None:
        if not self.scenario_name:
            raise ValueError("scenario_name must be non-empty")
        if not 0.0 <= self.min_success_rate <= 1.0:
            raise ValueError("min_success_rate must be in [0, 1]")
        if self.min_episodes <= 0:
            raise ValueError("min_episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS: dict[str, BlockSMBSuccessThreshold] = {
    "level_1_flat.json": BlockSMBSuccessThreshold(
        scenario_name="level_1_flat.json",
        min_success_rate=1.0,
        min_mean_return=55.0,
        min_episodes=3,
        max_steps=200,
        rationale="Flat run: reach the goal reliably without relying on one lucky rollout.",
    ),
    "level_2_gap.json": BlockSMBSuccessThreshold(
        scenario_name="level_2_gap.json",
        min_success_rate=1.0,
        min_mean_return=55.0,
        min_episodes=3,
        max_steps=200,
        rationale="Gap run: cross the gap and reach the goal reliably within the time budget.",
    ),
    "level_3_stairs.json": BlockSMBSuccessThreshold(
        scenario_name="level_3_stairs.json",
        min_success_rate=1.0,
        min_mean_return=55.0,
        min_episodes=3,
        max_steps=200,
        rationale="Stair run: climb the stepped platforms and reach the elevated goal.",
    ),
    "level_4_platforms.json": BlockSMBSuccessThreshold(
        scenario_name="level_4_platforms.json",
        min_success_rate=1.0,
        min_mean_return=55.0,
        min_episodes=3,
        max_steps=200,
        rationale="Platform run: traverse separated platforms and reach the final goal.",
    ),
}


def fixed_scenario_success_threshold(name: str) -> BlockSMBSuccessThreshold:
    try:
        return FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS[name]
    except KeyError as exc:
        raise KeyError(f"unknown fixed Block SMB scenario {name!r}") from exc


def evaluate_success_threshold(
    scenario_name: str,
    result: Mapping[str, float],
    *,
    evaluation_episodes: int,
    evaluation_max_steps: int,
) -> dict[str, Any]:
    """Return threshold diagnostics for one fixed scenario result."""
    threshold = fixed_scenario_success_threshold(scenario_name)
    success_rate = float(result.get("success_rate", 0.0))
    mean_return = float(result.get("return", 0.0))
    enough_episodes = evaluation_episodes >= threshold.min_episodes
    within_step_budget = evaluation_max_steps <= threshold.max_steps
    meets_success_rate = success_rate >= threshold.min_success_rate
    meets_return = mean_return >= threshold.min_mean_return
    threshold_met = (
        enough_episodes
        and within_step_budget
        and meets_success_rate
        and meets_return
    )
    return {
        "threshold": threshold.to_dict(),
        "evaluation_episodes": evaluation_episodes,
        "evaluation_max_steps": evaluation_max_steps,
        "enough_episodes": enough_episodes,
        "within_step_budget": within_step_budget,
        "meets_success_rate": meets_success_rate,
        "meets_return": meets_return,
        "threshold_met": threshold_met,
    }


def evaluate_fixed_success_thresholds(
    fixed_scenario_results: Mapping[str, Mapping[str, float]],
    *,
    evaluation_episodes: int,
    evaluation_max_steps: int,
) -> dict[str, dict[str, Any]]:
    """Return threshold diagnostics for all fixed scenario results."""
    return {
        scenario_name: evaluate_success_threshold(
            scenario_name,
            result,
            evaluation_episodes=evaluation_episodes,
            evaluation_max_steps=evaluation_max_steps,
        )
        for scenario_name, result in fixed_scenario_results.items()
        if scenario_name in FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS
    }
