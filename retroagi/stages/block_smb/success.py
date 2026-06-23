"""Success thresholds for fixed Block SMB scenarios."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from retroagi.core import SMB_GAME_SPEC


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


def _fixed_block_smb_success_thresholds() -> dict[str, BlockSMBSuccessThreshold]:
    thresholds: dict[str, BlockSMBSuccessThreshold] = {}
    for task in SMB_GAME_SPEC.fixed_tasks:
        if task.stage_name != "block_smb" or task.success_threshold is None:
            continue
        threshold = task.success_threshold
        thresholds[task.name] = BlockSMBSuccessThreshold(
            scenario_name=task.name,
            min_success_rate=threshold.min_success_rate,
            min_mean_return=threshold.min_mean_return,
            min_episodes=threshold.min_episodes,
            max_steps=threshold.max_steps,
            rationale=threshold.rationale,
        )
    return thresholds


FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS = _fixed_block_smb_success_thresholds()


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


def summarize_fixed_success_metrics(
    fixed_scenario_results: Mapping[str, Mapping[str, float]],
    threshold_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Summarize deterministic fixed-scenario metrics for tuning comparisons.

    The scalar score intentionally orders threshold coverage before raw return:
    crossing a documented success threshold is more important than collecting a
    higher reward in one unsolved scenario.
    """
    scenario_names = [
        name
        for name in FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS
        if name in fixed_scenario_results
    ]
    if not scenario_names:
        return {
            "scenario_count": 0.0,
            "threshold_pass_rate": 0.0,
            "mean_success_rate": 0.0,
            "mean_return": 0.0,
            "score": 0.0,
        }

    threshold_pass_rate = sum(
        1.0
        for name in scenario_names
        if bool(threshold_results.get(name, {}).get("threshold_met", False))
    ) / len(scenario_names)
    mean_success_rate = sum(
        float(fixed_scenario_results[name].get("success_rate", 0.0))
        for name in scenario_names
    ) / len(scenario_names)
    mean_return = sum(
        float(fixed_scenario_results[name].get("return", 0.0))
        for name in scenario_names
    ) / len(scenario_names)
    score = threshold_pass_rate * 1_000_000.0 + mean_success_rate * 1_000.0 + mean_return
    return {
        "scenario_count": float(len(scenario_names)),
        "threshold_pass_rate": float(threshold_pass_rate),
        "mean_success_rate": float(mean_success_rate),
        "mean_return": float(mean_return),
        "score": float(score),
    }
