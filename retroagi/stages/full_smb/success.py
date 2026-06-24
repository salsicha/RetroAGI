"""Success thresholds for fixed Full SMB benchmark tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from retroagi.stages.full_smb.tasks import full_smb_task_catalog


@dataclass(frozen=True)
class FullSMBSuccessThreshold:
    """Deterministic evaluation threshold for one fixed Full SMB task."""

    task_name: str
    min_progress: float
    min_completion_rate: float
    min_survival_rate: float
    min_mean_score: float
    min_mean_coins: float
    max_deaths: int
    min_mean_return: float
    min_episodes: int
    max_steps: int
    rationale: str

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name must be non-empty")
        if self.min_progress < 0:
            raise ValueError("min_progress must be non-negative")
        for name in ("min_completion_rate", "min_survival_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("min_mean_score", "min_mean_coins"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.max_deaths < 0:
            raise ValueError("max_deaths must be non-negative")
        if self.min_episodes <= 0:
            raise ValueError("min_episodes must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not self.rationale:
            raise ValueError("rationale must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


FIXED_FULL_SMB_SUCCESS_THRESHOLDS: dict[str, FullSMBSuccessThreshold] = {
    "benchmark_1_1_start": FullSMBSuccessThreshold(
        task_name="benchmark_1_1_start",
        min_progress=3200.0,
        min_completion_rate=1.0,
        min_survival_rate=1.0,
        min_mean_score=500.0,
        min_mean_coins=0.0,
        max_deaths=0,
        min_mean_return=0.0,
        min_episodes=3,
        max_steps=2400,
        rationale=(
            "Level 1-1 should be a solved baseline: complete every deterministic "
            "episode, survive, and reach the flag within the fixed budget."
        ),
    ),
    "benchmark_1_2_start": FullSMBSuccessThreshold(
        task_name="benchmark_1_2_start",
        min_progress=2800.0,
        min_completion_rate=2.0 / 3.0,
        min_survival_rate=2.0 / 3.0,
        min_mean_score=500.0,
        min_mean_coins=0.0,
        max_deaths=1,
        min_mean_return=0.0,
        min_episodes=3,
        max_steps=2400,
        rationale=(
            "Level 1-2 validates transfer to underground visuals and tighter "
            "spacing; at least two of three deterministic episodes must clear."
        ),
    ),
    "benchmark_2_1_start": FullSMBSuccessThreshold(
        task_name="benchmark_2_1_start",
        min_progress=2400.0,
        min_completion_rate=1.0 / 3.0,
        min_survival_rate=2.0 / 3.0,
        min_mean_score=250.0,
        min_mean_coins=0.0,
        max_deaths=1,
        min_mean_return=0.0,
        min_episodes=3,
        max_steps=2600,
        rationale=(
            "Level 2-1 is a later-world transfer benchmark: require meaningful "
            "progress, mostly surviving episodes, and at least one clear."
        ),
    ),
}


def fixed_full_smb_success_threshold(task_name: str) -> FullSMBSuccessThreshold:
    try:
        return FIXED_FULL_SMB_SUCCESS_THRESHOLDS[task_name]
    except KeyError as exc:
        raise KeyError(f"unknown fixed Full SMB task {task_name!r}") from exc


def evaluate_full_smb_success_threshold(
    task_name: str,
    result: Mapping[str, float],
    *,
    evaluation_episodes: int,
    evaluation_max_steps: int,
) -> dict[str, Any]:
    """Return threshold diagnostics for one fixed Full SMB task result."""

    threshold = fixed_full_smb_success_threshold(task_name)
    progress = _metric(result, "max_progress", "progress", "mean_progress")
    completion_rate = _metric(result, "completion_rate", "success_rate")
    survival_rate = _metric(result, "survival_rate")
    mean_score = _metric(result, "mean_score", "score")
    mean_coins = _metric(result, "mean_coins", "coins")
    deaths = _metric(result, "death_count", "deaths")
    mean_return = _metric(result, "mean_return", "return")

    enough_episodes = evaluation_episodes >= threshold.min_episodes
    within_step_budget = evaluation_max_steps <= threshold.max_steps
    meets_progress = progress >= threshold.min_progress
    meets_completion = completion_rate >= threshold.min_completion_rate
    meets_survival = survival_rate >= threshold.min_survival_rate
    meets_score = mean_score >= threshold.min_mean_score
    meets_coins = mean_coins >= threshold.min_mean_coins
    within_death_budget = deaths <= threshold.max_deaths
    meets_return = mean_return >= threshold.min_mean_return
    threshold_met = (
        enough_episodes
        and within_step_budget
        and meets_progress
        and meets_completion
        and meets_survival
        and meets_score
        and meets_coins
        and within_death_budget
        and meets_return
    )

    return {
        "threshold": threshold.to_dict(),
        "evaluation_episodes": evaluation_episodes,
        "evaluation_max_steps": evaluation_max_steps,
        "observed": {
            "progress": progress,
            "completion_rate": completion_rate,
            "survival_rate": survival_rate,
            "mean_score": mean_score,
            "mean_coins": mean_coins,
            "deaths": deaths,
            "mean_return": mean_return,
        },
        "enough_episodes": enough_episodes,
        "within_step_budget": within_step_budget,
        "meets_progress": meets_progress,
        "meets_completion": meets_completion,
        "meets_survival": meets_survival,
        "meets_score": meets_score,
        "meets_coins": meets_coins,
        "within_death_budget": within_death_budget,
        "meets_return": meets_return,
        "threshold_met": threshold_met,
    }


def evaluate_fixed_full_smb_success_thresholds(
    fixed_task_results: Mapping[str, Mapping[str, float]],
    *,
    evaluation_episodes: int,
    evaluation_max_steps: int,
) -> dict[str, dict[str, Any]]:
    """Return threshold diagnostics for all fixed Full SMB task results."""

    return {
        task_name: evaluate_full_smb_success_threshold(
            task_name,
            result,
            evaluation_episodes=evaluation_episodes,
            evaluation_max_steps=evaluation_max_steps,
        )
        for task_name, result in fixed_task_results.items()
        if task_name in FIXED_FULL_SMB_SUCCESS_THRESHOLDS
    }


def summarize_fixed_full_smb_success_metrics(
    fixed_task_results: Mapping[str, Mapping[str, float]],
    threshold_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    """Summarize Full SMB fixed-task metrics for tuning comparisons."""

    task_names = [name for name in FIXED_FULL_SMB_SUCCESS_THRESHOLDS if name in fixed_task_results]
    if not task_names:
        return {
            "task_count": 0.0,
            "threshold_pass_rate": 0.0,
            "mean_completion_rate": 0.0,
            "mean_survival_rate": 0.0,
            "mean_progress": 0.0,
            "mean_return": 0.0,
            "score": 0.0,
        }

    threshold_pass_rate = sum(
        1.0
        for name in task_names
        if bool(threshold_results.get(name, {}).get("threshold_met", False))
    ) / len(task_names)
    mean_completion_rate = _mean_metric(
        task_names,
        fixed_task_results,
        "completion_rate",
    )
    mean_survival_rate = _mean_metric(task_names, fixed_task_results, "survival_rate")
    mean_progress = sum(
        _metric(fixed_task_results[name], "max_progress", "progress", "mean_progress")
        for name in task_names
    ) / len(task_names)
    mean_return = _mean_metric(task_names, fixed_task_results, "mean_return", "return")
    score = (
        threshold_pass_rate * 1_000_000.0
        + mean_completion_rate * 10_000.0
        + mean_survival_rate * 1_000.0
        + mean_progress
        + mean_return
    )
    return {
        "task_count": float(len(task_names)),
        "threshold_pass_rate": float(threshold_pass_rate),
        "mean_completion_rate": float(mean_completion_rate),
        "mean_survival_rate": float(mean_survival_rate),
        "mean_progress": float(mean_progress),
        "mean_return": float(mean_return),
        "score": float(score),
    }


def _validate_threshold_catalog() -> None:
    fixed_tasks = {
        task.name: task for task in full_smb_task_catalog().tasks_for_set("fixed_benchmark")
    }
    missing = sorted(set(fixed_tasks).difference(FIXED_FULL_SMB_SUCCESS_THRESHOLDS))
    extra = sorted(set(FIXED_FULL_SMB_SUCCESS_THRESHOLDS).difference(fixed_tasks))
    if missing:
        raise ValueError(f"missing Full SMB success thresholds: {missing}")
    if extra:
        raise ValueError(f"unknown Full SMB success thresholds: {extra}")
    for task_name, threshold in FIXED_FULL_SMB_SUCCESS_THRESHOLDS.items():
        task = fixed_tasks[task_name]
        if threshold.min_episodes > task.episodes:
            raise ValueError(f"Full SMB threshold {task_name!r} requires more episodes than task")
        if threshold.max_steps != task.max_steps:
            raise ValueError(f"Full SMB threshold {task_name!r} must use the task max_steps")


def _metric(result: Mapping[str, float], *names: str) -> float:
    for name in names:
        if name in result:
            return float(result[name])
    return 0.0


def _mean_metric(
    task_names: list[str],
    fixed_task_results: Mapping[str, Mapping[str, float]],
    *names: str,
) -> float:
    return sum(_metric(fixed_task_results[name], *names) for name in task_names) / len(task_names)


_validate_threshold_catalog()
