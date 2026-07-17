"""Multi-seed evaluation aggregation.

Promotion gates and benchmark comparisons in this repo have historically
compared single-seed point estimates, which makes both false promotions and
false regressions cheap. These helpers run an evaluation callable across
several seeds and report mean/std/min/max per metric so go/no-go decisions can
weigh dispersion, not a single draw.
"""

import math
from typing import Any, Callable, Mapping, Sequence

DEFAULT_EVALUATION_SEED_COUNT = 3


def evaluation_seeds(base_seed: int, seed_count: int) -> tuple[int, ...]:
    """Deterministic seed ladder derived from a base seed."""

    if seed_count <= 0:
        raise ValueError("seed_count must be positive")
    return tuple(int(base_seed) + 1000 * index for index in range(seed_count))


def aggregate_seed_metrics(
    per_seed_metrics: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    """Aggregate numeric metrics across seeds into mean/std/min/max/count.

    Only keys whose values are numeric in every run are aggregated; ``std`` is
    the population standard deviation (N, not N-1 — the seeds are the whole
    set being described, not a sample of a larger one).
    """

    if not per_seed_metrics:
        return {}
    numeric_keys = None
    for metrics in per_seed_metrics:
        keys = {
            key
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        numeric_keys = keys if numeric_keys is None else numeric_keys & keys
    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(numeric_keys or ()):
        values = [float(metrics[key]) for metrics in per_seed_metrics]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        aggregated[key] = {
            "mean": mean,
            "std": math.sqrt(variance),
            "min": min(values),
            "max": max(values),
            "count": float(len(values)),
        }
    return aggregated


def evaluate_over_seeds(
    evaluate: Callable[[int], Mapping[str, Any]],
    *,
    base_seed: int,
    seed_count: int = DEFAULT_EVALUATION_SEED_COUNT,
) -> dict[str, Any]:
    """Call ``evaluate(seed)`` for each seed and aggregate its numeric metrics.

    ``evaluate`` must return a flat mapping of metric name to value for one
    seed. The result carries the per-seed payloads and the aggregates.
    """

    seeds = evaluation_seeds(base_seed, seed_count)
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        metrics = dict(evaluate(seed))
        metrics["seed"] = int(seed)
        per_seed.append(metrics)
    return {
        "seeds": list(seeds),
        "seed_count": len(seeds),
        "per_seed": per_seed,
        "aggregate": aggregate_seed_metrics(per_seed),
    }
