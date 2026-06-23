# Full SMB Success Thresholds

These thresholds define when a Full SMB policy is considered successful on the
fixed benchmark tasks used by P9 evaluation. They are intentionally stricter
than observing a valid emulator rollout: each task has gates for progress,
completion, survival, score/coins, death count, return, required episodes, and
step budget.

In short, every fixed task is judged by progress, completion, survival, score/coins,
time budget, death count, and minimum return.

The machine-readable source of truth is
`retroagi.stages.full_smb.success.FIXED_FULL_SMB_SUCCESS_THRESHOLDS`.

## Evaluation Protocol

Use deterministic evaluation with:

- the `fixed_benchmark` task set from [full-smb-tasks.md](full-smb-tasks.md),
- the Full SMB ViT checkpoint selected for policy inference,
- at least the threshold's required episode count,
- no more than the threshold's max steps per episode,
- local save-state artifacts generated from
  [full-smb-save-states.md](full-smb-save-states.md) when a task uses them.

An evaluation with fewer episodes or a larger per-episode step budget can still
be useful for debugging, but it does not qualify as a fixed-task success run.

## Fixed Benchmark Thresholds

| Task | Progress | Completion | Survival | Score | Coins | Max Deaths | Return | Episodes | Max Steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `benchmark_1_1_start` | `3200` | `1.0` | `1.0` | `500` | `0` | `0` | `0.0` | `3` | `2400` |
| `benchmark_1_2_start` | `2800` | `0.667` | `0.667` | `500` | `0` | `1` | `0.0` | `3` | `2400` |
| `benchmark_2_1_start` | `2400` | `0.333` | `0.667` | `250` | `0` | `1` | `0.0` | `3` | `2600` |

## Metric Meanings

`progress` is the maximum x/progress signal observed across deterministic
episodes. `completion` is the fraction of episodes that end with a flag/level
completion signal. `survival` is the fraction that avoid death. `score` and
`coins` are mean final values across episodes. `max_deaths` is the maximum total
death count permitted across the evaluated episodes. `return` is mean scalar
reward from the Full SMB adapter.

The fixed thresholds are staged:

- `benchmark_1_1_start` is the solved-baseline gate and requires completing all
  deterministic episodes with no deaths.
- `benchmark_1_2_start` validates transfer to underground visuals and tighter
  spacing by requiring two of three clears and no more than one death.
- `benchmark_2_1_start` validates later-world transfer by requiring meaningful
  progress, mostly surviving episodes, and at least one clear.

## How To Check A Run

The diagnostic helper accepts the metric names produced by current and planned
Full SMB evaluation code:

```python
from retroagi.stages.full_smb import evaluate_full_smb_success_threshold

diagnostics = evaluate_full_smb_success_threshold(
    "benchmark_1_1_start",
    {
        "max_progress": 3300,
        "completion_rate": 1.0,
        "survival_rate": 1.0,
        "mean_score": 900,
        "mean_coins": 0,
        "death_count": 0,
        "mean_return": 1.0,
    },
    evaluation_episodes=3,
    evaluation_max_steps=2400,
)
assert diagnostics["threshold_met"]
```

Each fixed task diagnostic includes:

- `threshold`: the task threshold,
- `observed`: the normalized metric inputs used for comparison,
- booleans for progress, completion, survival, score, coins, death budget,
  return, episode count, and step-budget checks,
- `threshold_met`, which is true only when every gate passes.

`summarize_fixed_full_smb_success_metrics(...)` returns a tuning summary whose
score orders threshold coverage before completion, survival, progress, and raw
return. A policy with impressive progress but failed fixed-task thresholds ranks
below one that crosses the documented gates.

## Relationship To Future Signal Extraction

These thresholds define the contract before the next P9 task expands Full SMB signal extraction
with memory variables and backend info fields. Once x/y
position, score, coins, lives, power state, death, timeout, completion, and
game-over signals are extracted more completely, the evaluator should feed
those fields into this threshold module rather than creating a separate success
definition.
