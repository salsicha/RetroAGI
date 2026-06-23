# Full SMB Task Sets

These are the supported Full SMB train/evaluation task sets for headless
training and reporting. They define task names, splits, reset seeds, episode
counts, maximum step budgets, and emulator start modes. They do not commit ROM
or save-state bytes.

The canonical catalog lives in `retroagi.stages.full_smb.tasks` and can be
inspected with:

```bash
python - <<'PY'
from retroagi.stages.full_smb import full_smb_task_catalog

for task_set, tasks in full_smb_task_catalog().to_manifest()["task_sets"].items():
    print(task_set)
    for task in tasks:
        start = task["start"]
        print(
            f"  {task['name']}: split={task['split']} "
            f"state={start['state']} start={start['mode']} "
            f"max_steps={task['max_steps']}"
        )
PY
```

## Task Sets

| Task Set | Split | Purpose |
| --- | --- | --- |
| `smoke` | `eval` | Short headless reset/step sanity checks after content setup. |
| `fixed_benchmark` | `eval` | Comparable seeded benchmark runs from stable-retro level starts. |
| `curriculum` | `train` | Ordered training rungs from simple starts to local save-state segments. |
| `heldout_generalization` | `heldout` | Levels withheld from tuning and used only for generalization checks. |

## Start Modes

`level_start` tasks pass a stable-retro state name such as `Level1-1` to the
environment. These tasks can run after the ROM is imported and
`retroagi check-env --game smb --stage full` passes.

`save_state_artifact` tasks name a local save-state artifact under
`local/full_smb/states/`. They also declare a stable-retro base state so the
emulator can be created before the local save state is loaded. The deterministic
local save-state recipes are documented in
[full-smb-save-states.md](full-smb-save-states.md). The repository commits the
recipes and expected paths, not ROM-derived emulator state bytes.

## Catalog Summary

| Name | Task Set | Split | Start | Seed | Episodes | Max Steps | Goal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `smoke_1_1_spawn` | `smoke` | `eval` | `Level1-1` | `210001` | `1` | `128` | Verify reset, stepping, signals, and short-horizon survival. |
| `benchmark_1_1_start` | `fixed_benchmark` | `eval` | `Level1-1` | `220101` | `3` | `2400` | Measure early-game progress, survival, score, and completion from 1-1. |
| `benchmark_1_2_start` | `fixed_benchmark` | `eval` | `Level1-2` | `220102` | `3` | `2400` | Measure transfer under underground visuals and tighter terrain. |
| `benchmark_2_1_start` | `fixed_benchmark` | `eval` | `Level2-1` | `220201` | `3` | `2600` | Measure progress on a later overworld level with denser hazards. |
| `curriculum_1_1_opening` | `curriculum` | `train` | `Level1-1` | `230101` | `4` | `600` | Learn stable rightward movement, jumping, and survival from spawn. |
| `curriculum_1_1_midpipe` | `curriculum` | `train` | `local/full_smb/states/curriculum/1_1_midpipe.state` | `230102` | `4` | `900` | Train obstacle timing after the opening segment is stable. |
| `curriculum_1_1_flagpole` | `curriculum` | `train` | `local/full_smb/states/curriculum/1_1_flagpole_approach.state` | `230103` | `4` | `900` | Train level-completion behavior after progress and survival work. |
| `curriculum_1_2_underworld` | `curriculum` | `train` | `Level1-2` | `230201` | `4` | `1200` | Adapt the transferred policy to underground visuals and spacing. |
| `heldout_2_2_water` | `heldout_generalization` | `heldout` | `Level2-2` | `240202` | `3` | `2600` | Measure generalization to water physics and non-overworld visuals. |
| `heldout_3_1_bridge` | `heldout_generalization` | `heldout` | `Level3-1` | `240301` | `3` | `2600` | Measure generalization to elevated terrain and bridge-like pacing. |
| `heldout_8_1_long` | `heldout_generalization` | `heldout` | `Level8-1` | `240801` | `3` | `3200` | Measure late-game generalization without tuning on this task. |

## Promotion Discipline

Use the sets in this order:

1. Run `smoke` after content setup and `check-env`.
2. Generate and review the local save-state artifacts.
3. Train with the ordered `curriculum` tasks.
4. Evaluate repeatedly on `fixed_benchmark` tasks.
5. Touch `heldout_generalization` only for promotion or regression reports.

Success thresholds are intentionally not defined in this document. The next
threshold task sets the required progress, completion, survival, score/coins,
time budget, death count, and return gates for the fixed benchmark tasks.
