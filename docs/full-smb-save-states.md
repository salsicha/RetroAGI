# Full SMB Save-State Artifacts

Full SMB save-state artifacts are generated locally from a legally provided SMB
ROM. The repository commits the deterministic recipes and expected paths, but
it does not commit emulator state bytes or any ROM-derived content.

The canonical plan lives in `retroagi.stages.full_smb.save_states`:

```python
from retroagi.stages.full_smb import full_smb_save_state_plan

plan = full_smb_save_state_plan()
print(plan.to_manifest()["categories"].keys())
for artifact in plan.artifacts:
    print(artifact.name, artifact.path, artifact.source_state)
```

## Artifact Categories

| Category | Purpose |
| --- | --- |
| `starting_position` | Reset snapshots used for smoke checks and simple level starts. |
| `benchmark` | Fixed benchmark starts used for repeatable evaluation reports. |
| `level_section` | Curriculum starts inside a level section, such as a pipe or flagpole approach. |
| `death_retry` | Terminal or near-terminal states used to verify death, retry, and episode-boundary handling. |

## Local Workflow

First verify the Full SMB backend and local content:

```bash
retroagi check-env --game smb --stage full \
  --seed 0 \
  --frame-skip 2 \
  --output artifacts/full_smb/env_check.json
```

Then write the recipe manifest:

```bash
python -m retroagi.stages.full_smb.save_states plan \
  --output local/full_smb/states/save_state_plan.json
```

Create one artifact for review:

```bash
python -m retroagi.stages.full_smb.save_states create \
  --only section_1_1_midpipe \
  --output-manifest local/full_smb/states/save_state_manifest.json \
  --overwrite
```

Create the whole local set after the first artifact is verified:

```bash
python -m retroagi.stages.full_smb.save_states create \
  --output-manifest local/full_smb/states/save_state_manifest.json \
  --overwrite
```

The generator resets the configured stable-retro level with the artifact seed,
uses `frame_skip=1`, replays the declared action script, then writes a local
pickle payload under `local/full_smb/states/`. The output manifest records the
artifact name, path, executed steps, total reward, terminal flags, final signal
info, observation checksum, and byte size.

Generated `.state` files are ROM-derived artifacts and must not be committed.

Only load save-state payloads that were generated locally from trusted content.
The payload format uses pickle because stable-retro state objects are
backend-native Python objects.

## Canonical Artifacts

| Artifact | Category | Path | Source | Linked Tasks |
| --- | --- | --- | --- | --- |
| `start_1_1_spawn` | `starting_position` | `local/full_smb/states/starts/1_1_spawn.state` | `Level1-1` | `smoke_1_1_spawn` |
| `benchmark_1_1_start` | `benchmark` | `local/full_smb/states/benchmark/1_1_start.state` | `Level1-1` | `benchmark_1_1_start` |
| `benchmark_1_2_start` | `benchmark` | `local/full_smb/states/benchmark/1_2_start.state` | `Level1-2` | `benchmark_1_2_start` |
| `benchmark_2_1_start` | `benchmark` | `local/full_smb/states/benchmark/2_1_start.state` | `Level2-1` | `benchmark_2_1_start` |
| `section_1_1_first_enemy_approach` | `level_section` | `local/full_smb/states/curriculum/1_1_first_enemy_approach.state` | `Level1-1` | obstacle-window labels |
| `section_1_1_midpipe` | `level_section` | `local/full_smb/states/curriculum/1_1_midpipe.state` | `Level1-1` | `curriculum_1_1_midpipe` |
| `section_1_1_flagpole_approach` | `level_section` | `local/full_smb/states/curriculum/1_1_flagpole_approach.state` | `Level1-1` | `curriculum_1_1_flagpole` |
| `section_1_2_underworld_opening` | `level_section` | `local/full_smb/states/curriculum/1_2_underworld_opening.state` | `Level1-2` | `curriculum_1_2_underworld` |
| `death_retry_1_1_first_gap` | `death_retry` | `local/full_smb/states/death_retry/1_1_first_gap.state` | `Level1-1` | episode-boundary tests |

## Review Checklist

Before using the artifacts for training or evaluation:

1. Confirm `check-env` passes for backend import, ROM availability, save/load
   state, frame stepping, and deterministic seeding.
2. Generate `save_state_plan.json` and preserve it with local run notes.
3. Generate each `.state` file under `local/full_smb/states/`.
4. Inspect `save_state_manifest.json` for unexpected early termination,
   unusually short step counts, or missing signal fields.
5. Do not commit `.state` files, ROMs, or generated screenshots derived from the
   ROM unless their licensing and provenance have been reviewed separately.

Full SMB fixed-task success thresholds that consume these task and save-state
names are documented in
[full-smb-success-thresholds.md](full-smb-success-thresholds.md).
