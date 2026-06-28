# Block SMB Monte Carlo Curriculum Plan

Block SMB is meant to do most policy, world-model, and motor-controller
training in a fast environment with exact semantic labels, exact symbolic state,
and deterministic physics. Fixed scenarios are useful sentinels, but they are
not enough training data for transfer: a policy can memorize nine or twelve
layouts and still fail as soon as Full SMB asks for slightly different timing,
spacing, enemy approach, or recovery behavior.

The target Block SMB curriculum is therefore a versioned parameterized scenario
distribution. Training should draw Monte Carlo samples from that distribution;
promotion should require both fixed-scenario success and held-out distribution
success.

## Current Limitation

The current code has two scenario sources:

- checked-in fixed JSON scenarios under `retroagi/stages/block_smb/scenarios/`;
- `MarioScenarioEnv.generate_scenario(...)`, which samples a loose random level
  from a few scalar ranges.

That is not yet a sufficient distribution contract. It lacks named scenario
families, train/validation/test splits, coverage reporting, difficulty bins,
held-out seeds, per-family thresholds, and artifact metadata that proves what
distribution a checkpoint saw.

## Design Goals

- Keep Block SMB as the high-volume, ground-truth training rung.
- Preserve fixed scenarios as deterministic regression tests and transfer
  sentinels.
- Add a parameterized scenario schema that can express the full simplified SMB
  distribution: terrain, gaps, stairs, moving platforms, enemies, coins, goals,
  recovery situations, and timing hazards.
- Use Monte Carlo sampling to cover many combinations of geometry and dynamics
  without committing every generated level.
- Record enough metadata that a checkpoint can be traced to a distribution
  version, seed policy, split, sample count, coverage histogram, and failure
  bins.
- Promote to Full SMB only when the policy passes fixed scenarios and held-out
  Monte Carlo scenarios.

## Scenario Families

Each generated sample should belong to a named family. Initial families should
mirror the fixed scenarios and then add interpolation/extrapolation ranges:

| Family | Parameters |
| --- | --- |
| `flat_run` | world width, floor height, coin spacing, goal distance |
| `single_gap` | gap x-position, gap width, approach length, landing width |
| `stair_climb` | step count, step width, step height, climb direction |
| `platform_chain` | platform count, width, height, gap spacing, vertical variance |
| `moving_bridge` | platform speed, travel range, phase, gap width |
| `enemy_hop` | enemy x-position, speed, patrol width, approach distance |
| `enemy_patrol` | enemy count, spacing, patrol overlap, speed variance |
| `enemy_gap` | enemy placement relative to gap, gap width, landing zone |
| `enemy_stomp` | enemy height/position, stomp window, safe landing distance |
| `retreat_recovery` | left/right recovery need, obstacle proximity, safe fallback |
| `wait_timing` | moving-platform phase, wait window, jump window |
| `mixed_section` | sampled composition of two or three families in one level |

The first implementation should keep geometry ranges conservative enough that a
scripted oracle can solve every sampled scenario. Harder ranges should be added
as named distribution versions instead of silently changing the old one.

## Parameter Schema

Add a `BlockSMBScenarioFamilySpec` style contract with:

- `schema_version`;
- `distribution_id`, for example `block_smb_mc_v1`;
- `family`;
- `split`: `train`, `validation`, `test`, or `stress`;
- `seed`;
- geometry ranges: world width, floor height, gap widths, platform sizes,
  vertical offsets, goal distance;
- entity ranges: enemy count, patrol bounds, speed, edge awareness, coins;
- timing ranges: moving-platform phase, speed, wait window, jump window;
- constraints: reachable path, spawn safety, minimum landing width, max
  impossible gap, max enemy density;
- oracle metadata: scripted action source, expected completion range, expected
  minimum progress.

Generated scenario JSON should include the sampled parameters and the generated
world. Training summaries should store compact manifests, not full generated
datasets unless an experiment intentionally preserves samples for debugging.

## Sampler And Splits

Sampling should be deterministic from `(distribution_id, split, seed,
sample_index)`.

- `train`: large Monte Carlo stream, reshuffled every epoch by seed.
- `validation`: stable held-out seeds for frequent evaluation and early stopping.
- `test`: stable held-out seeds used only for promotion reports.
- `stress`: intentionally difficult edge bins, reported separately from the main
  distribution score.

The sampler should support:

- uniform per-family sampling;
- weighted family sampling for curriculum stages;
- adaptive replay of recent failure bins;
- minimum coverage per family and difficulty bin;
- deterministic replay by scenario ID.

## Curriculum Schedule

Training should progress by distribution coverage, not only by fixed-scenario
names.

1. **Oracle-verifiable easy distribution:** conservative geometry, one obstacle
   family per sample, no dense combinations.
2. **Family-balanced distribution:** all families sampled with moderate
   variation.
3. **Mixed-section distribution:** two or three obstacle families composed in
   one level.
4. **Hard-bin replay:** oversample failed bins while keeping a background
   stream from the full distribution.
5. **Held-out validation/test gates:** evaluate without replay bias.

The policy should continue to see fixed scenarios during training, but those
scenarios should be a small sentinel fraction rather than the whole curriculum.

## Metrics And Gates

Block SMB promotion to Full SMB should require:

- fixed-scenario threshold pass rate `1.0`;
- held-out Monte Carlo validation pass rate above the configured gate;
- held-out Monte Carlo test pass rate reported in the promotion artifact;
- per-family pass rates above minimum family gates;
- no missing coverage bins for the selected distribution version;
- world-model dynamics metrics reported by C-stream slot on Monte Carlo samples;
- action distribution diagnostics for required actions such as `LEFT`,
  `RIGHT_JUMP`, wait/release behavior, and recovery primitives.

Suggested initial gates for `block_smb_mc_v1`:

- train samples per epoch: at least `512`;
- validation samples: at least `128`;
- test samples: at least `256`;
- validation pass rate: at least `0.95`;
- per-family pass rate: at least `0.90`;
- fixed-scenario pass rate: exactly `1.0`.

These numbers are starting points. They should be tuned after the first full
Monte Carlo implementation produces failure histograms.

## Implementation Steps

1. Add a versioned scenario-family schema and distribution config.
2. Replace the loose `generated_scenarios` path with a sampler that emits
   scenario IDs, family names, parameters, split names, and seeds.
3. Add reachability/oracle checks so invalid generated levels are rejected
   before training.
4. Add deterministic train/validation/test split generation and replay by
   scenario ID.
5. Add coverage histograms to training logs and run summaries.
6. Add Monte Carlo validation and test evaluation commands.
7. Add per-family thresholds and promotion gates.
8. Update Block SMB distillation so teacher/oracle traces can be generated from
   sampled scenarios, not only fixed scripted scenarios.
9. Update action diagnostics to aggregate failures by scenario family and
   parameter bin.
10. Require fixed plus Monte Carlo gates before a Block SMB checkpoint is used
    as the Full SMB transfer source.

## Artifact Requirements

Every Monte Carlo Block SMB run should record:

- distribution ID and schema version;
- train/validation/test split names;
- base seed and sample-count policy;
- family weights and curriculum schedule;
- sampled coverage histograms;
- rejected-sample counts and rejection reasons;
- per-family metrics;
- fixed-scenario metrics;
- held-out validation/test metrics;
- checkpoint path and source checkpoint provenance.

This makes the simplified rung useful for rapid iteration: most concepts can be
rejected in Block SMB with exact ground truth before Full SMB emulator time is
spent.
