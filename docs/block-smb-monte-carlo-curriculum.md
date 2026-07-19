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

## Implementation Status

P3A is implemented for `block_smb_mc_v1`.

- checked-in fixed JSON scenarios remain regression sentinels;
- generated Block SMB scenarios now come from
  `retroagi.stages.block_smb.monte_carlo`;
- `--generated-scenarios` is preserved as a compatibility alias for Monte Carlo
  train samples;
- `retroagi-block-smb evaluate-monte-carlo` evaluates replayable held-out
  `train`, `validation`, `test`, or `stress` splits;
- Full SMB transfer requires fixed-scenario pass rate `1.0` and a passing
  held-out Monte Carlo validation gate in the source checkpoint metrics.

The legacy `MarioScenarioEnv.generate_scenario(...)` helper remains available
for low-level environment tests, but trainer-facing generated scenarios should
use the versioned sampler so checkpoints carry distribution evidence.

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
| `chained_obstacles` | multiple obstacle sections with enemies and pipes |
| `chained_enemy_gauntlet` | enemy, gap, patrol, and pipe sequence in one level |
| `full_smb_opening_proxy` | Block SMB approximation of the early Full SMB 1-1 demands |
| `mixed_section` | sampled composition of two or three families in one level |
| `tall_pipe_jump` | a single tall pipe (56-68px, taller than other pipe families) that must be mounted and cleared, staying under the jump-height ceiling so the scripted oracle remains reachable |

The first implementation should keep geometry ranges conservative enough that a
scripted bootstrap oracle can solve every sampled scenario. Harder ranges
should be added as named distribution versions instead of silently changing the
old one.

The scripted oracle is a bootstrap teacher, not the long-term architecture.
Future Block-level learning should be guided by the learned cross-game oracle
described in [Universal Retro Oracle Roadmap](universal-retro-oracle.md). The
Monte Carlo sampler should therefore record enough action, primitive, outcome,
confidence, and provenance metadata that scripted traces can become training
data for the universal oracle and later be replaced by learned labels.

## Parameter Schema

`BlockSMBScenarioFamilySpec` and `BlockSMBScenarioSample` provide:

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

Generated scenario dictionaries include the sampled parameters, generated world,
oracle actions, reachability result, and replay metadata under
`metadata.block_smb_monte_carlo`. Training summaries store compact manifests;
full generated scenarios are only preserved when an evaluation or debugging run
records them.

## Sampler And Splits

Sampling should be deterministic from `(distribution_id, split, seed,
sample_index)`.

- `train`: large Monte Carlo stream, reshuffled every epoch by seed.
- `validation`: stable held-out seeds for frequent evaluation and early stopping.
- `test`: stable held-out seeds used only for promotion reports.
- `stress`: intentionally difficult edge bins, reported separately from the main
  distribution score.

The sampler supports:

- uniform per-family sampling;
- weighted family sampling for curriculum stages;
- adaptive replay of recent failure bins;
- minimum coverage per family and difficulty bin;
- deterministic replay by scenario ID.

Routine fresh CLI runs use a failure-focused train sampler over `single_gap`,
`stair_climb`, `platform_chain`, `enemy_gap`, `retreat_recovery`, `wait_timing`,
`mixed_section`, and `full_smb_opening_proxy`. The Full SMB opening proxy is
weighted `4x`; the other weak families are weighted `1x`. Runs that pass
`--monte-carlo-family-weight` keep the exact requested sample count because
those are intentionally targeted curricula.

Training can enable failure replay with
`--monte-carlo-failure-replay-samples-per-epoch N`. After a Monte Carlo
validation run produces failure bins, later epochs sample additional train
scenarios weighted by the failing families. Fresh real-volume train CLI runs
default this replay budget to `64` samples per epoch after validation failures.
Held-out validation and test sampling stay unweighted so promotion gates still
measure broad family coverage.

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

## Gate Integrity

**Never force-pass the gates.** A gate failure means the source policy is not
ready, not that the gate is in the way. Do not use `--allow-ungated-block-source`
(or any equivalent bypass) to promote, transfer-for-keeps, or ship a policy; the
only legitimate use of that flag is a one-off measurement of an ungated policy's
Full SMB behavior, and its results must never be presented as a passing run.
When a gate fails, fix the Block SMB policy — train longer, improve coverage,
break action collapse — until it passes honestly. Reported "passes" must always
be real gate passes.

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

These numbers are starting points. The default code gates are configurable with
`--monte-carlo-pass-rate-gate` and `--monte-carlo-family-pass-rate-gate`.
Fresh `retroagi-block-smb train` and `retroagi-block-smb-distill` runs now use
the initial real-volume train/validation/test counts and the failure-focused
train family weights by default. Pass explicit `0` counts for a smoke run, or
`--monte-carlo-parameter-sweep` for the deterministic family/difficulty
coverage sweep.

Fresh `retroagi train --stage block` runs also default to a real training
budget: `200` epochs at `160` rollout steps (evaluating every `25` epochs),
roughly three orders of magnitude more environment frames and gradient updates
than the previous single-epoch/32-step shape, which was too short for Mario to
even reach most scenario goals. Pass explicit `--epochs`, `--rollout-steps`, or
`--evaluation-interval-epochs` to override, or `--monte-carlo-parameter-sweep`
for the tiny coverage sweep. Expect a fresh run to take hours on a single GPU.

## Commands

Train with versioned Monte Carlo samples:

```bash
retroagi-block-smb train \
  --monte-carlo-train-samples-per-epoch 512 \
  --monte-carlo-family-weight single_gap=1 \
  --monte-carlo-family-weight stair_climb=1 \
  --monte-carlo-family-weight platform_chain=1 \
  --monte-carlo-family-weight enemy_gap=1 \
  --monte-carlo-family-weight retreat_recovery=1 \
  --monte-carlo-family-weight wait_timing=1 \
  --monte-carlo-family-weight mixed_section=1 \
  --monte-carlo-family-weight full_smb_opening_proxy=4 \
  --monte-carlo-validation-samples 128 \
  --monte-carlo-test-samples 256 \
  --monte-carlo-failure-replay-samples-per-epoch 64 \
  --monte-carlo-pass-rate-gate 0.95 \
  --monte-carlo-family-pass-rate-gate 0.90 \
  --checkpoint data/block_smb/policy.pth \
  --output artifacts/block_smb/latest/run_summary.json
```

Evaluate a held-out split directly:

```bash
retroagi-block-smb evaluate-monte-carlo \
  --checkpoint data/block_smb/policy.pth \
  --split validation \
  --samples 128 \
  --output artifacts/block_smb/latest/mc_validation.json
```

Distill from sampled oracle trajectories:

```bash
retroagi-block-smb-distill \
  --checkpoint data/block_smb/distilled_mc.pth \
  --monte-carlo-samples 512 \
  --monte-carlo-family-weight single_gap=1 \
  --monte-carlo-family-weight stair_climb=1 \
  --monte-carlo-family-weight platform_chain=1 \
  --monte-carlo-family-weight enemy_gap=1 \
  --monte-carlo-family-weight retreat_recovery=1 \
  --monte-carlo-family-weight wait_timing=1 \
  --monte-carlo-family-weight mixed_section=1 \
  --monte-carlo-family-weight full_smb_opening_proxy=4 \
  --monte-carlo-validation-samples 128 \
  --monte-carlo-test-samples 256 \
  --monte-carlo-pass-rate-gate 0.95 \
  --monte-carlo-family-pass-rate-gate 0.90
```

For distillation, `--monte-carlo-samples` is the target total train volume.
Required family/difficulty coverage is included in that count and acts as a
floor, so requesting `512` yields 512 train scenarios rather than a 45-scenario
coverage sweep plus 512 more samples.

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
