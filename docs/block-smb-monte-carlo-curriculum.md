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
| `tall_pipe_jump` | composite traversal: approach and mount a 56–68px pipe, then descend and touch the ground-level finish; mounting and episode completion are reported separately |
| `pit_leap` | B-level isolation family: jump over a pit whose width bands (40-66px) make the required hold grow monotonically; A-level given, goal-distance shaping, jump-energy cost |
| `stomp_mount` | single-jump interception with a supplied A-level jump intent; stationary easy targets, moving medium/hard targets with varied direction and patrol phase; actual collision geometry determines stomp credit and coaching |
| `bridge_wait` | B-teaching family for the WAIT decision: the opening action is given (forced NOOP, first primitive only), the event-terminated wait releases when the bridge arrives, and the policy crosses on its own. Waits grow 8-30 frames across tiers (hard exceeds the old 16-frame menu); wait-survival shaping and hindsight wait coaching apply |
| `platform_hop` | B-level isolation family: jump onto a narrow slow-moving platform over a pit (68-110px bands) and ride it to the far ledge; A-level given, shaping, energy cost |
| `pipe_mount` | B-level isolation family: the A-level decision is given (`a_level_action` forces RIGHT_JUMP through the rollout), the goal sits on the pipe top (42-66px, disjoint height bands per difficulty), and a per-scenario goal-distance shaping reward gives a dense vertical-progress gradient, so only the B-level jump parameters (hold duration versus pipe height) remain to be learned |

`tall_pipe_jump` training and success rehearsal use at least 160 frames, even
when `rollout_steps` is smaller. Its oracle requires 82–86 frames; the former
60-frame training limit prevented any successful episode. The epoch metrics
`training_rollout_steps_max` and `training_rollout_budget_extensions` expose the
effective budget. Evaluation continues to honor `evaluation_max_steps` exactly.

The mounting jump is coached toward the pipe top, with its target retained
through landing. Actual support on the pipe anchors a successful hold and
prevents an erroneous `jump_overreach` penalty. Reaching the pipe horizontally
without sufficient height instead requests a longer hold. After mounting (or
passing the pipe), the mount goal encoding is cleared and the final goal remains
available in the symbolic state. Actions remain policy decisions; walking after
mounting is not forced. Retreating off the approach side reactivates mounting.
The family also opts into goal-distance reward shaping to provide feedback when
approaching or overshooting the finish.

Scenario, difficulty, and family evaluation records include `pipe_metrics`:
`mount_success_rate` measures actual pipe support, and
`finish_after_mount_success_rate` measures episode completion among attempts
that mounted (`null` if none did). Counts are included so aggregation weights
episodes correctly. The existing `success_rate` still requires touching the
original goal rectangle. Existing checkpoints load unchanged, but these training
changes require further training before improved learned accuracy can be claimed.


`stomp_mount` samples now carry `parameters.family_revision: 2`. Distance bands
remain 52–60 / 62–68 / 70–76px. Medium and hard patrols move at 0.6 / 0.9px per
frame with 20 / 28px total travel, start in either direction, and first reverse
about 6–12 frames after takeoff, while release can still affect the jump. The
sample records initial direction and nominal frames to the first turn. Each
sample's scripted oracle is calibrated against the engine's 1–16-frame hold
menu and checked for reachability. Evaluation still uses the learned policy.
Saved older scenarios remain replayable; compare revision 2 results separately
from the previous patrol distribution, whose reversals came after the jump.
Variation exercises interception across motion phases; reachability alone does
not establish that every sample requires mid-flight adaptation.

The engine and coaching share the stomp predicate: integer collision rectangles
must overlap while Mario descends, with his inferred previous bottom at or above
the enemy's center. Recorded rectangles and pre-bounce velocity are re-evaluated
for coaching. Valid off-center stomps anchor the executed hold; misses use the
horizontal gap during the descending contact window to request one bin longer
or shorter. Every frame in the arc receives the same contact-time target. A
historical enemy position is never compared with Mario's later landing position,
and an unfinished arc without contact-window evidence receives no hold label.

Duration coaching uses categorical cross-entropy on the bin the controller
samples or selects by argmax; fitting only the distribution's mean could leave
the executed mode unchanged. Wait targets account for their four-frame bin
scale. Supplied A-level intent conditions B, the world model, and motor decoding
before prediction. It receives no on-policy actor credit and cannot be replaced
by critic candidate search. Oracle demonstrations retain actor supervision and
one jump span through landing, without off-policy REINFORCE credit.

Stomp scenarios disable positive-x progress reward while retaining target-distance
shaping. A completed single-jump miss ends with the enemy-hit penalty and a
terminal training mask. Scenario, difficulty, and family evaluation records
include `stomp_outcome_counts`: `success`, `collision`, `undershoot`, `overshoot`,
`no_contact`, `environment_timeout`, or `budget_timeout`. Temporal spans retain
those failure categories, so a safe miss is no longer reported as an evaluator
timeout. These changes add no model parameters; checkpoint loading remains
compatible, but improved learned accuracy requires retraining and evaluation.


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

### Mastery-Gated Schedule

Fresh real-volume `retroagi train --stage block` runs use the mastery-gated
schedule by default (`--mastery-gated-schedule`, disable with
`--no-mastery-gated-schedule`). The intent of Block SMB is that each family is
a small constrained skill; the schedule makes the trainer actually run each
family until the model learns that skill:

- Every family always has a nonzero sampling weight, so no gated family can be
  silently excluded from training.
- After each periodic held-out evaluation, per-family pass rates update a
  mastery record. Unmastered families are weighted `1 + (gate - pass_rate)` so
  the furthest-from-mastery skills draw the most samples; families at or above
  the family pass-rate gate drop to a small retention weight
  (`--mastery-retention-weight`, default `0.25`) so mastered skills keep being
  rehearsed and regressions surface at the next evaluation.
- Difficulties unlock per family: training samples `easy` until the easy bin
  clears the gate, then adds `medium`, then `hard`. Unlocks are monotonic so
  the training mix does not thrash when a later evaluation regresses.
- The Monte Carlo train curriculum is regenerated deterministically after each
  evaluation phase; failure replay continues to oversample recent held-out
  failures on top of the mastery mix.

The schedule concentrates training where skills are unlearned; it does not by
itself create a learning signal where none exists (for example, jump families
whose failure mode is a painless wall stall). Pair it with expert supervision
such as the distillation/DAgger path or an imitation warm start when a family
is stuck at a zero pass rate across evaluations.

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
