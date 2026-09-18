# Generated families from Full SMB failures

The September 15 frozen transfer evaluation exposed unreliable stair-to-gap
transitions and enemy encounters around landings and raised platforms. The
final Block policy cleared NES level 1-1 once in three trials only after jump
hold calibration; it cleared level 2-1 zero times. These new families supply
independent Block training situations motivated by those failures.

| Family | Situation | Difference from existing coverage |
|---|---|---|
| `stair_gap` | Climb three steps, cross a pit from the top, then descend to the finish. | `stair_climb` stops after the ascent; this adds the immediate edge departure and landing transition. |
| `landing_enemy` | Begin an airborne descent near an approaching enemy, land, then clear it and finish. | `enemy_gap` starts grounded before a gap; this isolates the decision after arrival without a long approach. |
| `enemy_on_platform` | Mount a solid raised surface occupied by a patrolling enemy, then finish. | Flat enemy tasks and widely spaced obstacle chains do not isolate an enemy sharing the mounting/landing area. |

Easy, medium, and hard settings vary gap width and available ledge width,
landing height and enemy approach distance, or platform height/width and enemy
position/speed. Enemy tasks permit safe avoidance or stomping; they do not
teach an unnecessary mandatory stomp to reach the finish.

The three names append to the existing 24-family registry, preserving old
family indices in cached demonstrations. Existing families retain their
explicit-family replay seeds and generators. New cases use the standard stable
split/seed/index machinery, full-route reachability validation, local skill
goals, duration labels, and generated demonstration collection. Family revision
1 describes their initial geometry distributions.

All three participate in bounded policy-state recovery. The stair-gap
collector keeps searching beyond early-step repairs and favors the later gap;
new-family enemy and gap objectives are eligible for repair. Complete suffixes
supply supervision, while failed prefixes only reconstruct state. Forced
landing-release walking frames are excluded from actor labels for the new
families, using the same correction as `stair_climb`.

The full-volume recipe assigns each new family positive sampling weight 2.0
and keeps the requested **10 epochs**. Fresh full-volume runs automatically
build demonstrations for all 27 families. With ten validation repeats per
family/difficulty, each complete validation/test panel now contains 810 cases.
Old demonstration caches contain no examples of the new families; rebuild
them or explicitly refresh all three when using the cached joint-learning tool.
The model architecture and observation dimensions are unchanged.

These are parameterized Block analogues, not exact NES failure-state copies.
They use the same Block physics as the existing curriculum. The NES jump-hold
mapping remains necessary and separate: mixing unmarked physics profiles into
a shared family curriculum would introduce ambiguous training targets. Full
SMB still has different flight dynamics, visual features, and unavailable
motion information. New-family reachability or Block accuracy does not prove
that a policy will complete NES levels.

The previous NES trials are now development evidence for choosing these
families. A future transfer claim should also use new start timings and untouched
levels, while retaining the old starts as regressions. No NES evaluation frames
or action traces are used as training labels here.

Validation artifacts are local under
`artifacts/block_smb/full_smb_failure_families_20260915/`.

## Verification and initial model baseline

The 22 new-family tests pass. The existing sampling, traversal, curriculum,
recovery and demonstration suites pass another 130 tests (152 total). Black,
Ruff and whitespace checks pass. Tests replay the generated routes through the
training collector, preserve family indices and split separation, exercise
successful repairs of actual failed prefixes, and check actor credit around
forced landing releases.

The physical audit covers 360 layouts: 20 train, 10 validation and 10 test per
family/difficulty bin. All complete with the physical teacher using a zero
rejection budget: no unreachable candidate was silently resampled away.

The frozen production epoch-30 model was evaluated on the 180 separate
validation/test cases with its real frozen ViT and normal policy executor.
It received no demonstration updates or oracle actions:

| New family | Validation | Test | Combined |
|---|---:|---:|---:|
| `stair_gap` | 0/30 | 0/30 | 0/60 |
| `landing_enemy` | 27/30 | 21/30 | 48/60 |
| `enemy_on_platform` | 0/30 | 0/30 | 0/60 |

This baseline exposes failures on physically reachable new situations. The
checkpoint predates the stair repair, so its stair-gap score includes existing
stair weakness and does not isolate gap departure alone. The baseline is not
evidence that training on these families has improved Full SMB transfer. That
requires learning followed by independent NES evaluation and retention checks.
No full-volume training was started as part of adding these families.

The executable local audit is `audit.py`; `physical_audit.json` records geometry
coverage, and `baseline.json` records checkpoint/source hashes, per-difficulty
scores and failure details. These held-out samples remain evaluation-only.

## Enemy-on-platform landing transitions (2026-09-15)

The epoch-7 regression exposed a stale local goal: walking off a platform after
completing an enemy jump restored the previous `enemy_clear` request. Training
and batched evaluation now retain a takeoff objective only while its primitive
is active. Ordinary descents recompute the current objective.

Mount-duration certification now includes the Block executor's landing-release
frame and following jump-suppression frame before checking the next enemy.
Canonical and varied teachers use the same landing timing. Recovery generation
inherits pending release frames from the executed prefix, validates completed
suffixes through the fixed jump executor, and retains late enemy arrivals when
selecting its limited repair set. Stomp bounces remain exempt from normal jump
release because they reset the executor.

Demonstration contract 8 records forced-release masks explicitly, including at
recovery suffix boundaries. Earlier cached labels require regeneration; changing
only the cache version or applying the old walking-mask migration is insufficient.
The separate canonical NES executor has different landing/press-edge semantics;
this change does not impose the Block release delay on that runtime.

Verification regenerated both route variants for all 180 production training
layouts: all 360 completed with identical actions through the real executor.
With unchanged epoch-7 weights, the exact 60 enemy-on-platform validation/test
cases improved from 32/60 to 50/60 (easy 14/20, medium 19/20, hard 17/20). The
remaining takeoff/duration errors require learning from the corrected labels;
this score is not a claim of complete mastery or improved Full SMB transfer.
Artifacts are in `artifacts/block_smb/enemy_on_platform_fix_20260915/`.

The exact 810-case, all-family validation comparison with epoch-7 weights rose
from 780/810 to 790/810. Enemy-on-platform improved from 17/30 to 27/30; every
other family's success count was unchanged. The comparison regenerated the
original layouts with the pre-fix sampler and checked their logged parameters
before evaluating with the corrected controller. The regression suite passed
235 tests; a final targeted run passed all 10 transition tests, including the
additional dangerous-duration repair case. Formatting, lint, and whitespace
checks passed. These checks used isolated checkpoint copies and did not restart
or update the running full-volume training job.

## Ten-epoch comparison run

The September 15 comparison recipe ran ten epochs and set `retain_checkpoint_epochs` to
`[5]`. In addition to the rolling `checkpoints/policy.pth` and best-primitives
checkpoint, it preserves `checkpoints/policy.epoch5.pth` with its JSON sidecar.
That snapshot includes model, optimizer, RNG state, configuration, and metrics;
later epochs do not overwrite it. A `checkpoint_retained` event identifies it
in the run log. Fresh runs regenerate contract-8 demonstrations so the landing
and recovery fixes enter both bootstrap training and later rehearsal.


## Piranha Plant avoidance (2026-09-16)

`piranha_avoidance` appends a 28th family without changing existing family IDs.
It samples a solid pipe with a stationary, vertically cycling plant in its
mouth. Easy/medium/hard tiers vary pipe height (22–42 pixels), pipe width
(32–48), and exposed plant height (16–24). Approach position, rise/retraction
speed, exposed/hidden dwell times, and initial phase vary independently.

Contact with an exposed plant is fatal from every direction, including above;
it never grants stomp credit or a bounce. Fully retracted plants have no
collision body and are absent from rendered and geometric observations. The
existing walking enemies retain their stomp behavior. Physics probes snapshot
and restore plant phase along with the rest of the environment.

The family uses the normal local traversal teacher, robust/varied demonstrations,
policy recovery, and independent train/validation/test sampling. Duration labels
must survive the actual moving hazard, landing release, and the following local
objective; complete demonstrations must reach the exit alive. The full-volume
recipe gives this family weight 2 and runs **15 epochs**, retaining epochs 5 and 10.
Both coached and autonomous CUDA preflight batches include the new family.

This is generated Block practice for clearing occupied pipes. It preserves the
existing observation dimensions and does not expose the cycle timer to the
policy. Its cycle is a proxy, not an exact reproduction of NES plant timing or
Mario-proximity suppression. Native NES evaluation remains necessary to measure
transfer after training.

Validation artifacts: `artifacts/block_smb/piranha_family_20260916/` contains
collision/observation/controller regression results, the production-layout route
audit, and the real frozen-vision CUDA preflight.


## Phase-robust plant teaching and observable history (2026-09-17)

The plant audit found identical single-frame inputs paired with different safe
jump-duration labels: a low rising plant and a low retracting plant looked the
same, while the teacher could inspect their cycle phases. More death penalties
would not resolve that ambiguity.

Plant duration probes, canonical routes, varied routes, and recovery suffixes
now hold each plant at its full collision height throughout certification. In
this generated family, that maximum is fixed by the visible pipe width (32/40/48
pixels corresponds to a 16/20/24-pixel plant). Certification is independent of
cycle phase; it does not invent a 24-pixel plant on the narrowest pipe. Completed
routes must also replay successfully against the original cycling environment
through the normal primitive executor. Probe snapshots restore all live state.
The simulation still cycles normally and all plant contacts remain lethal.

The full-volume recipe enables `hazard_observations`, a versioned extension to
the existing motion-aware geometry contract. Six shared Block/NES features add
visible-enemy presence, observed vertical velocity, velocity availability,
visibility transitions, time since a visibility transition, and time since last
sighting. Velocity comes from consecutive observed positions; occluded/offscreen
objects and missing frames do not imply a measured zero. These inputs contain no
cycle phase or future rollout. Visibility ages saturate at 64 frames and vertical
velocity uses an 8-pixel-per-frame scale. History resets between episodes and
re-encoding a frame does not advance it.

Recovery trajectory budgets now count failures only. Plant repair selection
continues through later arrivals and attempts, retaining later valid suffixes
within the existing per-trajectory budget. Successful episodes still enter the
normal success-replay machinery.

Plant family revision 2 and demonstration contract 9 identify the new teaching
rules. Old plant labels must be regenerated; the new history input layout also
requires fresh demonstrations. Checkpoint restore rejects a history-layout
mismatch even though the model tensor dimensions happen to match. Checkpoints
without the flag retain their prior inputs, including Full SMB transfer. New
full-volume training remains 15 epochs with retained checkpoints at 5 and 10.
The history extension currently supports `smb_geometry_v1`; it is not enabled for
the separate canonical `smb_scene_v2` projection.

Validation passed 201 distinct regression tests across the main and additional
history/cache checks. All 34 previously failed audit trajectories still have
verified repair suffixes. Route verification completed all 60 original plant
audit layouts and all 360 canonical
and varied routes across 180 generated training layouts with the real executor.
Tests cover phase-independent labels, shifted-cycle replay, identical pixels with
distinct observed vertical motion, NES history, checkpoint compatibility, and
bounded failure recovery. A real frozen-ViT/CUDA preflight exercises both coached
and autonomous optimization. These are correction checks, not evidence of a new
trained policy's success rate or Full SMB transfer. Validation artifacts are in
`artifacts/block_smb/piranha_corrections_20260917/`. Applying these changes does not
modify the code already loaded by an existing trainer.


## Demonstration history collection correction (2026-09-18)

The first history-aware run exposed a collection bug: `collect_demonstrations`
advanced `env.step` directly, but enemy history belongs to `BlockSMBStage.step`.
Every bootstrap, rehearsal, and recovery demonstration therefore repeated the
six reset-time history values throughout its trajectory. Autonomous training
and evaluation advanced the adapter correctly, producing different inputs for
the same route. The previous tests covered live Block/NES history and input
layout compatibility but did not compare collected demonstrations with playback.

Collection now advances the adapter before saving each next observation. This
also reconstructs history throughout unsupervised recovery prefixes. New tests
compare every collected C input and the terminal next-C target against normal
playback, both with history enabled and with the legacy layout, including a
recovery splice after frame 12. They fail on the old collector and pass with the
correction. No policy-time action rule or hazard physics changed.

Demonstration contract 10 invalidates all history-enabled caches from earlier
contracts, including non-plant families and recovery rows. Regenerate their
observations from the original routes; relabeling the manifest is insufficient.
Checkpoint input dimensions and feature meanings are unchanged. An already
running trainer continues to use its loaded collector and stale in-memory data;
it needs a restart with regenerated demonstrations to use this correction.

The frozen epoch-12 checkpoint reproduced the logged 28/60 plant successes on
the exact validation/test layouts. A separate audit collected 72 routes from
36 independent training layouts using the real frozen ViT: 6,950 of 7,064 rows
had incorrect history before the fix. Velocity was marked available in zero
old rows versus 5,030 corrected rows. All other observation columns, actions,
goals, masks, and safe-duration labels were identical between collectors.
All 107 regression tests, formatting, and lint checks passed.
Artifacts are in `artifacts/block_smb/piranha_history_fix_20260918/`.


A controlled 1,000-update fine-tune used the same frozen epoch-12 weights,
training routes, seed, and optimizer settings in both conditions. Stale-history
and corrected-history data each produced 23/60 held-out successes, below the
28/60 baseline. Freezing runtime history to mimic the old demonstrations gave
32/60 with unchanged weights, as a diagnostic only. These checks establish the
input mismatch and its removal; they do not establish that this correction
alone resolves plant avoidance or improves retention. The experimental weights
are isolated artifacts and are not installed into the active training run.


## Twenty-epoch run with five-epoch snapshots (2026-09-18)

The full-volume recipe now runs 20 epochs and natively retains snapshots at
5, 10, 15, and 20. Each `policy.epochN.pth` has a matching JSON sidecar and
contains model, optimizer, RNG state, configuration, and metrics. The rolling
checkpoint continues to update independently. Fresh training regenerates
contract-10 demonstrations, including the enemy-history collection correction.
