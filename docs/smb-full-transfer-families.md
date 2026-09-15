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
