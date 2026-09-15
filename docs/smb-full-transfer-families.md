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
