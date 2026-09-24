# Piranha avoidance and enemy stomp audit — September 23, 2026

The epoch-20 policy still fails the per-difficulty requirement. The investigation
found and fixed training and coverage problems, but has not established a new
policy above 90% in every bin. The original checkpoint remains unchanged.

## Reproduced baseline

All 120 recorded validation/test cases were regenerated with matching parameters,
and every success/failure was reproduced using the autonomous executor.
Each cell below contains ten trials.

| Family / split | Easy | Medium | Hard |
|---|---:|---:|---:|
| enemy_stomp validation | 100% | 50% | 100% |
| enemy_stomp test | 100% | 60% | 90% |
| piranha_avoidance validation | 60% | 60% | 70% |
| piranha_avoidance test | 30% | 40% | 80% |

Collision checks enumerated the 16 holds at actual grounded takeoffs. Among the
ten failed stomp episodes, seven first departures had no certified hold and three
selected an incorrect hold from a nonempty safe set. Among 26 failed plant
episodes, those counts were 20 and six. Certification requires target completion
and recovery; plants use the maximum-exposure envelope, not hidden phase.
An uncertified first primitive is not necessarily immediately fatal: successful
episodes sometimes recover after one, so this is not a complete causal attribution.

Examples: medium stomp validation 256 chooses 14 frames when only 16 is certified.
Medium plant validation 823 chooses nine when only ten is certified; its retry
chooses nine when the safe set is disjoint, `{3, 11}`. Increasing every hold would
therefore be incorrect.

## Implemented fixes

1. **Duration supervision follows execution.** Online training previously applied
   the original takeoff-duration target throughout a committed jump, although the
   fixed executor only consumes the head at initiation. The feedforward policy
   lacks the original departure state and chosen hold during continuation. Fixed
   control now receives duration loss only at initiation, matching demonstration
   fitting. Adaptive supervision and world-model outcome targets are preserved.
   Tests inspect which duration predictions receive gradients.
2. **Distinct stomp recovery states remain distinct.** The recovery key previously
   merged repeated mistakes against one enemy despite different direction and
   momentum. It now includes direction, position bin and velocity bin.
3. **Keep initial and later corrections.** With the normal three-repair budget,
   prioritizing late plant retries could discard the initial mistake. In medium
   validation case 824, frames 38/57/85 displaced frame 8. The sampler now reserves
   the first correction and later recoveries when capacity permits. The existing
   one-slot behavior is preserved. Regression fixtures replay complete successful
   corrections while checking that the failed prefix remains unsupervised.
4. **Decouple plant route coverage from difficulty.** Both difficulty and plant
   route selection cycled modulo three. The production recipe systematically
   omitted one alternate in each difficulty and replaced the canonical route.
   Every plant layout now retains the canonical route and all three alternates,
   following the existing bridge-variant coverage pattern.

These changes correct mechanisms; their eventual learned-policy improvement
still requires a controlled training comparison.

## Bounded learning experiment

The pilot generated 48 fresh training layouts per weak family with canonical and
alternate routes, plus three retention layouts per other family: 462 routes and
46,134 frames. Corrections from 25 failed training episodes added 4,484 frames.
It ran four 500-update supervised blocks with prioritized replay and fourfold
weight for each weak family. Validation chose the 500-update candidate by its
minimum-bin score; none improved the baseline minimum. No held-out layout entered
optimizer updates.

This pilot used the pre-fix recovery sampler already loaded in its process and
did not run the online loss. It tests expanded route coverage and focused fitting,
not the combined effect of all the fixes above.

| Family | Original test E/M/H | Selected pilot E/M/H | New test seed E/M/H |
|---|---|---|---|
| enemy_stomp | 100 / 60 / 90 | 100 / 90 / 80 | 100 / 73.3 / 86.7 |
| piranha_avoidance | 30 / 40 / 80 | 40 / 50 / 40 | 63.3 / 63.3 / 43.3 |

The new test seed used 30 cases per difficulty, evaluated after selection.
On identical retention cases, the original passed 78/78 and the candidate passed
70/78. Regressions included single_gap, retreat_recovery, stomp_recovery and
bridge_dismount. The candidate is rejected. Training loss falling from 0.832 to
0.595 did not translate to reliable rollout improvement.

## Next experiment

- Compare the corrected online loss with the previous loss using identical fresh
  training seeds and budgets. Retain the full original demonstration pool for the
  other 26 families; the pilot's three-layout retention pool was inadequate.
- Refresh successful corrections from the current greedy policy each short round.
  Increase the present two-failures-per-bin recovery budget as a controlled
  experiment, keeping initial and later corrections. Repeated fitting of a stale
  correction pool misses states reached after policy changes.
- For stomp, cover both approach directions, reversals, takeoff velocity and the
  full safe departure window. For plants, cover hidden/emerging/retracting states,
  pipe approaches and post-landing retries with conservative labels. Vary departure
  timing as well as hold length; existing plant variants mainly vary holds.
- Track takeoff-choice accuracy and duration-set accuracy separately. Test
  normalizing unsafe-takeoff loss over relevant decisions: its current average
  over all frames dilutes sparse corrections. Treat this as an ablation, not an
  already-validated fix.
- Select by the worst difficulty bin plus retention, including the unchanged
  baseline as a candidate. Aim for at least 95% per bin across several validation
  seeds, then use an untouched test seed with at least 100 cases per bin. Ten-case
  bins and family averages are insufficient qualification evidence.

No additional family or larger model is established as necessary by this audit.

## Verification and artifacts

The trainer/recovery suite passed 87 tests, demonstration/execution passed 59,
and the initial family-focused suite passed 33 (two overlap with the later suite).
Ruff passes. System pytest plugins were disabled, with pytest_timeout enabled.

Artifacts are under `artifacts/block_smb/weak_family_audit_20260923/`:
`baseline.json`, `replay.json`, `physical.json`, `pilot_results.json`,
`retention_baseline.json`, experiment scripts, saved demonstration tensors,
isolated pilot model state dictionaries and test logs. The saved pilot data retain
the pre-fix recovery selection; recollection using the new sampler is a distinct
experiment. None of the pilot files replaces the production checkpoint.
