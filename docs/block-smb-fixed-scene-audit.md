# Fixed-scene audit — September 7, 2026

The saved epoch-34 policy completes 2/16 original fixed scenes. The same weights
complete 13/16 when the scenes receive corresponding family metadata and
controller behavior. No optimization or layout changes were used in this comparison.
All 16 original scenes have successful action sequences verified by replay in
MarioScenarioEnv. This is primarily a controller/training integration failure,
with a real missing left-jump skill and additional bridge coverage weaknesses.

## Reproducible evidence

Artifacts: `artifacts/block_smb/fixed_level_audit_20260907/`.
`probe.py` records original and family-contract policy rollouts with actual
frozen Block ViT perception, CUDA, greedy actions, and the production executor.
`results.json` and the individual frame traces record actions, positions,
grounding, skill requests, and completion. `physics_probe.py` verifies physical
solutions separately; these are solvability evidence, not learned-policy scores.
`followup.py` isolates bridge request handling and duration coaching.

Checkpoint: epoch 34 of `full_volume_20260906_qualified_restart_seed20260906`.
SHA256: `970d74e3c969daff5c1897a920c7ea2498115c38c9b3766cb5edf023a2654d29`.
All experiments use this frozen copy. The continuing training process and
production Python source were not changed. These are single-layout diagnostic
replays, not a new multi-seed learning qualification.

## Actual per-scene results

| Fixed scene | Original policy path | Corresponding family path | Verified physical route frames |
|---|---|---|---:|
| level_1_flat.json | Pass | Pass | 70 |
| level_2_gap.json | Fail | Pass | 67 |
| level_3_stairs.json | Fail | Pass | 84 |
| level_4_platforms.json | Fail | Pass | 75 |
| level_5_enemy_hop.json | Fail | Pass | 70 |
| level_6_enemy_patrol.json | Fail | Pass | 73 |
| level_7_moving_bridge.json | Fail | Fail | 222 |
| level_8_enemy_gap.json | Fail | Pass | 70 |
| level_9_enemy_stomp.json | Fail | Pass | 70 |
| level_10_left_retreat.json | Fail | Pass | 52 |
| level_11_left_jump_recovery.json | Fail | Fail | 40 |
| level_12_wait_bridge.json | Fail | Pass | 85 |
| level_13_variable_pits.json | Fail | Pass | 154 |
| level_14_under_enemy_platform.json | Pass | Pass | 110 |
| level_15_wait_long_bridge.json | Fail | Fail | 162 |
| level_16_wait_enemy_gate.json | Fail | Pass | 110 |

The family mapping is explicit in `probe.py`. The old enemy-stomp scene maps to
`enemy_hop` to preserve its original requirement of surviving to the finish;
passing that scene does not establish a stomp. Bridge family metadata activates
an additional physical boarding/crossing requirement. It does not relax success.
The overhead-enemy scene remains a flat-run request, not an instruction to climb
onto its optional overhead platform. This is a diagnostic mapping, not a proposed
production solution that disguises fixed scenes as Monte Carlo samples.

## 1. Missing metadata selects a different controller path

All fixed JSON files omit `metadata.block_smb_monte_carlo.family`.
`skills.requested_block_smb_skill_goal` derives its request from that field.
`train.collect_trajectory` uses it to activate local obstacle objectives and
otherwise supplies a zero/neutral request (train.py:2351). Generated tasks switch
between clear-gap, mount-platform, enemy-clear, retreat, and finish requests;
fixed tasks bypass that machinery.

Observed failures include 320 consecutive RIGHT actions against the first stair,
320 RIGHT actions when the goal is to the left, and walking directly into gaps
and stationary enemies. Restoring the matching family path fixes eleven of the
fourteen failures immediately. The trained policy therefore has much of the
necessary behavior, but the original fixed controller does not request it.
This integration gap was missed by the family qualification.

## 2. Fixed tasks miss retention and receive conflicting coaching

The fixed scenes do receive live policy rollouts, approximately 16 of the 528
base scenarios in each epoch before additional replay. They do not receive the
same support as generated families:

- `BlockSMBSuccessReplay.add` rejects trajectories without a family.
- `demonstrations.build_balanced_demonstrations` builds examples only from the
  21 generators. The 1,000 rehearsal updates per epoch contain no fixed scenes.
- Mastery weights and adaptive failure replay are keyed by generated families;
  failing fixed scenes cannot trigger those same recovery mechanisms.
- Fixed jumps bypass local target geometry and valid hold sets. Their fallback
  duration coaching compares the jump with the final episode goal.

A controlled replay of the exact successful fixed-gap action sequence held jump
for 11 frames. Both scene variants completed. Original fixed coaching requested
12 frames; matching family coaching preserved 11 and supplied valid holds 6–16.
This is direct evidence of inconsistent supervision, even on a successful route.
It does not establish that every fixed-scene duration label is wrong.

## 3. Bridge approach is mistaken for completed waiting

After adding bridge family metadata, scenes 7 and 15 still fail. Their first
RIGHT action at x=20 clears `bridge_opening` in train.py:2736. The subsequent
boarding phase supplies a neutral goal even though Mario is still approaching
on the near shore. Both policies then walk off the shore.

A process-local diagnostic recomputed the wait/ride request from physical bridge
phase instead of treating that first approach action as completed waiting.
The same policy then completed scene 7 in 122 frames and scene 15 in 105 frames,
with actual bridge crossing credit. However, it regressed scene 12 from a pass
to a failure. This supports a phase-contract diagnosis; it is not an accepted
fix. A repair must distinguish approach, waiting, boarding, riding, and exit,
then retest all bridge families and all three fixed bridges together.

There is also distribution mismatch: fixed bridges are 50 pixels wide and move
at 0.5–1.0 pixels/frame; generated bridge tasks use 100-pixel bridges at roughly
1.6–2.4 pixels/frame with a different travel range and initial approach. The
family qualification does not exhaust these older geometries.

## 4. Leftward jumping is not covered by the qualified retreat family

Adding the retreat request repairs scene 10 (walk left). Scene 11 then walks
left off its platform and dies because it requires a jump onto a platform
50 pixels higher. The generated `retreat_recovery` family always uses continuous
flat floor and an all-LEFT reference route (monte_carlo.py:1356).

`local_objective` immediately returns `retreat` when the goal is behind Mario,
before inspecting gaps or raised platforms (local_traversal.py:63). Several
numeric terrain features also search to Mario's right. Thus this is both missing
training coverage and a direction-asymmetric objective/observation design.
It is not evidence that a larger network is needed. The original scene is
physically solvable: ten LEFT_JUMP frames followed by LEFT completes in 40 frames.

The fixed leftward tasks also retain the default rightward progress reward and
lack the goal-distance shaping explicitly enabled in the generated retreat
family. Their reward contract was not updated alongside that family's repair.

## 5. Training budgets and success meanings are inconsistent

Original fixed tasks receive 160 training frames; evaluation allows 320.
Bridge family tasks receive a 240-frame floor and other generated compounds can
receive oracle-derived extensions. The verified walking bridge routes take
222 frames (scene 7) and 162 (scene 15), so that valid behavior would be truncated
in fixed-scene training. These are route lengths, not minimum possible solution
lengths: the diagnostic learned routes above are faster. Budget mismatch is an
additional training problem, not a claim that success within 160 is impossible.

The fixed `enemy_stomp` scene requires no actual stomp. Fixed bridge scenes do
not require boarding/crossing, and a scene named `wait_enemy_gate` can be passed
by jumping over its enemy. The score measures scene completion, not necessarily
the skill suggested by its filename. A repaired benchmark needs explicit
behavioral requirements, or names and descriptions that match ordinary traversal.

## Recommended repair order

1. Give every scene a consistent, explicit controller/task contract independent
   of whether it came from a generator. Share local goal and duration geometry
   handling between fixed and generated tasks; keep provenance separate.
2. Repair bridge approach/wait transitions using physical phase information,
   with regression coverage for narrow/slow bridges and the generated families.
3. Add leftward elevated-platform/gap cases and corresponding direction-aware
   local objectives, observations, demonstration routes, and rewards.
4. Include fixed tasks in successful demonstrations, retention replay, and
   failure-based practice; align rollout budgets with actual verified routes.
5. Make success requirements explicit and require both fixed-scene and generated
   family gates before claiming the combined training curriculum is reliable.

The 13/16 diagnostic is evidence for the diagnosis, not a claimed production
accuracy improvement: no runtime fix has been applied to the active run.
