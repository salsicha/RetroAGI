# Coaching for composable SMB training

The composability migration introduced a new collector that bypassed important
parts of the previously qualified teaching path. The September 9 repair restores
those capabilities under `canonical_collision_coaching_v1`. Earlier family
mastery results describe the earlier pipeline; they do not qualify the new model.

The distinction is between teacher information and player information. During
Block training, the teacher can inspect collision geometry, snapshot the engine,
and test alternatives. It supplies action and duration labels. The student's
inputs are still the canonical pixel scene and explicit task request. During
validation and gameplay, the actor chooses actions and the shared executor
executes them. There is no teacher search or supplied demonstration action.

## Common observations and goals

`retroagi/core/smb_objectives.py` supplies the same visible-scene objective logic
for Block and Full SMB. A bridge has approach, wait, board, ride, and exit targets.
The bridge can be the target even before the far shore enters the viewport.
Only visible support, positions, and collision-box estimates are used; no hidden
patrol or platform reversal bounds enter these goals. These are target hints,
not commands to walk or wait. Observable adjacency is not a guarantee of safe
boarding: the model still learns that choice from physical outcomes.

The existing goal vector encodes bridge waiting/riding as `wait_pass`, boarding
as `mount_platform`, and exiting as `clear_gap`. A required stomp uses the
`enemy_clear` type with magnitude 1, distinguishing it from an optional enemy
clear. Block scenarios declare `task_objective="stomp"`; the Full adapter accepts
the same explicit `task_objective` request. Direction is likewise an explicit
`task_direction`. Neither request reveals enemy velocity or future collisions.
Static targets remain fixed in world coordinates during a committed flight.
Required stomp targets follow the observed enemy; bounce recovery clears the
skill goal until support returns. The current [stomp repair](smb-stomp-scrolling-repair.md)
adds camera-independent closing velocity and duration labels at policy takeoffs.

The actor, hierarchy, LSTM, critic, and motor controller keep their architecture
and tensor shapes. Perception remains swappable across domains. Goal meanings
are versioned as `observable_traversal_v1` in component and runtime manifests;
older components lacking that contract are rejected instead of silently treated
as compatible. The CNN/ViT creation and training lifecycle remains described in
[smb-segmentation-curriculum.md](smb-segmentation-curriculum.md).

## Physical teaching and action boundaries

`retroagi/core/smb_coaching.py` is training-only. It certifies the 16 physical
jump holds through the same executor used for playback, following collisions
through landing or terminal completion. Required stomps must earn actual stomp
credit; a safe jump over an enemy is insufficient. Nonterminal stomps must also
survive the bounce, and an immediately trapped next-enemy landing is rejected.
All probed environment state, including fractional NES motion and credit, is
restored afterward.

For bridges, the teacher tests real walking support, platform carry, reversal,
and boarding/crossing credit. It examines every potential departure frame up to
32, maps successful departures to the duration menu, and schedules an earlier
recheck when a narrow window falls between bins. When no departure is within
that horizon, labels allow physically safe waits followed by a new observation.
This replaces the legacy walking predictor in the new coaching path.

Each decision records its full verified duration set. The existing supervised
loss accepts any member and favors the interior of each contiguous safe set.
Committed frames retain the initiation label but do not train a new actor or
duration decision. A shared executor `prepare` hook resolves observed landings,
bounces, and bridge readiness transitions before the collector labels a
new decision. Playback uses that same hook. Readiness ends an existing wait;
the actor must still choose what to do next.

Both original routes and reactive alternatives run through the live stage and
executor. A route enters replay only after actual final completion. Alternatives
vary takeoff distance and safe hold selection; failed alternatives are recorded
and excluded. Family/action-balanced, prioritized rehearsal and accepted action
sets for identical observed states remain enabled. World-model targets remain
one physical frame ahead, matching the composable recurrent cadence.

## Family audit

| Families | Restored or preserved coaching |
| --- | --- |
| bridge_wait, wait_timing, moving_bridge | Visible bridge goals; real collision departure sets; consistent wait boundaries and bridge completion credit. |
| enemy_stomp, stomp_mount | Explicit stomp task input; stomp-required safe durations; bounce survival; recovery labels; varied takeoffs. |
| tall_pipe_jump, pipe_mount, pit_leap, platform_hop | Actual landing/finish criteria; all safe physical holds; fixed flight goals; alternative takeoffs and landings. Single-attempt platform tasks retain their stricter completion rules. |
| enemy_hop, enemy_patrol, enemy_gap, chained_enemy_gauntlet | Safe interception/clear durations; bounce masking; next-enemy recovery checks; alternative routes. Hidden patrol bounds remain unavailable to the pixel player. |
| single_gap, stair_climb, platform_chain, chained_obstacles, mixed_section, full_smb_opening_proxy | Local obstacle targets; collision-safe duration sets; full-route completion; varied takeoffs and downstream landing states. The earlier descent-versus-gap fix is preserved. |
| retreat_recovery | Signed observable goals; leftward action balance; left-jump collision checks and route alternatives. |
| flat_run | No missing special coach was found. Autonomous walking, complete-route verification, and balanced action supervision are retained. |

## Verification and restart

Run `python -m scripts.smb_coaching_audit --output <fresh-directory>` for a
bounded executable-teacher audit across all 21 families and all three difficulties.
Add `--perception <checkpoint> --device cuda` to test the actual pixel adapter.
An optional `--learning-updates N` probe reports both training layouts and a
separate validation split. Teacher feasibility is never reported as learned
accuracy; a small learning probe is not full-family qualification.

Regression tests in `scripts/tests/test_smb_canonical_coaching.py` cover physical
wait windows, state restoration, stomp task inputs, commitment boundaries,
collision-coached routes, teacher-free playback, and component compatibility.

The full run initializes a fresh shared policy. By default it trains vision for
8,000 updates; `--perception-checkpoint` instead reuses qualified, frozen vision
and starts epoch 1 with fresh policy weights. See the
[sensorimotor repair](smb-sensorimotor-repair.md) for the version 2 tracking,
stomp recovery and landing contract and the vision-only restart command.
There is no policy bootstrap stage. Each of the 30 epochs collects 25 layouts
per family and interleaves coaching batches with its 1,000 rehearsal updates.
Earlier epochs remain in replay. Each accepted layout also attempts a varied
reactive route; there are no independent prerequisite family models.
The current run contains 20 independent families: `wait_timing` is an explicit
alias of `bridge_wait` and receives no duplicate sampling weight. Fixed scenes
remain excluded. Family scores
after each epoch measure autonomous completion. Full SMB promotion still
requires the final measured qualification.

Development verification on September 9, 2026:

- The executable teacher completed 63/63 generated cases: one case at each
  difficulty for each family (seed 20260909, training indices 100–102).
- A small bridge probe using the previously trained pixel perception module
  completed 12/12 teaching routes. A fresh 64-dimensional policy improved from
  0/12 to 11/12 autonomous training-layout completions after 1,000 updates and
  completed 2/3 separate validation layouts. This is preliminary evidence, not
  a 99% reliability claim or qualification of the final shared model.
- 81 focused regression tests passed, covering coaching, components, shared
  restart scheduling, transfer contracts, and demonstration learning. Subsequent
  adapter checks cover the explicit Full SMB task inputs as well.

Local detailed reports are under
`artifacts/smb_composable/coaching_repair_20260909/`. Diagnostic weights are not
used to initialize the fresh full-volume run.

The subsequent [motion/curriculum repair](smb-motion-curriculum-repair.md) adds
bounded motion memory, per-frame bridge re-observation, phase-balanced replay,
actual-policy miss recovery, and an explicit `wait_timing` alias. Its v4 interface
supersedes the historical contracts and pre-restart status described above.
