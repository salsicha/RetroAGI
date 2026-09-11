# Bridge jump family debugging — 2026-09-11

The latest `block_smb` family runs exposed execution and supervision mismatches,
not a failure of the collision-credit rules. The canonical composable path's
oracle already completed these tasks, but the older Block training path could
not execute the same labels.

## Reproduced causes

1. **Different meanings for the same duration bin.** Bridge scenarios use
   `nes_land_v1`. Their demonstrations encode the 16-entry `NES_JUMP_FRAMES`
   menu, ending at 32 frames. `collect_trajectory` and batched qualification
   instead instantiated the legacy executor with 1–16-frame durations. For
   example, the demonstrated 28-frame hold (index 14) executed as 15 frames.
   Changing the tensor alone would still encounter the executor's 16-frame cap.
2. **Waiting for the wrong event.** The jump-family wait cue used the bridge's
   nearest track endpoint. That is not a certified jump window. Long wait
   commitments could hide the window from the actor, and hindsight training
   reinforced waiting until the endpoint.
3. **Walking-bridge handling remained in other paths.** Demonstration collection
   and batched evaluation used `require_bridge_before_goal` to classify both
   prerequisites as complete walking crossings. This cleared their requested
   goal encoding in board/exit phases. Alternate-route generation also attempted
   to walk across a gap that requires jumping.
4. **Replay suppressed intermediate wait decisions.** Steady-action alignment
   combined bridge NOOP rows into waits of up to 64 frames. The observations
   needed to learn the transition from waiting to jumping lost their actor mask.

A direct physics sweep of `bridge_jump_scenario(random.Random(101), difficulty,
family)` found these values (each call uses a fresh RNG):

| Family | Difficulty | Oracle departure frame | Shortest successful hold at departure | Endpoint cue frame |
|---|---|---:|---:|---:|
| mount | easy | 35 | 26 | 100 |
| mount | medium | 30 | 26 | 91 |
| mount | hard | 21 | 26 | 80 |
| dismount | easy | 108 | 26 | 105 |
| dismount | medium | 97 | 26 | 95 |
| dismount | hard | 85 | 26 | 83 |

All six oracle routes used 28-frame holds. At the three mount endpoint cues,
none of the tested holds from 1 through 48 frames succeeded. The reported
minimum holds are specific to these departure states, not a universal family
threshold.

The saved dismount `anchor` and `noshape` runs both collapsed to waiting or
walking off the bridge; removing goal-distance shaping could not repair an
unexecutable jump label.

## Repair

- Block sequential and batched execution now map jump bins through the scene's
  physical profile, with enough executor capacity for the NES menu. Walk/wait
  bins retain their existing mapping; mapping does not mutate shared model
  weights or other batch rows.
- Bridge prerequisite waits reconsider the policy every frame, as canonical
  bridge playback does. There is no hidden track-end action cue or wait-duration
  hindsight label for these tasks.
- Jump-family goals remain present in demonstrations and both evaluation paths.
  Alternative routes use the jump oracle and must pass physical validation.
- Training carries physical duration values through release and uses certified
  landing holds for the bridge jump's duration loss.
- Replay keeps bridge wait decisions and their immediate next-state targets.
  Demonstration contract version 3 requires cached bridge jump families to be
  refreshed rather than silently reusing the old goals and masks.

## Verification and limits

`test_bridge_jump_execution.py` exercises policy-executor landings for both
families at all three difficulties, unchanged shared duration tensors, matching
batched/sequential goals, alternate jump demonstrations, every-frame wait
supervision, and physical duration coaching through button release. The existing
collision-credit, demonstration-learning, and Block training suites are also run.

These are execution and supervision checks, not a claim of 99% learned
qualification. A separate bounded CPU learning check and its configuration are
stored in `artifacts/block_smb/bridge_family_debug_20260911/`.

The already-running `full_volume_20260910_bridge_jump_fix` process was not
restarted or modified. It retains its previously imported code. Existing weights
trained with the conflicting labels are not evidence of qualification under the
repaired execution contract.

The bounded learning check used one fresh shared policy, the existing frozen
`data/block_vit/block_vit.pth` perception model, six training layouts per family,
and 400 imitation updates. On six held-out layouts per family, mount scored
0/6 and dismount 1/6. Failed routes now execute 28 jump-hold frames, confirming
the duration repair is active; premature takeoff remains the observed policy
failure. More training is required before claiming either family is learned.

The broader regression run passed 124 tests and exposed a test double that did
not accept the new replay-alignment keyword. After updating that test double,
all 15 affected checks passed, covering all 125 tests in the selected suites
across the runs. Lint and whitespace checks also passed.

The follow-up timing audit measured failed mount takeoffs 9–16 frames earlier
than their verified oracle departures; failed dismount takeoffs were 3–5 frames
earlier. The per-layout values are saved in `timing_audit.json` beside the
learning results. These are remaining errors in the bounded learned policy,
separate from the repaired duration execution and replay contracts.
