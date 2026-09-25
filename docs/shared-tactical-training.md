# Shared tactical training

Tactical supervision now covers 18 families, including the existing
`piranha_avoidance` family:

| Decision type | Added families |
| --- | --- |
| Bridge timing | `wait_timing`, `moving_bridge`, `bridge_wait`, `bridge_mount`, `bridge_dismount` |
| Enemy approaches | `enemy_stomp`, `enemy_patrol`, `enemy_on_platform`, `landing_enemy` |
| Positioning and recovery | `tall_pipe_jump`, `stair_climb`, `stair_gap`, `retreat_recovery` |
| Sequences | `chained_obstacles`, `chained_enemy_gauntlet`, `mixed_section`, `full_smb_opening_proxy` |

Successful demonstrations label grounded, uncommitted decisions as advance,
hold, or retreat. Direction is relative to the goal: walking left toward a
leftward goal is advance. Waiting and riding a bridge are hold decisions,
including directional braking. Airborne commitments and forced landing releases
do not receive new tactical targets. Equivalent observed decisions accept the
union of successful tactical choices, so valid early and late departures do not
contradict each other.

Online targets use the current state, bridge phase, collision-tested jump holds,
and short ground-motion probes. A blocked approach can request retreat only
when the fallback remains supported. Ground probes certify four frames, not a
complete recovery route; uncertain states are left unlabeled. Existing temporal
piranha supervision still uses observation history. These teachers supply
training targets and evaluation diagnostics; they do not override policy actions
at inference. There are no new alternate-route targets for these linear tasks.

The joint objective trains the tactics transformer, A's skill/action choice,
and B's physically valid primitive duration. In addition to stance supervision,
an auxiliary loss rewards A for assigning probability to actions consistent
with the target tactic. Both tactical terms use `tactic_loss_weight` (default
0.5). Existing action, duration, dynamics, and reinforcement objectives remain.
Flat bridge wait durations still train B; single-frame waits for bridge jumps
and timed piranhas do not pretend to consume a duration prediction.

Recovery demonstrations cover the added families and can repair premature flat
bridge departures into successful wait/ride/exit sequences. Evaluation exposes
`tactics_by_family` with decision counts, stance accuracy, and agreement between
the predicted stance and executed action. `piranha_tactics` remains specific to
piranha. These diagnostics do not replace family success rates.

Demonstration contract version 12 rebuilds cached supervision with tactical
action sets and explicit duration-consumption masks. No model architecture or
checkpoint shape changes are needed. An already running trainer must be
restarted to load this implementation.
