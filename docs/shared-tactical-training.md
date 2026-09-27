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

## Repositioning families and enemy wait routes (contract 14)

`stomp_recovery` and `platform_chain` now receive tactical targets. Their
demonstrations head straight for the goal, so demonstration labels are all
advance. `platform_chain` also receives online labels from the certified
positioning probes above (hold at an edge without a safe jump, back off when
blocked). `stomp_recovery` has no local objective to certify online positioning
with, so it is labelled from demonstrations only. Both join policy recovery:
failed `stomp_recovery` episodes were repaired in 12 of 12 attempts, and failed
`platform_chain` episodes in none, so the latter gains only online tactical
labels.

Canonical enemy routes never wait, so the tactical layer never saw a hold near
an enemy, and stomp takeoffs covered the enemy positions of a single timing.
`demonstration_enemy_wait_routes` adds one route per layout that requires a
stomp and has a moving walker. The route replays the canonical prefix to the
first grounded walking decision with the enemy 32–112 pixels ahead. It waits
8–24 frames (a multiple of four, the executor's wait granularity) while
grounded and alive, then the recovery teacher completes the level. Only
complete routes are kept. Stationary enemies are skipped, since waiting for
them changes nothing but the clock. In 9 sampled layouts per family, only
`enemy_stomp` qualifies, with routes for 6.

The option is off in the recipe because no probe showed a reliable effect. Each
probe fitted 4,000 updates on 60 training layouts of `enemy_patrol`,
`enemy_stomp`, `landing_enemy` and `enemy_on_platform`, then counted successes
on 30 held-out layouts per family:

| Run | patrol | stomp | landing | platform | Total |
| --- | --- | --- | --- | --- | --- |
| Without wait routes (earlier strategy variant) | 30 | 14 | 30 | 29 | 103 |
| Wait routes in every enemy family (same variant) | 22 | 27 | 30 | 16 | 95 |
| Without wait routes (final code) | 30 | 23 | 30 | 26 | 109 |
| Wait routes in every enemy family (final code) | 26 | 30 | 30 | 30 | 116 |
| Wait routes in stomp layouts only (final code) | 27 | 13 | 30 | 10 | 80 |

Single families moved by up to 17 of 30 between runs whose data differed
little, so single-seed probes cannot rank these options. Stomp-only routes
follow the mechanism (waiting lets an enemy walk into stomp range; in avoidance
layouts it only shifts timing). Enabling them needs a multi-seed comparison.

## Strategy objectives

The strategy network previously read only a history of stance probabilities.
That history was reset whenever no recurrent state was carried, which included
every batched demonstration row, so the network produced a constant context.
The skill goal the policy acted on (clear a gap, mount, clear an enemy, retreat,
or none) came from the scripted `local_objective` selector, both in Block
evaluation and in Full SMB playback.

The strategy network now also has an objective head that names the current
skill goal, one of the five skill types or none. It reads the C stream, the
carried world-model memory and the stance-history context. It is supervised with
the goal the teacher or selector supplied (`strategy_loss_weight`), both in
demonstration fitting and online. The head only reads its inputs: they are
detached, its prediction does not feed the tactics network, and its gradients
are clipped separately. Training it therefore leaves every other weight
unchanged, which a test checks exactly. In the piranha probe, earlier variants
that fed the objective into the tactics context, or let its loss train the LSTM,
lowered success (41 and 43 of 60 against 52 without the loss).

The stance history now records distinct tactics, not the last eight frames: a
repeated stance refreshes the latest entry. It travels inside the carried
`WorldModelState` with the LSTM state, and demonstration refreshes store it
per row. Batched fitting therefore trains it, and callers carry it without any
hidden per-model state.

With `learned_skill_goals`, the objective's goal replaces any supplied skill
goal inside the model. The scripted selector then only labels training data, in
Block and, through the runtime contract, in Full SMB playback. Evaluation
reports `strategy_by_family`, the objective's accuracy against the scripted
goal. The recipe trains the objective (`strategy_loss_weight: 0.5`) but keeps
scripted goals until a full-volume run shows learned goals match them.

Evidence so far comes from demonstration-fit probes, each evaluated with the
same weights under scripted and then learned goals. The objective matched the
scripted goal on 90–93% of frames in `chained_obstacles`, `mixed_section`,
`stair_gap` and `full_smb_opening_proxy` (10 held-out layouts each, all solved
both ways). In the four enemy families above it matched on 84–99% of frames,
and learned goals reproduced every scripted result. For piranha (60 layouts) it
matched on 98% of frames, with 50 successes both ways. Learned goals changed no
outcome in these probes; the full-volume curriculum remains the test.
