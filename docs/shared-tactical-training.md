# Shared tactical training

## Joint hierarchy training and prerequisite stages

All basic skill families now supervise tactical and strategic intent alongside
the action and duration losses. Clearing a gap teaches `advance`; waiting for
or riding a moving platform teaches `hold_area`. Successful demonstrations and
physics-based teachers label decisions. The selected intent is retained through
the committed skill and its landing release, so a jump's flight still trains
the higher layers. Motor-action compatibility is supervised only when the
executor allows a fresh decision, avoiding contradictory targets during a jump.

Strategy has two supervised outputs: the current skill objective and the
strategic intent. The intent loss trains the strategy history encoder directly,
and its predicted distribution conditions tactics. Tactics continues to
condition the skill/action policy. The objective head retains its separate
gradient clipping. The production recipe enables `learned_skill_goals`, so the
predicted objective chooses the skill goal at execution; scripted goals remain
training targets and evaluation references. Every minibatch updates all layers
together, including during bootstrap and rehearsal.

Four explicit families add continuous sequences in one world and recurrent
episode:

| Family | Sequence | Prerequisites |
| --- | --- | --- |
| `tactics_bridge_sequence` | Wait, board, hold while riding, exit | `wait_timing`, `bridge_mount`, `bridge_dismount` |
| `tactics_obstacle_sequence` | Approach, clear enemy, climb obstacles, clear enemy | `enemy_hop`, `tall_pipe_jump`, `enemy_patrol` |
| `strategy_bridge_then_gap` | Wait/board/ride/exit, jump a gap, climb to the goal | `tactics_bridge_sequence`, `single_gap`, `stair_climb` |
| `strategy_mixed_sequence` | Clear enemy, jump gap, clear enemy, climb | `tactics_obstacle_sequence`, `single_gap` |

The tactical sequences reuse the existing randomized bridge and obstacle
generators. The bridge/gap strategy sequence extends the world beyond the
bridge, switches to terrain traversal after crossing, and retains the same
memory throughout. Oracle reachability checks the complete sequence against
the normal final goal; reaching an intermediate obstacle does not end it.

With `hierarchy_curriculum` enabled (the production recipe), prerequisites must
meet the held-out family success gate in every difficulty bin before a sequence
unlocks. The gate is currently 90%. Explicit hierarchy families additionally
require 90% tactical and strategic-intent accuracy. Existing `chained_obstacles`,
`chained_enemy_gauntlet`, `mixed_section`, and `full_smb_opening_proxy` also have
prerequisites. See `FAMILY_PREREQUISITES` in `hierarchy.py` for the full graph.

Locked families are excluded from bootstrap data, rollout sampling, rehearsal,
and failure replay. Their demonstrations are generated only after unlocking.
Held-out evaluation still measures all families. Unlocks are retained to avoid
oscillating between curricula; basic skill retention continues alongside the
sequences. Checkpoints store mastery and unlock state, and logs report active,
locked, and newly unlocked families. A weights-only initialization still needs
fresh held-out evidence before unlocking sequences.

`strategy_intent_loss_weight` defaults to 0.5. `strategy_loss_weight` controls
skill-objective supervision separately. Evaluation reports
`strategy_intent_by_family` alongside `tactics_by_family`, `strategy_by_family`,
and whole-sequence success. Older demonstration caches require regeneration
or refreshing every included family to obtain the new intent labels. Older
policy/actor weights initialize the added intent-to-tactics projection at zero.

These are implementation changes; their effect on learning has not yet been
measured in a new training run. A running process must be restarted to load them.

## Earlier implementation and measurements

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

Demonstrations carry tactical action sets and explicit duration-consumption
masks; the joint-learning tool rejects caches built by other teacher code. No
model architecture or checkpoint shape changes are needed. An already running trainer must be
restarted to load this implementation.

## Repositioning families and enemy wait routes

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
