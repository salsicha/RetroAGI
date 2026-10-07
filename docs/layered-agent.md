# Layered SMB agent

The runtime hierarchy is **strategy → tactic → skill → predictive executor**.
Two layers are learned; the executor is a controller. Strategy is an externally selected objective.

## Information passed between layers

| Component | Inputs | Output |
|---|---|---|
| Vision | Screen pixels | Mario, objects, surfaces, gaps, support/contact |
| Scene memory | Encoded scene at each action boundary, previous LSTM state | Predicted scene at the next boundary |
| Tactic | Strategy, current and predicted scene, held tactic/age, tactic memory | Persistent categorical tactic, termination probability, option values |
| Skill | Tactic, strategy context, current scene, memory prediction, last 16 skill commands | Run/jump/hold mode and a relative destination `(x, y)` |
| Executor | Spatial destination and per-frame vision | One of six emulator button combinations |

The seven tactics are `advance`, `retreat`, `climb_forward`,
`climb_backward`, `descend_forward`, `descend_backward`, and `hold_ground`.
The tactic token is one-hot with no separate direction field. Forward means
right; backward means left. The strategy switch separately carries the goal side.

A skill command names a location relative to the center of Mario's feet at
that decision: x is positive right, y is positive down, in integer pixels.
The local range is x −256…256 and y −240…240. `jump(40, -24)` names a landing
point 40 pixels to the right and 24 pixels up. `run` can include grounded
movement or airborne steering after a timed jump-button plan has ended.
`hold` anchors the current platform-relative spot and ignores x/y.

The skill uses a transformer with positional encoding and a context of its
last **16 used commands**, including teacher commands during training. History
is newest first, marked by decision age, and reset between episodes. Its
three output heads choose movement mode and pixel coordinates. Coordinates
are categorical bins internally; the message delivered to the executor is a
spatial vector (mode one-hot plus normalized x/y), not a tactic category.

Strategy context distinguishes otherwise identical inputs with conflicting
demonstrations: the same retreat or climb tactic can require different
destinations under speed run and max points.

There is no action network. The executor retains a run target and evaluates
short acceleration/braking trajectories every frame. Jump targets use the
terrain/hazard predictor to select the initial hold and correct the flight.
The controller chooses the buttons and durations; the skill chooses the target.

## Memory and timing

At action boundaries, scene memory updates from the scene encoder's summary.
It predicts the next boundary's scene and that prediction is re-encoded for
skill and tactic. The prediction is currently made before the next command
is selected: it is not an action-conditioned counterfactual world model.

Tactic remains an option-critic: termination is checked at each action
boundary, while a new tactic is selected only when no tactic is held or the
old one ends. Tactic memory updates on those starts. Tactic receives its
held category and age in both frames and decisions.

Skill selects a destination at each action boundary. A checked jump remains
one executor maneuver through any required run-up, takeoff, button release and
flight to landing. The skill supplies the landing point; the executor searches
collision-checked approaches of up to 64 frames when an immediate jump cannot reach it.
The selected 1–32 frames describes the jump-button hold, not the whole flight.
Landing interrupts a plan when the contact detector and visible support under
Mario's feet agree. Side contact with a ledge cannot interrupt the jump.
An airborne command following an interrupted or unchecked plan releases the
preceding jump hold. Separate jumps always have a physical button release
between them, even if landing interrupted the previous plan.
A run completes on arrival with low residual speed, or reports a blocked path
or timeout. Arrival requires less than one pixel of error; one-pixel requests
must produce actual progress. Dense acceleration candidates and passive braking
avoid small-target stalls. Zero-distance targets remain stationary. Ground
prediction projects platform support and carry, binds points on moving supports
to their visual tracks, and hands boarding back to skill before chasing an old
shore waypoint. Geometric support resolves bridge/ground classification errors;
bridge-occluded floor edges are excluded from camera-motion estimates.

Before a grounded jump, the executor collects visual motion measurements and
checks the full body trajectory against visible walls, ceilings, supports and
moving hazards. It searches holds of 1–32 frames over a 96-frame horizon using
the shared NES motion model. It forecasts tracked objects from camera-corrected
visual motion, includes uncertainty for lethal hazards, and checks eight frames
after arrival for an approaching enemy. Walker stomps are allowed when the
destination identifies the predicted contact; monsters cannot be stomped.
When the accepted trajectories identify one walker, the executor binds the
stomp to that visual track. Camera motion and a patrol reversal preserve the
association. During the jump it predicts contact with the updated enemy
position and velocity, and searches steering and remaining hold durations
when the current trajectory would miss. This behavior applies to any family
requesting a stomp, without a family name or enemy ID in the skill command.
An ambiguous initial target retains fixed-point execution. Losing an acquired
track reports `lost_target`, never silently selects a different enemy, and
does not count an ordinary floor landing as completing the stomp. Actual
contact still comes from visual landing feedback and the scenario's stomp
event; a predicted interception alone cannot complete an objective.
An unverified takeoff waits and exposes `observing_motion` or
`no_safe_trajectory` through `AgentStep.execution_status`.

During flight, observations update the trajectory check each frame. A safe
correction may change steering or the remaining hold within the same jump;
once released, A cannot be pressed again in that flight. If no safe continuation
is found, the executor preserves the committed maneuver and reports
`no_safe_continuation`, rather than treating an unrelated short landing as
success. Missing Mario observations release buttons and end the maneuver.
This is a bounded visual model, not a guarantee about unseen terrain, future
enemy turns or inaccurate detections. The executor reads decoded visual geometry and its own button history, never
simulator state or teacher trajectories. On featureless scrolling terrain it
dead-reckons displacement from executed buttons until landmarks return; this
model estimate is not an independent visual measurement.

The monster curriculum teacher now has an approach phase for an offscreen,
sleeping monster. It advances only while a forward frame plus stopping distance
stays behind the tunnel lip and leaves at least 40 pixels of supported retreat
space inside the camera boundary. Once the monster is visible, the teacher
returns to waiting, retreating and selecting a safe crossing. These are training
labels for the existing tactic/skill interfaces; the runtime executor does not
receive hidden monster activation state. Existing tactic weights are unchanged.

## Hold-ground control

The action menu is the six button plans (`NOOP`, `RIGHT`, `RIGHT_JUMP`,
`LEFT`, `LEFT_JUMP`, `JUMP`) plus the executor's `HOLD_GROUND` controller.
The controller reads vision each frame, tracks the supporting platform, and
corrects horizontal drift with left/right/neutral buttons. The anchor moves
with the platform and persists across consecutive hold plans. Another action
clears it. Remembered platform width handles clipping of an edge at the screen
boundary. Missing/ambiguous support releases buttons rather than creating a
new anchor. Tracking is visual and has no simulator object IDs.
When both platform edges are clipped, there is no usable horizontal landmark;
the controller releases buttons rather than treating the viewport edge as a
world anchor. Consecutive holds keep their anchor, but the agent reconsiders
whether to continue holding every frame.

## Training

Train **skill → tactic**. Skill can start from scratch or a compatible warm start;
tactic requires a qualified skill checkpoint. Only the stage's parameters change:

- **Skill:** scene encoder, scene memory, and skill transformer. Teacher supplies
  tactic and spatial destinations. The predictive executor executes both teacher
  and policy targets. Skill also learns the scene prediction loss.
- **Tactic:** tactic transformer and tactic memory. Skill, scene encoder, and
  scene memory stay frozen. Teacher labels tactic choice/termination and return
  targets train the critic.

Teacher-controlled execution decreases across imitation rounds. Reward rounds
use sampled choices, a clipped policy-gradient objective, and an entropy bonus.
Tactic reward updates additionally use the option-critic termination objective
once the critic meets its readiness threshold. Reward rounds freeze shared
scene/memory modules and update only the selected layer.

The spatial teacher probes the certified route and restores the simulator
state. A jump command targets the first landing/stomp; a run command targets
the endpoint of a complete bounded maneuver or reached objective. Acceleration
before a jump belongs to that jump's landing label, rather than a short run
label that loses momentum. Boarding targets use the support's current position,
so passive carry does not become a world destination. A hold targets the current
spot. Executor v2 checkpoints use these semantics; older weights can warm-start
training but their previous skill/tactic qualifications are cleared.
Uncertified recovery states are not used as skill demonstrations. The teacher
can read simulator state during training; deployed networks cannot.

Scenario completion is checked every simulation frame: reaching the final
goal ends the episode immediately once its required contact and task conditions
are met. Intermediate route destinations are retired on arrival, including a
destination already under Mario at spawn or a segment transition. Backtracking
does not reactivate a completed destination within that segment. A generated
skill endpoint completes a movement command, not the scenario's final goal.

## Curriculum

Skill includes 17 isolated maneuver families (their historical `action_` names remain):

| Families | Maneuver |
|---|---|
| `action_walk`, `action_walk_back` | Walk right/left |
| `action_jump_gap`, `action_jump_gap_back` | Jump across a gap right/left |
| `action_climb`, `action_climb_back` | Jump onto a solid step right/left |
| `action_platform_up`, `action_platform_up_back` | Jump onto a raised thin platform right/left |
| `action_descend`, `action_descend_back` | Walk off a ledge right/left |
| `action_jump_down`, `action_jump_down_back` | Jump to a lower landing across a gap right/left |
| `action_stomp`, `action_stomp_back` | Jump onto an enemy on level ground right/left |
| `action_stomp_up`, `action_stomp_up_back` | Jump onto an enemy on a raised platform right/left |
| `action_wait` | Preserve a spot on a moving platform |

The new jump families certify an immediate jump through its first landing,
without accepting a walking-off route or a retry as a successful jump.
An optional parameter sweep remains available for explicitly selected small
skill families. Normal training samples the full skill curriculum and adds
extra layouts for weak families.

Skill uses **45 scene families** for local destination selection: individual
bridge holds/mounts/dismounts, enemy encounters, platform traversal, local
recovery, supplied-tactic choice clones and the 17 action families. Each
supplies spatial destination labels instead of button labels at the skill stage.
The choice clones deliberately reuse one scene with different supplied tactics;
they test following that input, rather than inferring a hidden assignment.

Two dedicated families, `skill_enemy_bypass` and `skill_enemy_bypass_back`,
require an immediate jump and supported landing beyond an enemy that remains
alive. Stomping it fails the episode.

For each strategy-route combination, the teacher evaluates both its default
jump selection and a bypass preference that selects non-stomping holds where
available. Speed run chooses the fastest complete measured route; max points
chooses the most points, breaking ties by time. Bypassing is therefore a real
candidate, without forcing it when a stomp happens to be faster.

Tactic uses **29 families**, defined by `TACTIC_TRAINING_FAMILIES`:

| Group | Families |
|---|---|
| Strategy (14) | Each of seven tactics under `speed_run_` and `max_points_` |
| Composed levels (8) | `chained_obstacles`, `chained_enemy_gauntlet`, `full_smb_opening_proxy`, `mixed_section`, `tactics_bridge_sequence`, `tactics_obstacle_sequence`, `tactics_bridge_then_gap`, `tactics_mixed_sequence` |
| Scene-driven routes/responses (4) | `upper_route`, `lower_route`, `dead_end_retreat`, `monster_retreat` |
| Waiting/proceeding (3) | `moving_bridge`, `wait_timing`, `piranha_avoidance` |

These families teach tactic selection and termination with the skill
layers frozen. All 29 are excluded from skill training and skill evaluation.
Local sequences such as `platform_chain` and `stair_gap` remain skill practice
for successive reachable destinations; individual bridge holds, mounts and
dismounts remain skill practice for carrying out the selected maneuver.

Strategy sibling families share layouts but reward different routes. The
hold-ground scene waits at the starting spot for 64 frames before traversing.
Speed-run qualification requires finishing within the teacher-relative time
limit; max-points qualification requires the specified points objective.

## Evaluation and checkpoints

Every family must meet its configured success threshold (90% by default).
Tactic additionally has a termination-boundary agreement gate. `passed.pt`
qualifies the next stage; `best.pt` is best overall; `last.pt` is latest.
Validation drawn from the exhaustive generator space is not an unseen-layout
generalization test. Report complete-sweep and composed-task performance
separately from the small validation sample.

Checkpoints store observation layout, strategy/tactic vocabularies, executor
buttons, skill coordinate schema, history length, and predictive-executor version.
Legacy spatial checkpoints migrate by discarding `action.*` weights while
preserving skill, tactic, scene encoder, and memories. New checkpoints contain
no action network and list only skill/tactic as trained layers. Incompatible
observation or token layouts are rejected. Older skill strategy-context inputs
can still migrate with zero-initialized added context weights.
Validation history includes per-family end reasons and training label coverage.
An unsuccessful single-jump landing is a terminal failed attempt, not a timeout.

```bash
retroagi-block-smb train-layer --learner skill --output artifacts/block_smb/skill
retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/skill/passed.pt --output artifacts/block_smb/tactic
retroagi-block-smb exam-layer --learner skill --checkpoint artifacts/block_smb/skill/passed.pt
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
```

Full SMB uses the same policy/executor with its own vision weights. Passing
Block SMB tests alone does not establish transfer or learned-vision robustness.
