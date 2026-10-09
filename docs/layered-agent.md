# Layered SMB agent

The runtime hierarchy is **strategy → tactic → skill → predictive executor**.
Two layers are learned; the executor is a controller. Strategy is an externally selected objective.

## Information passed between layers

| Component | Inputs | Output |
|---|---|---|
| Vision | Screen pixels | Mario, objects, surfaces, gaps, support/contact |
| Scene memory | Encoded scene every four frames and at decisions, elapsed time, visual camera shift, previous LSTM state | Next-boundary scene plus jointly predicted action-end time and platform positions |
| Tactic | Strategy, current and predicted scene, held tactic/age, tactic memory | Persistent categorical tactic, termination probability, option values |
| Skill | Tactic, strategy context, current scene, memory prediction, last 16 skill commands, measured execution feedback | Run/jump/hold mode and a relative destination `(x, y)` |
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
Mario motion model to select the initial hold and correct the flight.
The controller chooses the buttons and durations; the skill chooses the target.

## Memory and timing

The existing scene LSTM updates every four frames and at every skill decision.
Elapsed frames and observed camera displacement accompany the scene summary.
Its next-boundary scene prediction is re-encoded for skill and tactic; this
expectation is made before choosing the command, so it is not action-conditioned.

The same LSTM jointly predicts **the next action's end scene and how many physical
frames remain until that end**. Its platform readout predicts displacement at
that same endpoint, with spatial uncertainty and track-visibility confidence.
Both skill and tactic receive the predicted timing with the expected scene.
Duration is a learned positive continuous output with its own uncertainty, not
a choice among preset horizons. It has no 64-frame cap: saved action traces
already contain 168-frame commands, ground execution permits 192 frames, and
jump-button hold length does not include the whole flight.

At a decision, the label is the state and elapsed time at the next decision.
Periodic memory ticks within that action are trained toward the same endpoint
with the remaining duration. A completed terminal action also supplies its
final observed state, even when no next decision is made. Unfinished actions
cut off by the episode budget are censored, not taught as completed actions.
Platform labels follow visual identity and compensate camera motion. Missing
or clipped identities have no position target. No separate motion predictor
is used. These predictions inform the learned destination selection.

Ground movement uses **eight-frame model-predictive control**: simulate Mario's
response to candidate button sequences, apply only the first button, then observe
and replan. A terminal braking-distance cost accounts for momentum. The executor
does not reconstruct terrain, simulate collisions or hazards, prevent walking off
edges, or reject a requested destination as unsafe. Choosing a supported landing
and any necessary clearance waypoint belongs to skill.

Tactic remains an option-critic: termination is checked at each action
boundary, while a new tactic is selected only when no tactic is held or the
old one ends. Tactic memory updates on those starts. Tactic receives its
held category and age in both frames and decisions.

Skill selects a destination at each action boundary. **Jump means take off now**;
any acceleration, positioning or clearance preparation is a separate skill-issued
`run` target. The controller searches jump holds of 1–32 frames using Mario's
button-response model, without inserting a run-up or waiting for motion samples.
It uses the current motion estimate, or its initialized estimate when no history
is available. Crossing the requested height on descent scores position error;
it does not certify a safe path or supported landing. When no candidate reaches
the point, the controller executes its closest immediate jump attempt. The flight
model has a 96-frame horizon and contains no terrain or hazard simulation.

Per-frame observations correct Mario's motion estimate and steering. Subsequent
measurements refine the initial estimate without postponing takeoff.
Once A is released, it cannot be pressed again in the same flight. Separate jumps
have a physical button release between them; if A is still down, this necessary
release frame precedes the new takeoff. An airborne request releases A and steers,
rather than starting another jump in midair. Visual contact after takeoff ends
the flight without reconstructing collision geometry. A run completes on arrival
with low residual speed or after 192 frames. Missing Mario observations release
buttons and end execution.

A destination on a uniquely identified moving platform or walker follows that
observed identity and its requested offset. Losing the identity retains the last
point; it never silently acquires a different object. No handwritten patrol or
hazard forecast changes the command. Camera motion is estimated from observed
landmarks; featureless scrolling uses Mario's button-response displacement until
landmarks return. This dead reckoning cannot independently verify progress.

If observed displacement remains within two pixels in both axes for 24 frames,
an active movement ends with `no_progress`. Skill receives eight explicit values:
feedback present, no progress, arrival, elapsed time, horizontal/vertical
displacement, and horizontal/vertical distance remaining. Training records the
same feedback supplied during execution. This lets skill learn to choose another
destination after a failed attempt. The controller does not invent a new target.
Hold commands retain their separate platform-relative controller.

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
  and policy targets. Skill also learns next-action-end scene, duration and platform prediction losses.
- **Tactic:** tactic transformer and tactic memory. Skill, scene encoder, and
  scene memory stay frozen. Teacher labels tactic choice/termination and return
  targets train the critic.

Teacher-controlled execution decreases across imitation rounds. Reward rounds
use sampled choices, a clipped policy-gradient objective, and an entropy bonus.
Tactic reward updates additionally use the option-critic termination objective
once the critic meets its readiness threshold. Reward rounds freeze shared
scene/memory modules and update only the selected layer.

The spatial teacher probes the certified route and restores the simulator
state. A jump targets the first landing/stomp. A run targets its bounded endpoint
or reached objective, stopping before the next jump in the demonstration. This
provides explicit approach waypoints, including clearance around overhead ledges.
Supported target centers are clamped within the observed-in-training support's
edges with room for Mario's width. Moving-support targets use the support's
current pose, so passive carry does not become a fixed world destination.
A hold targets the current spot. `goal_following_v2` checkpoints use immediate takeoff. Older executor checkpoints
can warm-start, but their skill/tactic qualifications are cleared. Checkpoints
without execution feedback initialize its new inputs with zero influence.
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

Skill uses **50 scene families** for local destination selection: individual
bridge holds/mounts/dismounts, enemy encounters, platform traversal, local
recovery, supplied-tactic choice clones and the 17 action families. Each
supplies spatial destination labels instead of button labels at the skill stage.
The choice clones deliberately reuse one scene with different supplied tactics;
they test following that input, rather than inferring a hidden assignment.

`choice_alternate_route` and `low_choice_alternate_route` belong to tactics.
Their local maneuvers have separate skill families:

| Skill family | Supplied tactic | Local destination |
|---|---|---|
| `skill_overhead_climb` | Climb forward | First overhead ledge, including any required reverse approach |
| `skill_raised_climb` | Climb forward | Next higher ledge |
| `skill_gap_descent` | Descend forward | Lower raised platform across the gap |
| `skill_ledge_descent` | Descend forward | Floor beyond the last ledge |
| `skill_lower_descent` | Descend forward | Lower floor beneath the raised route |
| `skill_lower_step_climb` | Climb forward | Step leading out of the lower route |
| `skill_lower_exit_climb` | Climb forward | Upper exit ledge from the step |

Each starts at its own maneuver, carries one local route destination, and ends
on supported arrival there. It does not require completing the rest of the level.
The full `choice_alternate_route` uses `max_points` and a visible upper-path
coin, which must be collected before finishing. That strategy input distinguishes
it from the direct `speed_run` choice instead of asking tactics to infer a hidden
route assignment from identical inputs. Executor limitations still affect these
local tasks; splitting the curriculum does not change execution behavior.

The lower full route likewise has a visible lower-floor coin and a `max_points`
objective. `choice_advance` finishes at its first supported far-side landing.
`choice_hold_area` only holds until its timed completion; departure timing belongs
to tactic lessons such as `wait_timing`, where a moving platform provides an
observable reason to proceed. Repeated movement commands share stall history,
including zero-distance requests, so their next skill decision receives no-progress
feedback. Deliberate holds reset that history.

Stomp destinations refer to the enemy's current center and top; the executor
tracks that visible target as it moves. While jump remains pressed, flight
feedback can extend or shorten its hold up to 32 total frames. After release it
cannot press jump again during the same flight. Vision decoding checks contact
predictions against the visible feet/support relationship before policies or
controllers consume them.
Changes in Mario's detected width are excluded from velocity corrections so a
changing silhouette cannot turn a leftward jump into an apparent rightward drift.

Two dedicated families, `skill_enemy_bypass` and `skill_enemy_bypass_back`,
require an immediate jump and supported landing beyond an enemy that remains
alive. Stomping it fails the episode.

For each strategy-route combination, the teacher evaluates both its default
jump selection and a bypass preference that selects non-stomping holds where
available. Speed run chooses the fastest complete measured route; max points
chooses the most points, breaking ties by time. Bypassing is therefore a real
candidate, without forcing it when a stomp happens to be faster.

Tactic uses **31 families**, defined by `TACTIC_TRAINING_FAMILIES`:

| Group | Families |
|---|---|
| Strategy (14) | Each of seven tactics under `speed_run_` and `max_points_` |
| Composed levels (8) | `chained_obstacles`, `chained_enemy_gauntlet`, `full_smb_opening_proxy`, `mixed_section`, `tactics_bridge_sequence`, `tactics_obstacle_sequence`, `tactics_bridge_then_gap`, `tactics_mixed_sequence` |
| Scene-driven routes/responses (4) | `upper_route`, `lower_route`, `dead_end_retreat`, `monster_retreat` |
| Alternate routes (2) | `choice_alternate_route`, `low_choice_alternate_route`, under `max_points` with visible points on the required path |
| Waiting/proceeding (3) | `moving_bridge`, `wait_timing`, `piranha_avoidance` |

These families teach tactic selection and termination with the skill
layers frozen. All 31 are excluded from skill training and skill evaluation.
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
