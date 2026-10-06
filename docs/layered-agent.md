# Layered SMB agent

The runtime hierarchy is **strategy → tactic → skill → action → executor**.
Three layers are learned. Strategy is an externally selected objective.

## Information passed between layers

| Component | Inputs | Output |
|---|---|---|
| Vision | Screen pixels | Mario, objects, surfaces, gaps, support/contact |
| Scene memory | Encoded scene at each action boundary, previous LSTM state | Predicted scene at the next boundary |
| Tactic | Strategy, current and predicted scene, held tactic/age, tactic memory | Persistent categorical tactic, termination probability, option values |
| Skill | Tactic, strategy context, current scene, memory prediction, last 16 skill commands | Run/jump/hold mode and a relative destination `(x, y)` |
| Action | **Only the spatial skill command** | Executor action and duration, 1–32 frames |
| Executor | Proposed plan, destination, per-frame vision for local motion/landing/hold feedback | One of six emulator button combinations |

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
are categorical bins internally; the message delivered to action is a
spatial vector (mode one-hot plus normalized x/y), not a tactic category.

Strategy context distinguishes otherwise identical inputs with conflicting
demonstrations: the same retreat or climb tactic can require different
destinations under speed run and max points. This context never reaches the
action network.

Action is a feed-forward network with positional features of that spatial
vector. It cannot receive a scene, tactic, LSTM state, memory prediction, or
choice history. Its input restriction also applies during training. Spatial
selection and obstacle interpretation belong to skill; action translates the
requested displacement into an executor plan.

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
one executor maneuver through takeoff, button release and flight to landing.
The proposed 1–32 frames describes the jump-button hold, not the whole flight.
Landing interrupts a plan when the contact detector and visible support under
Mario's feet agree. Side contact with a ledge cannot interrupt the jump.
An airborne command following an interrupted or unchecked plan releases the
preceding jump hold. Separate jumps always have a physical button release
between them, even if landing interrupted the previous plan.
Visual destination tracking can finish a run before the proposed
duration expires.

Before a grounded jump, the executor collects visual motion measurements and
checks the full body trajectory against visible walls, ceilings, supports and
moving hazards. It searches holds of 1–32 frames over a 96-frame horizon using
the shared NES motion model. It forecasts tracked objects from camera-corrected
visual motion, includes uncertainty for lethal hazards, and checks eight frames
after arrival for an approaching enemy. Walker stomps are allowed when the
destination identifies the predicted contact; monsters cannot be stomped.
An unverified takeoff waits and exposes `observing_motion` or
`no_safe_trajectory` through `AgentStep.execution_status`.

During flight, observations update the trajectory check each frame. A safe
correction may change steering or the remaining hold within the same jump;
once released, A cannot be pressed again in that flight. If no safe continuation
is found, the executor preserves the committed maneuver and reports
`no_safe_continuation`, rather than treating an unrelated short landing as
success. Missing Mario observations release buttons and end the maneuver.
This is a bounded visual model, not a guarantee about unseen terrain, future
enemy turns or inaccurate detections. No simulator state, teacher action,
ViT embedding or LSTM output is added to the action network's input.

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

Train **action → skill → tactic**, starting each stage from a passing lower
checkpoint. Only the stage's parameters change:

- **Action:** the destination-to-plan network. Teacher supplies spatial skill
  commands and exact button/duration labels; scene encoder and memories stay frozen.
- **Skill:** scene encoder, scene memory, and skill transformer. Teacher supplies
  the tactic and spatial destinations; action stays frozen. Skill also learns
  the scene prediction loss.
- **Tactic:** tactic transformer and tactic memory. Skill, action, scene encoder,
  and scene memory stay frozen. Teacher labels tactic choice/termination and
  return targets train the critic.

Teacher-controlled execution decreases across imitation rounds. Reward rounds
use sampled choices, a clipped policy-gradient objective, and an entropy bonus.
Tactic reward updates additionally use the option-critic termination objective
once the critic meets its readiness threshold. Reward rounds freeze shared
scene/memory modules and update only the selected layer.

The spatial teacher probes the certified route and restores the simulator
state. A jump command targets the first landing/stomp; a run command targets
the endpoint of its bounded movement. A hold command targets the current spot.
Uncertified recovery states are not used as skill demonstrations. The teacher
can read simulator state during training; deployed networks cannot.

For a specific spatial command, action supervision preserves the exact plan's
duration. Other certified jump holds may land elsewhere and are not substituted.

## Curriculum

Action uses 17 isolated maneuver families:

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
The action stage enumerates the declared discrete parameter space (floating
speeds at 0.01 increments). Every generated reachable layout is played each
round, with family-balanced losses. Rejected combinations and counts are
recorded in `combinations.json`; enumeration is coverage of these generators,
not all possible incoming motion or arbitrary levels.

Skill uses **all 75 registered scene families**, including every leftover
family: piranha avoidance, moving bridges, mounting/dismounting, enemy patrols,
landing enemies, occupied platforms, recovery, route-choice clones and composed
sequences, as well as the 17 action and 14 strategy families. Each supplies
spatial destination labels instead of button labels at the skill stage.

Two dedicated families, `skill_enemy_bypass` and `skill_enemy_bypass_back`,
require an immediate jump and supported landing beyond an enemy that remains
alive. Stomping it fails the episode. The existing `piranha_avoidance` family
covers both clearing a plant and waiting for retraction before crossing.

For each strategy-route combination, the teacher evaluates both its default
jump selection and a bypass preference that selects non-stomping holds where
available. Speed run chooses the fastest complete measured route; max points
chooses the most points, breaking ties by time. Bypassing is therefore a real
candidate, without forcing it when a stomp happens to be faster.

Tactic uses 14 strategy families: each of seven tactics under `speed_run` and
`max_points`. Sibling families share layouts but reward different routes. The
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
actions, skill coordinate schema, history length, and the action input contract.
Older tactic-to-action checkpoints are rejected even where tensor widths
happen to match. Retrain action, then skill, then tactic for this architecture.
Spatial checkpoints from before strategy context are migrated by appending
zero-initialized context weights. All existing weights, including action,
are preserved; the new context is learned in subsequent skill training.
Validation history includes per-family end reasons and training label coverage.
An unsuccessful single-jump landing is a terminal failed attempt, not a timeout.

```bash
retroagi-block-smb train-layer --learner action --output artifacts/block_smb/action
retroagi-block-smb train-layer --learner skill --init artifacts/block_smb/action/passed.pt --output artifacts/block_smb/skill
retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/skill/passed.pt --output artifacts/block_smb/tactic
retroagi-block-smb exam-layer --learner action --checkpoint artifacts/block_smb/action/passed.pt
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
```

Full SMB uses the same policy/executor with its own vision weights. Passing
Block SMB tests alone does not establish transfer or learned-vision robustness.
