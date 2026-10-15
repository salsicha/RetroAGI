# Layered SMB agent

The runtime hierarchy is **strategy → tactic → skill → predictive
executor**. Two layers are learned by imitation and reward (tactic, skill);
the executor is a controller; strategy is an externally selected objective.
Beside them are two prediction networks: the **world model** (an LSTM run
once at the start of each action, after the skill has chosen it, to predict
that action's outcome) and the **target trackers** (one per enemy, run every
frame, predicting when and where the enemy and Mario will collide). The
skill and tactic layers read no prediction (user, 2026-10-15). While the
skills and tactics train in Block SMB, neither prediction network is used:
each "prediction" is the simulator's ground truth, and the networks are
trained apart from recorded truth.

Decisions behind this design, with their reasons and status, are listed in
[design decisions](design-decisions.md).

## Information passed between layers

| Component | Inputs | Output |
|---|---|---|
| Vision | Screen pixels | Mario, objects, surfaces, gaps, support/contact |
| Tactic | Strategy, current scene, held tactic/age, tactic memory | Persistent categorical tactic, termination probability, option values |
| Skill | Tactic, strategy context, current scene, last 16 actions, measured execution feedback | An action: a verb and a pointer to one object it sees |
| World model (LSTM; not used in Block SMB training) | The chosen action, the scene at its start, its own state from earlier actions | The action's successful outcome: where Mario's feet and its object end, how many frames it takes (for a wait, how long), the chance it succeeds |
| Target trackers (not used in Block SMB training) | Every frame's vision of each enemy and of Mario | When and where that enemy and Mario will collide, and whether as a stomp or a hit |
| Executor | The action's end as its destination (in Block SMB training: the teacher's ground-truth version of the action), per-frame vision | One of six emulator button combinations |

The seven tactics are `advance`, `retreat`, `climb_forward`,
`climb_backward`, `descend_forward`, `descend_backward`, and `hold_ground`.
The tactic token is one-hot with no separate direction field. Forward means
right; backward means left. The strategy switch separately carries the goal side.

The skill chooses an **action**: a verb (hold, run to, approach, land on,
jump over, stomp, back away) and a pointer to one of the objects the vision
reports (a surface, an enemy, a gap, a pipe, a block or a moving platform:
one of 60 scene slots). It chooses in order: the verb first (a verb with
nothing in view it can aim at is excluded), then the pointer reading the
verb (the decision token, adjusted by the verb, scored against each object's
scene token; absent objects and kinds the verb cannot aim at are excluded).
A floor is reported cut where an enemy stands on it, so each piece is the
floor between two threats, and an action aimed at a piece names where it
ends. The action's end, in the verb's mode, is the executor's destination: a
location relative to the center of Mario's feet (x positive right, y
positive down). In Block SMB training it is the teacher's version of the
action (`teacher_actions.carry_out`: the ground truth the world model will
predict). A hold anchors the current platform-relative spot and lasts as
long as its wait: one action whose length is the ground truth in training
(`SkillToken.frames`). A jump at a walking enemy aims where it really is when
Mario lands (`tracks.exact`), the place the trackers will predict.

The skill uses a transformer with positional encoding and a context of its
last **16 actions**, each with where it sent the executor (teacher commands
in training are kept as their mode's verb). History is newest first, marked
by decision age, and reset between episodes. It is taught by imitation: the
teacher's command named as an action (`teacher_actions`): its verb, and
(except hold) the object it is aimed at.

Strategy context distinguishes otherwise identical inputs with conflicting
demonstrations: the same retreat or climb tactic can require different
destinations under speed run and max points. It also tells a stomp from a
jump over: every lesson that must stomp an enemy is played under max points
(kills score), and under speed run the teacher jumps over an enemy whenever
that works (see the spatial teacher below).

There is no action network. The executor retains a run target and evaluates
short acceleration/braking trajectories every frame. Jump targets use the
Mario motion model to select the initial hold and correct the flight.
The controller chooses the buttons and durations; the skill chooses the target.

## Memory and timing

The skill and tactic layers read no prediction (user, 2026-10-15). The world
model's LSTM (`layered_policy.Memory`) does not update every few frames: it is
to run once at the start of each action, after the skill has chosen it, with
the chosen action as input, and predict that action's outcome. It is trained
apart from recorded ground truth and is not used while the skills and tactics
train; its current weights are kept, unused, until it is reshaped so.

The target trackers (`core/target_tracker.py`) run every frame. They are to
predict, for each enemy, when and where it and Mario will collide (a stomp or
a hit); the simulator's truth stands in until they are trained
(`stages/block_smb/collision_truth.py`). Watched frames before a lesson's
first decision let the tracks see enemies move.

In Block SMB training a wait is one action: the teacher's hold lasts until it
would do something else, or until a coming enemy is close enough that a jump
over it is due (`controller_teacher._dodge_wait`). The skill then picks the
next action, which it learns from the teacher.

Evaluation is Full SMB played many times, learning continuing on every
playthrough, the world model and the trackers included, in speed mode (time
to each checkpoint) and points mode. The measure is how few playthroughs it
takes before Mario plays the level successfully.

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

**In-flight re-prediction is off for Block SMB training**
(`smb_trajectory.REPLAN_IN_FLIGHT`, 2026-10-10). With it on, each frame of a
flight re-predicts the path from what vision shows and re-chooses the
steering and the remaining hold: that corrects the motion estimate, and
re-aims a jump at a walking enemy (a stomp) or a moving platform at where it
is now. Off, a flight plays the hold and steering chosen at
takeoff, then keeps steering toward the goal. Holding a spot on a moving
platform is the separate hold controller and is unaffected. The per-frame
search was half the cost of every teacher trial; with it off, a jump is set
entirely at takeoff by its steering and hold, which lets the teacher try
every jump Mario can make (below).
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

- **Skill:** the skill transformer (and, with `--train-reading`, the scene
  encoder). Teacher supplies the tactic and its own commands, labelled as
  actions. The executor executes the teacher's commands and the skill's own
  actions, which the teacher carries out (its best safe version of each, the
  ground truth the world model will predict; a choice it has none of loses the
  episode; `teacher_actions.carry_out`).
- **Tactic:** tactic transformer and tactic memory. Skill and scene encoder
  stay frozen. Teacher labels tactic choice/termination and return
  targets train the critic.

Teacher-controlled execution decreases across imitation rounds. Reward rounds
use sampled choices, a clipped policy-gradient objective, and an entropy bonus.

**No single values; ends explained by the scene.** No kind of action may end
at one value whatever the scene. Teacher qualification (before learning and
before every round) fails when, over every family's labels together, one end
of a verb (within 2 pixels) covers more than 40% of that verb's labels (at
least 6). A single family may concentrate on one end where its scenes call
for it (a pit nearly as wide as the longest jump is crossed by the longest
jump); whether they do is to be measured on the world model, which learns
ends from scenes of every family. After every round, the learner's own actions on held-out layouts are
checked family by family: a family where the learner's most common end is
above 40% and more than 20 points above the teacher's share collapses, and a
round with a collapsed family cannot pass the gate. The decision layers are
regularized: dropout of 0.1 in the skill and tactic transformers while
training (`PolicySettings.decision_dropout`) and weight decay 0.05
(`LayeredTrainConfig.weight_decay`).
Tactic reward updates additionally use the option-critic termination objective
once the critic meets its readiness threshold. Reward rounds freeze shared
scene/memory modules and update only the selected layer.

The spatial teacher probes the certified route and restores the simulator
state. A jump targets the first landing/stomp. A run targets its bounded endpoint
or reached objective, stopping before the next jump in the demonstration. This
provides explicit approach waypoints, including clearance around overhead ledges.
For static gap crossings and descents without live enemies, the spatial teacher
uses a supported takeoff waypoint near the edge and an inset landing point.
This accounts for the ground controller braking at a run destination, instead
of copying the takeoff of an uninterrupted button sequence. The teacher supplies
the preparation run separately from the jump and avoids zero-distance approach
loops. Immediate-jump action lessons keep their original contract.
The `enemy_patrol` spatial teacher certifies candidate run, stomp and bypass
destinations through a copy of the existing executor, initialized with the
current execution feedback. Simulator probes include actual braking, enemy
motion and the first landing or stomp; a missed destination or collision is not
a certified label. Candidate endpoints stay on the supported floor or a live
enemy. After a stomp, the next run targets floor height rather than Mario's
airborne height. The teacher selects a progressing command with separation
from surviving enemies, or a command that completes the existing level goal.
These checks are confined to training; the runtime executor has no new hazard
veto, collision model or search behavior.
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

**Variety around pits and enemies** (`stages/block_smb/threat_variety.py`).
Every family that teaches jumping over or past a pit or an enemy starts Mario
at varied distances from it, right next to it in about 40% of layouts (as in
Full SMB, where he can end up beside one), and follows it with a second pit
or enemy at a varied distance. The landing is then bounded, so the policy
learns to keep away from every threat during a jump, not the longest jump.
A following pit is cut into the landing floor (leaving at least 32 pixels of
floor past it); a following enemy stands or walks either way at up to 0.6
pixels per frame. Every enemy patrols exactly the platform it stands on, so it
turns only at an edge the vision can see. Single-jump lessons take the floor
between the two threats as their goal; longer lessons continue past both.
The isolated maneuvers, route lessons and older families also vary where
Mario starts and the size and place of what he jumps onto (steps, raised
platforms, pipes, the route platforms' widths, heights and places), so the
right destination is a different distance in each scene.

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

A climb completes on a supported landing anywhere on the step: a tall step's
landing can be at its near edge, and walking on from there would be a second
move under another tactic.
The new jump families certify an immediate jump through its first landing,
without accepting a walking-off route or a retry as a successful jump.
An optional parameter sweep remains available for explicitly selected small
skill families. Normal training samples the full skill curriculum and adds
extra layouts for weak families.

Skill uses **41 scene families** for destination selection under one supplied
tactic: local enemy encounters, platform maneuvers, holding a spot and all 17
action families. Each supplies spatial destination labels instead of button
labels at the skill stage. Any family that changes tactics within an episode
belongs to tactic training, regardless of its name or number of written schedule
segments. This includes changes produced dynamically by the teacher.

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
| `skill_upper_gap_entry` | Advance | First raised ledge across the first gap |
| `skill_upper_gap_exit` | Advance | Far raised ledge across the second gap |

Each starts at its own maneuver, carries one local route destination, and ends
on supported arrival there. It does not require completing the rest of the level.
The full `choice_alternate_route` uses `max_points` and a visible upper-path
coin, which must be collected before finishing. That strategy input distinguishes
it from the direct `speed_run` choice instead of asking tactics to infer a hidden
route assignment from identical inputs. Executor limitations still affect these
local tasks; splitting the curriculum does not change execution behavior.

The lower full route likewise has a visible lower-floor coin and a `max_points`
objective. `choice_advance` finishes at its first supported far-side landing.
`choice_hold_area` holds for 120 to 200 frames while one or two walkers, each
at its own speed, come at Mario, between pits, pipes and low brick rows 32 to
80 pixels from him. He holds where he is until a walker comes within its
trigger distance, a buffer around him just wide enough at that walker's speed
for a jump over it, then jumps over it and lands behind it, into no pit or
obstacle (`controller_teacher._dodge_in_area`; the jump is taught when the
same jump started 5 frames later would no longer pass it). Each wait is one
hold action whose length is the number of frames until the jump is due
(`SkillToken.frames`; the simulator's truth in training, the predictor's when
deployed). There is no fixed area: the lesson completes anywhere on his floor. Departure timing belongs to tactic lessons
such as `wait_timing`, where a moving platform provides an observable reason to
proceed. Repeated movement commands share stall history,
including zero-distance requests, so their next skill decision receives no-progress
feedback. Deliberate holds reset that history.
The full two-gap `low_choice_advance` route belongs to tactics; its two crossings
are separate skill lessons above. A new movement attempt gets its own observation
window before old stall history can cancel it, while completed zero-distance
requests still report failure. Supervised decisions following no-progress feedback
receive three times the imitation weight so corrective targets are not overwhelmed
by easy approach labels.

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

One picture cannot show an enemy's speed or direction, and at the first
decision of an episode the memory has seen only one. In `enemy_stomp` and both
bypass families the right landing depends on exactly that (a standing enemy
needs a longer jump than one walking toward Mario; an approaching enemy needs
a short approach, a retreating one a long chase). These layouts therefore
start with 12 pre-episode frames (`watch_frames`): the simulator runs them at
reset while Mario stands, and the agent watches their pictures, its tracks
(and target trackers) measuring each enemy's motion, but decides nothing. The episode, its routes and its frame budget start after
them.

For each strategy-route combination, the teacher evaluates both its default
jump selection and a bypass preference that selects non-stomping holds where
available. Speed run chooses the fastest complete measured route; max points
chooses the most points, breaking ties by time. Bypassing is therefore a real
candidate, without forcing it when a stomp happens to be faster.

Tactic uses **42 families**, defined by `TACTIC_TRAINING_FAMILIES`:

| Group | Families |
|---|---|
| Strategy (14) | Each of seven tactics under `speed_run_` and `max_points_` |
| Composed levels (8) | `chained_obstacles`, `chained_enemy_gauntlet`, `full_smb_opening_proxy`, `mixed_section`, `tactics_bridge_sequence`, `tactics_obstacle_sequence`, `tactics_bridge_then_gap`, `tactics_mixed_sequence` |
| Scene-driven routes/responses (4) | `upper_route`, `lower_route`, `dead_end_retreat`, `monster_retreat` |
| Alternate routes (2) | `choice_alternate_route`, `low_choice_alternate_route`, under `max_points` with visible points on the required path |
| Raised route (1) | `low_choice_advance`, under `speed_run`, preserving the upper path through two gaps |
| Waiting/proceeding (3) | `moving_bridge`, `wait_timing`, `piranha_avoidance` |
| Additional tactic switches (10) | `choice_retreat`, `stair_climb`, `platform_chain`, `retreat_recovery`, `tall_pipe_jump`, `bridge_wait`, `bridge_mount`, `bridge_dismount`, `stair_gap`, `enemy_on_platform` |

These families teach tactic selection and termination with the skill layer
frozen. All 42 are excluded from skill training and skill evaluation.
`choice_advance`, `enemy_gap` and `enemy_patrol` moved to the skill stage on
2026-10-11: once jumps stopped landing beyond the goal or a route platform,
their demonstrations no longer switched tactics. For
example, `choice_retreat` retreats then advances; bridge phases alternate
holding and advancing; `platform_chain` can change from advance to retreat
for the final landing. Local skill lessons must finish under one supplied tactic.

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

Spatial teacher certification applies to both skill and tactic layouts. A
button route is a proposal source: each destination is replayed through the
same ground and flight controllers used by the agent. The teacher labels a
command only when that execution reaches a supported destination, a required
contact, a route milestone, or the final goal. Failed proposals are repaired
with supported platform positions and edges. Raised landings can use a
separately certified run waypoint followed by an immediate jump from that
stopped position. Airborne commands account for the current post-contact
motion rather than assuming that the next tactic's direction is already
achievable. When the production vision model is available, accepted jump
proposals are also replayed through it, including intermediate jumps.

The teacher does not stop at the first command that works. Among the commands
that make the same move, it labels the one with the most **room**: during the
replay it measures, for every threat separately, the closest Mario's body
comes to it: each enemy he does not stomp, and each **pit edge** (the top
corner of a platform with nothing he may stand on beyond it; a platform the
lesson forbids counts as a pit). When he ends standing it also measures his
distance from the nearer edge of that platform. These cover the threat before
a jump (where he takes off), during it and after it (where he lands), and the
next pit or enemy as well as the one he jumps over. Commands are compared by
their closest threat, then their next closest, and so on, in two-pixel steps
up to 64 pixels: a threat they all share, such as the pit edge Mario starts
beside, does not hide the others. Two commands make the same move when they
end on the same platform, kill the same enemies, reach the same route mark
and leave Mario on the same side of every other enemy. A jump that lands back
on its own platform must also make at least as much progress, so the shortest
hop is not taught in place of a real approach. Among equal room, the label is
the destination where the jump actually lands (for a stomp, the enemy's
middle). Destinations beyond Mario's reach all give the same longest jump;
labelling the middle of them taught one value, 72 pixels, for every such
jump. When no tested destination is within 3 pixels of the landing, the
landing point itself is tried and used if it makes the same move.

**No teacher picks from a sparse list.** Every destination is considered:

- **Jumps.** A jump is set at takeoff by its steering (left, none, right) and
  how long the button is held (1 to 32 frames). Every destination within 128
  pixels, each pixel in x, at the height of every platform and enemy top,
  maps to one of these few dozen jumps by the executor's own takeoff rule
  (`smb_trajectory.best_holds`, an exact copy of its choice for many
  destinations at once). The teacher tries each jump once (the jump table);
  the label for a jump is, of the destinations that make it, the one nearest
  where it lands (for a stomp, the enemy's middle top).
- **Runs.** Every pixel Mario can walk to within 128 pixels is screened (where
  it ends, the progress it makes, its room) and the best six are tried. In
  the air, steering to every pixel within 32 pixels is tried.
- **Takeoffs and waits.** Running first (every length from 8 pixels up to the
  pit's edge, 24 pixels before a jump past an enemy, 32 pixels or holding
  still before a jump over one, 48 back to 128 forward below a raised
  platform) followed by every jump is screened: each jump's measured path is
  moved along the floor, the enemies moved by their patrol rule. The best are
  certified by trials. Running or waiting first is taught only when it gives
  at least 4 more pixels of room than jumping now, and no run shorter than 8
  pixels is taught: an enemy walking away, or a re-estimated takeoff, would
  otherwise have it asked again and again, a pixel at a time.
- **Routes.** The button routes behind the teacher's first proposal, and the
  sampler's check that a layout can be won, try every hold from 1 to 32
  frames (an older executor could play only 16 of them) and, for timing a
  plant, every run-on length.

The patrol, plant and monster teachers draw on the same jumps and runs.
Lessons whose point is one immediate jump keep their takeoff.

A run label follows the teacher's route until its next event (a jump, a
wait, a landing, the objective), up to 160 frames; a 64-frame limit made
every long run on flat ground the same 104 pixels. A proposed run of less
than 8 pixels on the ground is not taught: it is the route's run-up before a
jump, which a run that brakes on arrival never completes, so it was asked
again, a few pixels shorter each time. Every jump and run is ranked instead.

The move itself is chosen by the strategy when Mario stands on the ground with
a live enemy within 128 pixels ahead at about his height. Under max points, a
stomp that works now is taught over any other move. Under speed run, a jump
over the enemy is taught over a stomp, and passing now over waiting, unless
the waiting command lets the same jump pass with at least 4 more pixels of
margin. Where no jump over works now, the teacher tries waiting first:
holding still, or a step of 4 to 32 pixels toward the target. It takes the
wait after which a jump over keeps the most margin; only if none works does it
stomp. In enemy_hop, for example, Mario stops 48 pixels from a standing
Goomba, where every jump from rest lands on it; the teacher now steps 32
pixels closer and jumps over it. The landing_enemy walker, like enemy_stomp's,
patrols the whole visible floor: with short hidden patrol ends it turned
around while Mario was in the air, so 1-pixel differences of distance decided
between waiting, jumping over and stomping.

For isolated downward jumps, a visual contact correction can interrupt edge
preparation before its waypoint is reached. If ordinary destinations fail,
the teacher can certify a run toward a takeoff with at least two pixels of
body support, followed by a jump that completes the objective. It replays both
steps and checks actual support after preparation; progress alone never
qualifies this fallback. The skill still issues the run and jump separately.

Teacher-only checks preserve unfinished route destinations behind the
forward-only camera, keep point-scoring objectives active until their rewards
are collected, and avoid ending under a low ceiling beside an approaching
enemy or over a hidden plant's mouth. Pipe landings also need room to brake
outside the plant. Monster jumps must complete the crossing; approach, waiting
and retreat use separate destinations. None of these checks is an executor veto
or a policy input. The executor and its four-frame initial duration search are
unchanged.

Before any learning, `train-layer` qualifies the exact held-out layouts through
the production episode pipeline. It also qualifies each round's prepared
training layouts before updating weights. Each family draws its training
difficulties from its own random sequence, and a weak family's extra layouts
follow its usual ones, so every layout a round can use (at most
`train_layouts_per_family + focus_layouts` per family) is known in advance
and can be qualified before a run starts. These checks use the same episode
budget and 400-frame progress timeout as learner episodes, production vision,
and spatial teacher destinations. Every episode must finish with valid teacher
labels throughout. A failure stops the run and saves its layout and outcome in
`teacher_qualification/`; it never redraws the layout or enlarges its budget.
This applies to both skill and tactic training, with teacher destinations used
to qualify tactic layouts independently of the learned skill. The first fully
teacher-driven skill round reuses these demonstrations. Qualification covers
the sampled layouts, not all possible layouts or learner-created states. Its
report also counts decisions where the teacher's choice names no action.

To check the teaching by eye, `python -m retroagi.stages.block_smb.family_atlas
--out <folder>` pictures every skill family's layouts with each drawn parameter
at its lowest, middle and highest value, at each difficulty: where the enemies
and moving platforms go (blurred over the episode's frames), the teacher's
path through the production pipeline, and a heat spot where Mario's feet are
at each decision, marked by whether he stands, is on an enemy he has just
stomped, or is in the air. Decisions in the air, other than on a stomped enemy,
are faults.

When approaching a moving stomp target that is travelling away, the teacher
measures relative progress over the complete run command. Tiny stop/start
waypoints that merely match the enemy's travel are rejected as demonstrations;
the teacher selects an approach that actually closes the distance. This avoids
spending the episode budget following an enemy without reaching jumping range.

Speed-run deadlines are measured with the spatial teacher and production ViT,
with the existing ten-percent timing allowance. The original button-route
search budget stays fixed during that measurement. Route
requirements, required points, and bypass-versus-stomp success rules remain
intact. The tall-pipe family now uses 56–64 pixel heights: its former 65–68 pixel
range depended on a continuous running takeoff, whereas a spatial run ends at
its destination before the immediate jump. Strategy spawns leave one pixel
above their intended support before settling, preventing a lower floor from
embedding Mario inside a shallow ledge. A failed reference is reported as an
error rather than silently redrawing that layout. There are no stomp-recovery
families in either curriculum.

Vision decoding resolves contradictory contact outputs before the controller
receives them. A visible surface exactly beneath Mario's feet establishes
standing even if the contact classifier says airborne. Mario cannot be inside
solid support: when his found box reaches at most six rows below the top of a
surface he mostly covers, those rows are support pixels read as Mario (seen on
thin brick ledges), so his feet are placed on that surface and he stands.
Without this, a standing Mario read as airborne left the ground controller
waiting for a landing that never came. On 27,632 held-out frames it cut
standing-read-as-airborne errors from 111 to 62, with one new airborne frame
read as standing. Near-contact predictions
still need both the classifier and supporting geometry; side contact cannot
establish a landing. This prevents valid jump goals from turning into horizontal
air-control requests while Mario is actually standing on a ledge.
