# Design decisions

Decisions about the layered agent and how it is taught, newest first. Each
says what was decided, why, and whether it is in the code yet (with its
commit). How the parts work is described in the
[architecture and training guide](layered-agent.md).

## Standing rules

- **Policies see only what vision draws.** The learned layers read the
  vision model's objects, their own memory and their own earlier choices.
  Simulator state is used only to make teacher labels and to score episodes.
- **Strategy is a switch.** Whoever runs the agent sets it (speed run or max
  points) together with the side the goal is on. No network chooses it.
- **The tactic layer is an option-critic.** It holds a tactic over many
  actions and checks at every action boundary whether to end it.
- **The skill names an action; the executor reaches its predicted end.**
  The skill chooses a verb aimed at an object it sees; the action predictor
  gives where the action ends, and the executor, which has no learned weights
  and never refuses a destination, goes there.
- **Every teacher label is certified through the same executor** the agent
  uses. A run stops when the teacher cannot finish a layout; a failed gate is
  fixed in the teacher or the policy, never bypassed.

## 2026-10-11 — The teacher carries out the skill's actions until the predictor works (user decision; built)

The first skill run on actions (`artifacts/block_smb/skill_actions_20261011`)
learned the teacher's choices (89% agreement by round 6) while its validation
stayed at about 60% from round 1: its own actions were executed at the
action predictor's ends, and those were off by 10 to 17 pixels on average for
every moving verb (`run_to` 16.7, `land_on` 13.2, `approach` 11.9,
`jump_over` 11.0, `stomp` 10.1; the 8-pixel figure first reported averaged in
the holds). In `single_gap` the skill chose the teacher's own action and the
predictor sent Mario 28 pixels past where the teacher stops, into the pit.

User decision: do not use the prediction until it works. Until then the
skill's own actions are carried out by the teacher (training only; the
policies still see only vision): its own command when the action is its
choice, else the best safe version of that action it tries
(`teacher_actions.carry_out`: a jump of its jump table, a run it screens, a
run-up before a jump aimed at the action's object). A choice it has no safe
version of ends the episode as lost (`unsafe_choice`), so validation measures
the skill's choices alone. Training without `--predictor` does this; an agent
played as deployed still needs a predictor. The predictor is developed
separately and switched on only when the skill's validation through it
matches its validation with the teacher carrying out its actions. With the
round-6 skill, `single_gap`, `enemy_stomp` and `landing_enemy` (0% through
the predictor) won 4, 4 and 3 of 6; every loss was a choice the teacher would
not make.

## 2026-10-11 — Fixes from the teacher's re-validation (built)

Round 0 of the re-validation lost two of 656 layouts:

- **No room to brake.** In `action_stomp_up` the teacher jumped onto the
  raised platform 6 pixels from a standing enemy; landing speed slid Mario
  into it. Any landing within 48 pixels of a live enemy at the same height
  is now safe only if holding from there until Mario stops does not carry
  him into the enemy where it stands (`controller_teacher.stop_safe`;
  before, only plants had this check). A standing enemy, or one walking
  toward him, is held where it stands, so only his own slide is judged: its
  approach is the next decision's to pass (judging it too left the teacher
  no command in the air after a stomp in `enemy_hop`). One walking away
  keeps walking (held, it barred every landing behind a receding walker,
  and under speed run a stomp was taught in `landing_enemy`). The teacher now wins that layout, in four
  commands: the direct stomp needs the longest jump, and no destination
  aimed at the enemy makes the executor choose it (aiming at its top picks a
  shorter jump that comes down on the platform edge beside it). Left as a
  known limit of aiming.
- **A descent the executor could not make.** A `skill_gap_descent` layout
  put a 61-pixel gap and a 35-pixel drop in front of a 40-pixel ledge; the
  reference route landed with 3 pixels of Mario on the lower ledge's edge,
  and the teacher's jumps all fell to the floor below. The sampler now
  redraws a local route lesson whose reference route stands anywhere but its
  start and its route platforms, or comes down onto a lower route platform
  with Mario's middle off it (`monte_carlo._route_lesson_fault`). The
  teacher's search over run-ups and jumps that land on a raised route
  platform (`mount_destination`) now also covers a lower one across a gap,
  so a drop below the target no longer counts as progress toward it.

The second attempt's round 0 lost two more:

- **Edge landings with no margin.** A reference jump landing with Mario's
  middle exactly on the lower ledge's edge passed; the executor's own jump,
  a few pixels shorter, fell past it. The middle must now land at least 4
  pixels inside (`LANDING_INSET`). The check had also missed routes that
  jump on their first frame: it took the first platform Mario stood on as
  his start, which was the landing; the start is now read at reset.
- **A stomp seen a frame early.** Vision reports Mario's feet on an enemy
  when they are within 6 rows of its top (`STOMP_ROWS`), a frame or two
  before the simulator stomps it. A teacher trial replayed through vision
  ended at that report, found Mario still in the air and nothing stomped,
  and judged the stomp unsafe; under vision the teacher then preferred
  landing beside the enemy (`action_stomp_up_back`), where it had no next
  command. A trial now plays on, as the episode does, until the simulator
  lands Mario or makes the stomp (`SETTLE_FRAMES`, at most 4 frames).

The third attempt lost one layout in round 3 and one in round 4, and round
4's labels concentrated:

- **Judging the walker's approach as the slide's** (round 3, `enemy_hop`):
  after a stomp's bounce every landing was 12 pixels from a walker coming
  toward Mario, which the brake check counted against the landing, so the
  teacher had no command in the air. The check now holds the enemies where
  they stand (above).
- **A jump back off the ledge** (round 4, `skill_ledge_descent`): the jump
  back reached the route's floor first, under the ledge, where the walker
  then trapped Mario. A move away from the target's side goes against the
  tactic; it now ranks after every move toward it, stomps included (here the
  teacher stomps the walker from the ledge).
- **Run-ups of exactly 8 pixels** (round 4: 4 of its 6 runs back): where
  run-ups of several lengths keep the same, most room, the shortest was
  taken, the 8-pixel minimum wherever the room stopped growing after it.
  The middle of the unbroken range of equal run-ups is now taken, as for
  equal landing destinations (`_middle_of_equal_runs`).

The teacher wins 246 of 246 fresh layouts of the 41 skill families with these.

Its rounds 6 to 10 lost five more, three of which the current teacher still
lost:

- **A stomp whose bounce lands against a walker** (`enemy_hop`): the teacher
  stomped a standing enemy as a walker arrived; the bounce came down 6
  pixels from the walker whatever the steering in the air. A stomp is now
  safe only if its bounce, played out, comes down at least 8 pixels from
  every live enemy (`bounce_lands_clear`; the stomp's end, which the
  predictor learns, stays on the enemy's top).
- **A run-up judged at running speed** (`pit_leap`, which starts Mario
  running): the jump from here was moved along the floor behind a run to the
  pit's edge, but a run ends at rest, from where no jump cleared the
  64-pixel pit. While Mario moves faster than half a pixel a frame, only
  jumps from where he is are screened (`MOVING`).
- **Stomping a walker on a raised platform** (`action_stomp_up`, twice): one
  picture does not show how the walker moves, and no destination aimed at
  it made the jump that stomps it; the teacher landed beside it instead.
  Action families with an enemy now start with the 12 watched frames that
  `enemy_stomp` has; the teacher then stomps at once. All 108 sampled
  layouts of the six action families with enemies are won.
- **A 1-pixel wobble read as speed** (`action_stomp_up_back`, under vision
  only): the seen box's left edge read a pixel further out in two pictures,
  so its middle moved half a pixel and back; Mario's speed was estimated
  as drifting right while he moved left, and the flight, planning its rest
  again from that speed, held the jump 7 frames too long and overshot the
  enemy. Speed samples now skip any change of the seen box's width, not
  only changes of more than a pixel (`SpatialFeedback.observe`). This is the
  agent's own executor: it played the same way.

The final attempt's round 3 lost three:

- **The bounce judged with no buttons** (`enemy_hop`, twice): played out
  with no buttons, the bounce stomped the arriving walker too and looked
  clear; the next command, a run in the air, came down beside it. The
  bounce is now played out through the executor, as a run to the floor
  below (every run command in the air came down at the same place).
- **A walker pacing a 30-pixel raised platform** (`action_stomp_up`): only
  the longest jump stomped it, and no destination aimed at it makes the
  executor choose that jump (the executor's aim at a moving target on a
  raised platform). A layout of an action family with an enemy is now
  kept only if the teacher wins it (`monte_carlo._teacher_wins`, on the
  true scene); about 15 seconds per layout, made in parallel.

Its round 9 lost one: a `skill_enemy_bypass` layout whose room between a
standing enemy and a walker coming on had shrunk to 2 pixels by the landing
(the reference route won by touching the goal; the teacher had no command
from the first frame). The bypass families' layouts are now checked the same
way (`TEACHER_CHECKED`).

With these (commit 0634d0e) the teacher won every layout the skill run can
use: its validation and fresh-evaluation layouts (369 each) and the training
layouts of all 12 rounds (656 each), 8,610 of 8,610, through vision and the
production episode pipeline. The tactic families' layouts are still to be
validated before tactic training.

## 2026-10-11 — Single values are judged by whether the scene explains them (approved; built)

A share of one value within a family can be physics (a pit nearly as wide as
the longest jump is crossed by the longest jump; a run-up of the 8-pixel
minimum), not a shortcut. Approved: judge single values by the scene. Built:

- Teacher qualification fails only when one end of a verb (within 2 pixels)
  covers more than 40% of that verb's labels over every family together
  (`layered_train.label_collapse`): one end for a kind of action whatever the
  scene.
- Per family, a concentration is explained when the action predictor, which
  learns ends from scenes of every family, reproduces the family's held-out
  ends within 8 pixels and its predicted end moves with the action's object
  when the object is moved in the picture (at least half the shift)
  (`predictor_train.explained`, written after each predictor training).
- The skill no longer outputs x and y, so the old destination gate is gone;
  the learner's own actions are checked per family against the teacher's
  ends (more than 40% on one end and 20 points above the teacher's).

## 2026-10-11 — Floors are cut where enemies stand (approved; built)

"Land on" a wide floor did not say where on it: the teacher landed 8 pixels
away in one scene and 45 in a similar one, depending on the threats around.
The scene description (both the true scene and the vision's) now reports a
surface cut where a live enemy stands on it (or a plant rises through it):
each piece is the floor between two threats or a threat and an edge
(`smb_scene_labels.split_at_enemies`; pieces under 8 pixels are dropped).
Mario's own support is still judged on the uncut surfaces; the target
tracker reads an enemy's floor as the pieces on either side of it.

A cut end walks with its enemy, so it is no landmark: the scene also keeps
the uncut surfaces (`SceneObservation.uncut_surfaces`, read through
`fixed_surfaces`), and the controllers measure camera scroll, hold a spot
and check for a static floor against those. Holding against a cut floor
made Mario follow a Goomba walking toward him and leave his hold area.

## 2026-10-11 — The skill chooses only actions (built)

The skill's mode, x and y heads are removed: it outputs a verb and a pointer,
and its own actions always execute at the action predictor's end state
(docs/action-predictor-controller.md).

The predictor reads the scene encoder's tokens and the scene memory of the
policy it was trained on. Skill training used to update both, which would
have moved the predictor's inputs under it during the run. Skill training now
learns the skill transformer only, and refuses to start when the predictor's
checkpoint records a different scene encoder and memory than the starting
policy's.

## 2026-10-11 — Three families move to the skill stage (built)

A family belongs to the tactic stage when its teacher changes tactic within an
episode. `choice_advance`, `enemy_gap` and `enemy_patrol` were there because
their reference routes landed beyond the goal (or beyond the platform their
route had to stand on) and walked back, a retreat. Jumps that land past the
goal, or past a route platform, are no longer certified
(`local_traversal.safe_jump_holds`, `LocalObjective.stand`); since then
neither the reference route nor the teacher switched tactic in any of 36
layouts of each, so they are skill families (41 skill, 42 tactic).

## 2026-10-10 — Actions, an action-conditioned predictor, a target tracker and an adaptive controller (approved; built except the adaptation network)

The user proposed, and the plan in
[action-predictor-controller.md](action-predictor-controller.md) describes,
a redesign: the skill chooses an action (a verb aimed at an object it sees);
the scene memory, told that action, predicts its successful end state; the
predictive controller computes the buttons and durations to reach it and
re-plans when its target is updated. The controller does not model or track
the world: a small object-centred target tracker (a recurrent network shared
by all objects, run every frame) predicts where each target will be at a given
time, senses when the world departs from its prediction, and sends only the
updated target. The controller is to be adaptive: a further deep network will
update the parameters of its motion model of Mario from sensed changes in how
he moves (details to be worked out later).

Why: the skill now has to output exact pixel destinations, the source of the
single values, value lists and imprecise run lengths found on 2026-10-09 and
10; and the scene memory predicts the next action's end without being told
the action.

Status: approved 2026-10-10; built 2026-10-11 (the skill's actions, the
predictor, the tracker and the controller's re-planning; the adaptation
network is a proposal). The predictor and tracker are trained before the
skill learns its actions.

## 2026-10-10 — Full SMB training: checkpoints every 500 pixels (decided, for later)

When the agent is trained on Full SMB levels, it is optimized for one of the
two strategies: the most points, or the least time. A level being learned gets
a checkpoint roughly every 500 pixels, and the time taken (speed run) or the
points scored (max points) at each checkpoint are what training uses, rather
than only the level's end.

Status: a note for the Full SMB training stage; nothing is built yet.

## 2026-10-10 — No teacher picks from a sparse list (built; being validated)

Decided: no teacher picks a destination from a fixed sparse list. A list
that is used must cover every value in the range.

Built:

- **Jumps:** every destination within 128 pixels, each pixel in x at the
  height of every platform and enemy top, is evaluated. With the flight
  replayed open loop (next entry) a jump is set at takeoff by its steering
  and button hold, and every destination maps to one of a few dozen jumps by
  the executor's own rule (`smb_trajectory.best_holds`, checked exact against
  the executor on 1,440 destinations). Each jump is tried once; its label is
  the destination making it that is nearest where it lands.
- **Runs, takeoffs and waits:** screened at every pixel (measured jump paths
  moved along the floor, enemies moved by their patrol rule), the best
  certified by trials. No run shorter than 8 pixels is taught, and running or
  waiting first only when it gives at least 4 more pixels of room: a 1-pixel
  run was otherwise taught again and again while an enemy walked away.
- **Routes:** every jump hold from 1 to 32 frames (an older executor could
  press only 16) and every plant run-on length.

Replaces the old candidate list (4, 8, 16, 24, 32, 48, 72, 96 pixels and a
few platform points), the takeoff stops (2, 6, 12, 20, 32), the waits (4 to
32 by 8), the monster jumps (48 to 128 by 16) and the patrol teacher's lists.

## 2026-10-10 — Why the longest jump dominates some families, and the room rule (built; being measured)

A census of 24 layouts per skill family (`artifacts/block_smb/action_census_20261010`)
still found single values. Each has a cause:

- **The longest jump.** With a run-up allowed, the jump keeping the most
  room past a pit of width g (from both of its edges and from what follows)
  covers g plus the landing room R, up to the longest jump J (50 pixels on
  level ground): it is g + R when R is at most J - g, else the longest. Most
  rooms were wider than J - g, so most labels were the longest jump. The
  room is now drawn by `threat_variety.best_jump_room`: evenly up to J - g in
  85% of layouts (each then best landed by a different jump), anywhere up to
  its widest in the rest. Applied to `action_jump_gap` (and its backward
  version) and to the floating ledge of the upper gap families.
- **Run-ups of 1 to 3 pixels** (93% of `stomp_mount`'s runs): the teacher no
  longer plans a run-up shorter than 8 pixels (`MIN_RUN`, as its other runs).
- **Hops of 13 pixels toward an enemy** (49 to 85% of four enemy families'
  jumps): the button route proposed them, and per frame a short hop
  progresses faster than a short run, which brakes. A hop that lands back
  on Mario's platform passing and stomping nothing is now taught only when
  no other move advances.
- **A forced jump.** Under the floating ledge only one jump climbs the step
  of `skill_lower_step_climb`; 60% of layouts now leave 24 pixels of floor
  before the step clear of the ledge.

The census also measures, for the actions of the redesign, the share of the
most common object slot per verb (structural: a jump-gap lesson's landing is
always "the second surface") and of the most common outcome per verb (what
the action predictor would learn).

Second census (`action_census_20261010b`, with these changes): every family
won all 24 layouts; six collapses are gone (the backward gap jump, the lower
step climb and exit, the enemy-stomp hops, the two creeping run-ups). What
remains is mostly boundary values: run-ups now gather at the 8-pixel minimum
(`pipe_mount` backs up 8 in 79%; two enemy families approach 9 to 10), and
the longest jump is forced where the pit is nearly as wide as it (hard gaps
of 31 to 40 pixels) or where a following enemy walks away and widens the
room (`action_jump_gap` 71% at 50). A share of one value is then physics,
not a shortcut; the formal validation proposed below (labels explained by
the visible quantities) is the right test for it.

## 2026-10-10 — The executor's in-flight re-prediction is off for Block SMB (built)

Each frame of a flight used to re-predict Mario's path and re-choose the
steering and remaining hold: it corrected the motion estimate and re-aimed a
stomp at a walking enemy, or a landing at a moving platform, where it is
forecast to be. Off (`smb_trajectory.REPLAN_IN_FLIGHT`), a flight plays the
hold and steering chosen at takeoff and keeps steering toward the goal;
stomps and moving-platform landings are aimed once, at takeoff, at the
forecast position. Holding a spot on a moving platform is a separate
controller and unchanged. It was half the cost of every teacher trial; Full
SMB play may turn it back on. Exception (2026-10-10): when vision corrects
Mario's speed during a flight (a jump made before his speed was seen: one
picture cannot show it), the flight plans its rest again once
(`Flight.replan`).

## 2026-10-10 — No family teaches, and no policy learns, one value (built)

Teacher qualification (before learning and before every round) fails a skill
family whose labels collapse: one destination x (within 2 pixels) covering
more than 40% of a command mode's labels. After every round, the learner's
own held-out choices are checked the same way, family by family; a round
with a family whose most common value is above 40% and more than 20 points
above the teacher's share cannot pass the gate. The decision layers are
regularized: dropout 0.1, weight decay 0.05, and the two-pixel spread of the
x and y targets.

## 2026-10-10 — Formal validation of the families (proposed)

The collapse checks catch one dominant value but not a short list of values
unrelated to the scene. Proposed, as one audit gating training:

1. **A specification per family:** its parameters and ranges, the visible
   quantities that decide the answer, the expected label as a function of
   them (for a jump into a window between two threats: about its middle,
   limited by reach), and what must not decide it (hidden quantities).
2. **Teacher verification:** the label is the best of every destination
   (now built, above); near-identical visible scenes get near-identical
   labels; exact relations hold (a mirrored layout gives the mirrored
   label; moving the second threat 16 pixels farther moves the landing
   forward by at most 8, never back); label response curves are smooth
   except where the move changes.
3. **Data validation:** a constant predictor must be badly wrong (the label
   varies) while a simple model of the visible quantities is right to about
   2 to 3 pixels (the label is a learnable function of what the agent sees);
   parameters sampled to cover every range and combination evenly.
4. **Policy validation:** pixel error against the teacher on held-out
   layouts, and the policy's destination responding like the teacher's when
   one parameter is changed (a policy that memorized values does not).

Status: proposed; the collapse checks above are its first part.

## 2026-10-10 — Pit and enemy families: room on both sides, enough variety (built; being validated)

Decided:

1. Families that teach jumping over or past a pit or an enemy teach the most
   distance from it **before** the threat (where Mario takes off and waits)
   and **after** it (where he lands).
2. These families vary enough, with enough combinations of their parameters,
   that a policy cannot get by on single remembered values.
3. In Full SMB, Mario can end up right next to a pit or an enemy. Layouts must
   also start him close to one and teach him to jump over or past it.
4. Further obstacles at varied distances after the threat (another pit,
   enemy, wall or low ceiling) bound the landing, so the policy cannot simply
   learn the longest jump.
5. The policies are regularized.

Why: the 8 episodes lost in the last fresh evaluation included the skill
running into an enemy with the 72-pixel destination it had learned from
jump-over lessons, and running off a ledge with a destination that fitted the
usual first run of that family rather than the distance to the edge.

What the code does today (a survey of 24 training layouts per family, 8 per
difficulty, the teacher alone; `artifacts/block_smb/threat_margin_census_20261010`).
Room is the median, with the least in brackets, in pixels between Mario's body
and the pit edge he takes off from, the nearer edge of the platform he lands
on, or the closest enemy (before takeoff / from takeoff until 30 frames after
landing):

| Family | Where Mario starts (x) | Different jump labels | Most common label | Room before the pit | Room on landing | Closest enemy before / after |
|---|---|---|---|---|---|---|
| single_gap | always 20 | 8 in 24 jumps | (72, 0), 7 times | 2 (−5) | 0 (−9) | – |
| pit_leap | always 70, already running | 10 in 24 | (96, 0), 7 times | 20 (20) | 19 (13) | – |
| platform_hop | always 60 | 11 in 24 | (78, −22), 4 times | 20 (20) | 7 (6) | – |
| action_jump_gap | 4 values, 106–110 | 1 in 24 | (72, 0), every time | 1 (0) | 10 (−2) | – |
| action_jump_gap_back | 4 values, 136–140 | 1 in 24 | (−72, 0), every time | 1 (0) | 26 (14) | – |
| skill_upper_gap_entry | 8 values | 12 in 24 | (54, 0), 3 times | 2 (2) | 13 (3) | – |
| skill_upper_gap_exit | 19 values | 2 in 24 | (72, 0), 22 times | 2 (2) | 11 (4) | – |
| skill_gap_descent | 8 values | 12 in 24 | (64, 40), 3 times | 2 (2) | 13 (5) | – |
| skill_enemy_bypass | always 80 | 1 in 24 | (72, 0), every time | – | – | 25 (20) / 13 (0) |
| skill_enemy_bypass_back | always 166 | 2 in 24 | (−72, 0), 21 times | – | – | 25 (19) / 21 (10) |
| enemy_hop | always 20 | 2 in 29 | (72, 0), 19 times | – | – | 17 (10) / 15 (8) |
| landing_enemy | 7 values | 7 in 27 | (32, 0), 12 times | – | – | 22 (14) / 15 (9) |
| enemy_stomp | 8 values | 13 in 25 | (59, −10), 4 times | – | – | 25 (8) / stomped |
| stomp_mount | always 40 | 18 in 27 | (63, −10), 3 times | – | – | 46 (10) / stomped |
| action_stomp | always 60 | 9 in 24 | (32, 0), 4 times | – | – | 52 (34) / stomped |
| action_stomp_back | always 186 | 8 in 24 | (−56, −10), 3 times | – | – | 51 (33) / stomped |

The teacher won all 384 layouts.

Findings:

- **Before a pit the room is kept as small as possible on purpose.** The
  teacher's run before a gap ends with Mario's body supported plus 2 pixels
  (`spatial_teacher.gap_destination`); in action_jump_gap he takes off 0 to 1
  pixel from the edge. Only the landing side is widened (the 2026-10-09
  margin decision below), and physics often leaves little: single_gap lands a
  median 0 pixels past the far edge, platform_hop 7.
- **Many families never move Mario's start**, and none starts him right at
  the threat: the nearest start is 20 pixels before a pit (pit_leap,
  platform_hop) or 20 pixels from an enemy (skill_enemy_bypass).
- **Nothing after the threat limits the landing:** a flat floor or one wide
  platform follows, so the longest jump always works.
- **Single values are taught.** Every skill_enemy_bypass and action_jump_gap
  label is 72 pixels (−72 mirrored), because from nearly one start every
  working jump is the same jump and the label is the middle of the working
  range. 22 of 24 skill_upper_gap_exit jumps and 19 of 29 enemy_hop jumps are
  (72, 0) as well.
- **A jump over can touch the enemy:** in skill_enemy_bypass Mario's box came
  within 0 pixels of the walker after takeoff in at least one layout, though
  that layout was still won.
- **Regularization today is light:** no dropout in the transformers, the
  optimizer's default weight decay (0.01), and the two-pixel spread of the x
  and y targets (below).

Status: built (2026-10-10), being validated; skill training is on hold
until the teacher wins every layout it can use. The teacher now measures
every pit edge as a threat (before, during and after a jump), compares
commands by their closest threat first, labels a jump where it lands and
chooses the takeoff before a pit. The pit and enemy families start Mario at
varied distances (right next to the threat in about 40% of layouts) and
follow the threat with a second pit or enemy at a varied distance
(`threat_variety.py`); 24 of 38 skill families collapsed on one destination
before (survey above) and were given variety: varied starts, platform and
step sizes and places, route platform geometry, and bounded landings. The
regularization and collapse checks are in the entry above.

## 2026-10-10 — Validate the teaching on every layout before training (skill families pass; 11 tactic layouts lost)

Decided: before training starts, the teacher alone, playing through the
production episode pipeline (vision, executor and episode budget), must win
every layout the run can use, with a label at every decision.

Each family now draws its training difficulties from its own random sequence,
and a weak family's extra layouts follow its usual ones, so the layouts a
round can use (at most 16 per family: 8 usual and 8 extra) are known in
advance (5eedea7).

Results (`artifacts/block_smb/teacher_validation_20261010`):

- **Skill families:** 7,980 of 7,980 won. That is 342 validation, 342 fresh
  evaluation and 12 rounds × 608 training layouts.
- **Tactic families, validation layouts: 394 of 405 won.** The teacher lost
  11, in 8 families: full_smb_opening_proxy (2), tactics_bridge_sequence (2),
  max_points_advance (2), piranha_avoidance, tactics_obstacle_sequence,
  max_points_climb_backward, max_points_descend_forward and
  max_points_hold_ground. Nine ran out of time; in three (two of them also
  timeouts) a decision had no teacher label. Not yet examined; whether the
  2026-10-10 teacher changes caused them is not known. These must be fixed
  before any tactic training. The fresh-evaluation layouts of the tactic
  families are still being checked.
- The first attempt at the tactic families crashed while preparing a
  speed-run layout: the teacher's platform-landing helper built a jump to a
  platform more than 256 pixels away before checking its distance. It now
  checks first; nothing changes where the old code did not crash.

## 2026-10-10 — The strategy decides between stomping and jumping over (5eedea7)

Every lesson that must stomp an enemy is played under max points (kills
score); the jump-over lessons are played under speed run. Before, stomp
lessons and jump-over lessons gave the skill identical inputs.

The teacher follows the strategy where an enemy is within 128 pixels ahead:

- **Max points:** a stomp that works now.
- **Speed run:** a jump over rather than a stomp, and passing now rather than
  waiting, unless waiting gives the same jump at least 4 pixels more room.
  Where no jump over works now, it tries holding still or stepping 4 to 32
  pixels closer and takes the wait whose jump over keeps the most room.
  It stomps only when nothing else works.

## 2026-10-10 — The skill chooses mode, then x, then y (5eedea7)

The destination's x reads the chosen mode, and y reads the mode and x.
Chosen apart, the three could combine two commands, for example a run with a
jump-over's distance. The new parts start with no influence, so older
checkpoints keep their outputs.

## 2026-10-10 — The destination loss grows with the pixel error (5eedea7)

x and y are taught against a two-pixel bell curve around the label, plus the
expected distance from the label in units of 32 pixels. Before, every wrong
pixel cost the same, so a destination 57 pixels off cost no more than one 1
pixel off.

## 2026-10-10 — No hidden patrol ends in landing_enemy (5eedea7)

Its walker patrols the whole visible floor, as enemy_stomp's already did.
With short hidden patrol ends it turned around while Mario was in the air,
so 1-pixel differences of distance decided between waiting, jumping over and
stomping.

## 2026-10-09 — Teach the working command with the most room (8323470)

Among commands that make the same move, the teacher labels the one that keeps
Mario farthest from enemies he does not stomp, and, where he ends standing,
farthest from the edges of that platform. Equal room (within 2 pixels) is
broken by the middle of the range of destinations that work.

## 2026-10-09 — Enemy forecast and watched frames (02685b8)

The scene memory forecasts where each enemy will be when the next action
ends, and the skill reads it. Enemy lessons start with 12 frames the agent
watches before its first decision, so it has seen the enemy move.
