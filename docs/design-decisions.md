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
- **The skill names a destination; the executor reaches it.** The executor
  has no learned weights and never refuses a destination.
- **Every teacher label is certified through the same executor** the agent
  uses. A run stops when the teacher cannot finish a layout; a failed gate is
  fixed in the teacher or the policy, never bypassed.

## 2026-10-10 — Actions, an action-conditioned predictor, a target tracker and an adaptive controller (approved; being built)

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

Status: approved 2026-10-10; being built. The current teacher and family work
(below) was checked in first as a checkpoint, with tests still failing (listed
in its commit); skill training on the current design is not started.

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

## 2026-10-10 — The executor's in-flight re-prediction is off for Block SMB (built)

Each frame of a flight used to re-predict Mario's path and re-choose the
steering and remaining hold: it corrected the motion estimate and re-aimed a
stomp at a walking enemy, or a landing at a moving platform, where it is
forecast to be. Off (`smb_trajectory.REPLAN_IN_FLIGHT`), a flight plays the
hold and steering chosen at takeoff and keeps steering toward the goal;
stomps and moving-platform landings are aimed once, at takeoff, at the
forecast position. Holding a spot on a moving platform is a separate
controller and unchanged. It was half the cost of every teacher trial; Full
SMB play may turn it back on.

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
