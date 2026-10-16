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
- **The skill names an action; the executor reaches its end.** The skill
  chooses a verb aimed at an object it sees; the action predictor gives where
  the action ends, and the executor, which has no learned weights and never
  refuses a destination, goes there. Until the predictor predicts well, the
  teacher carries out the skill's actions in training instead (2026-10-11).
- **Every teacher label is certified through the same executor** the agent
  uses. A run stops when the teacher cannot finish a layout; a failed gate is
  fixed in the teacher or the policy, never bypassed.

## 2026-10-16 — Phase 2: the executor's kinematics, collision trackers and the world model (user decision; building)

Approved plan (`/home/amoran/.claude/plans/resilient-wobbling-abelson.md`):

- **The executor keeps a first-order state estimate** (position and velocity,
  constant velocity) of every tracked sprite and of Mario. User: "vision
  should not be doing any state estimation since the executor is already
  doing that. send the state estimates to the LSTM". The collision trackers
  read the same estimates. Built: `smb_trajectory.Track` (16-sighting mean,
  identity kept 8 frames unseen), `core/smb_kinematics.py`.
- **Truth is recorded** with collection episodes (`record_truth`) and labels
  are made afterwards from recorded real paths (`core/collision_labels.py`,
  `core/action_outcomes.py`), the same way Full SMB's memory will fill them.
- **Collision trackers** (`core/target_tracker.py`, rewritten): per enemy,
  every frame, a stomp or a hit in two-frame bins over 96 frames, or none,
  and the contact place; inputs are the kinematics, what vision shows of the
  enemy and the action in progress (an enemy-aimed action's destination is
  the enemy's seen point, never the teacher's command). The old
  displacement tracker and the action controller are deleted.
- **The world model** (`core/world_model.py`, the policy's `memory`): one
  LSTM step per action; trained on executed and tried actions
  (`world_model_train.py`). The action predictor and its trainer are
  deleted. Checkpoints load at memory version 5 with it untrained.

- **Deployed execution** (built): the world model's outcome is the
  destination; `core/smb_threat.py` ends waits and ground moves when the
  trackers expect a hit within the executor's jump lead (+5 frames; an enemy
  not yet seen closing is given 0.5 px a frame); stomps aim at the trackers'
  contact point. `exam-layer --execution` measures each network apart.

Collection and training of both: after the tactic re-check's validation and
fresh stages (user, 2026-10-17; chained). Then the Full SMB loop.

- **Full SMB's truth from the emulator's memory** (built; user: "go ahead
  with the Full SMB foundation"): the boxes the game itself tests contacts
  with (Mario's and each enemy slot's), each slot's drawn box, alive or
  defeated, stompable; the level's progress (Mario's distance into it, score,
  coins, time left, dying, left the level) (`stages/full_smb/truth.py`). The
  game reuses its six enemy slots as the level scrolls; a slot filled again is
  a new enemy column. The arrays have Block SMB's recording format, so the
  same labels and the same trainers read them. `stages/full_smb/practice.py`
  records a playthrough with the deployed agent (no teacher). How the
  playthroughs feed learning (how many per update, replay, the rewards of the
  two modes, the skill's and tactic's reward rounds) is to be settled with the
  user once the networks' first results exist.

## 2026-10-16 — Stomp variety; the plant and monster teachers win their lessons (user decision; built)

skill_waits_20261015 stopped at round 11's teacher qualification: 29 of the
round's 72 stomp labels came down 45-47 pixels ahead (40.3%, against the 40%
limit), 21 of them in stomp_mount, whose walker always started beyond a
standing jump's reach (52-76 pixels): the teacher met it with the longest
standing jump every time. User: add variety, do not restart the skill run.
stomp_mount's walker now starts within a standing jump's reach in most
layouts (REACHABLE, 75%: right next to Mario up to about 44 pixels; a moving
walker no closer than it walks while watched), beyond it in the rest. 18
drawn layouts: all won, stomps 9-47 pixels ahead, 4 of 18 at 45-47.

Next is the tactic layer (user: tactics, and build the world model and the
trackers meanwhile). Two tactic families were lost by the teacher:

- **piranha_avoidance** (timeouts). The teacher's runs were ranked by
  progress and screened only against enemies and pits, so the six tried were
  aimed into or behind the plant's pipe; each stopped at it, and none to its
  foot was tried. Runs along the floor now stop before a solid in their way
  (`_screened_runs`). The plant teacher then crosses with a run-up into a
  jump onto the pipe or past it (the takeoff search pits use), certified by
  trials, waits 32 pixels before the pipe while none works, and searches
  only while the plant will be down within 64 frames. The teacher's enemy
  paths now include a plant's rise and fall by its cycle (`_enemy_paths`).
  Layouts 100 and 101: won in 5 and 8 decisions.
- **monster_retreat** (no command). The monster teacher refused every jump
  over the monster: it compared the jump's landing floor with the floor
  object found before the trials, and every trial restores the simulator
  with new platform objects. Compared by place in the list now; Mario backs
  out of the tunnel, waits, and jumps the monster in the open. Layouts 100
  and 101: won.

## 2026-10-16 — The skill's held-out choices are judged against the teacher's at the same moments (user decision; built)

skill_waits_20261015 round 0-1 flagged "enemy_hop approach (20, 0) 71%
(teacher 17%)" as a collapse. A replay of the 9 held-out layouts showed 33 of
the skill's 34 choices were the teacher's own (verb and object), and every one
of the 7 run-ups was the teacher's command (carry_out: 21, 24, 20, 22, 18, 72,
20 pixels). The check compared where actions ended on held-out layouts with
the teacher's labels on training layouts: since the skill picks only the
action, it measured the teacher. The one real mistake: hard layout 8, Mario
standing 36 px before a standing walker, the skill chose "jump over" where
the teacher ran up first; no standing jump gets over it (each comes down
before it or on it), so the choice had no safe version.

User: "make the check". Each skill decision now records the teacher's choice
at that moment, labelled or not (`EpisodeRecord.teacher_verb`,
`same_as_teacher`). Each round reports per family the share matched and the
differences (`choices_against_teacher`); the collapse gate compares the
skill's verb shares with the teacher's at the same decisions.

## 2026-10-15 — Predictions come after the choice; ground truth stands in until they are trained (user decision; building)

The policy's LSTM used to step every 4 frames and before each decision, and the
skill and tactic layers read its forecasts: the scene expected at the action's
end, each enemy's place then, and its state, fed into the tactic memory. The
forecasts were useless for timing. Ending a wait on them won 1 of 27
choice_hold_area layouts; aiming stomps with them landed 18 of 51 moving
walkers on the first jump.

User decisions:

- **The skill reads no prediction.** It chooses from vision, the tactic, the
  strategy, its own last choices and the executor's feedback. "The prediction
  reads the skill chosen."
- **The world model** is the LSTM run once at the start of an action, after
  the skill has chosen it, with that action as input. It predicts the action's
  outcome: where Mario and its object end, how many frames it takes (for a
  wait, how long), and the chance it succeeds. The action predictor folds into
  it.
- **Target trackers** run every frame. Each predicts when and where its enemy
  and Mario will collide, as a stomp or a hit.
- **The executor's per-enemy estimate** ends a wait or walk sooner the faster
  an enemy comes at Mario; the skill then picks again and learns from the
  teacher to jump. A stomp aims at the predicted collision point. No Kalman
  filter for now.
- **Neither network is used while the skills and tactics train.** Every
  "prediction" is the simulator's ground truth until they are trained, from
  recorded truth.
- **Evaluation is Full SMB played many times,** learning continuing on every
  playthrough, the two networks included. Speed mode is rewarded by time to
  each checkpoint, points mode by points. The measure is how few playthroughs
  it takes before Mario plays the level successfully.

Built so far:

- The skill's own holds carried out with the teacher's wait length (659e6d0).
- The LSTM out of skill and tactic play and training. The skill's forecast
  input, the expected-scene inputs and the tactic memory's action-memory input
  are gone; old checkpoints migrate (`memory_version` 4); the predictor is
  refused in training.
- Collision truth (`stages/block_smb/collision_truth.py`).
- Every wait is one action of true length. The teacher's one-frame holds are
  measured when played (`teacher_tokens.whole_wait`). A wait ended by a coming
  walker is marked (`SkillToken.threat`), and the executor ends it as `threat`.
  The skill's execution feedback gains a ninth number, `collision_due`. Older
  checkpoints read it with zero weight.

In training a walk is the teacher's certified-safe version, so no collision
ever cuts one short: forced ends of walks come with the trackers. The world
model and trackers follow
(`/home/amoran/.claude/plans/resilient-wobbling-abelson.md`, Phase 2).

## 2026-10-15 — Holding: a trigger buffer around Mario for each walker (user decision; built)

In `choice_hold_area` the hold area was a fixed place 48 to 64 pixels either
side of the start, leaving it lost the lesson, and the teacher jumped a walker
when the jump kept the most room (after 14 to 67 holds). The policy timed these
jumps badly (round 0 of skill_practice_20261013: 4 of 9). The user's rule:

- **The box is a buffer around Mario, not a place.** When a walker enters
  it, Mario jumps over the walker and lands behind it.
- **Its size depends on the walker's speed:** just wide enough for the jump.
  Each walker has its own (walkers now draw their own speeds).
- **Where the walker will be when Mario lands** is the simulator's truth in
  training; in deployment the predictor will have to forecast it.

- **Pits and obstacles near Mario** (user): on each side, 32 to 80 pixels
  away, his floor ends at a pit, or a pipe stands, or the floor goes on;
  sometimes a brick row hangs low over him. A jump over a walker must not
  land in the pit, hit the pipe or go up into the bricks
  (`tactic_families._hold_layout`). Walkers turn at the pit's lip or the
  pipe, so they come back.
- **A wait is one action, its length predicted** (user: "the wait action
  should be using the prediction, just like stomp"). A hold carries a number
  of frames (`SkillToken.frames`): until the walker is close enough that the
  jump is due. In training that is the simulator's truth; deployed it is the
  predictor's. The executor plays such a hold to its end (`ActionPlan.wait`)
  instead of deciding again every frame, so an episode has 2 to 5 decisions,
  not about 130. A planned wait before a pit crossing is one hold as well.

Built: the schedule's `dodge` hold (no `area`; nothing about where Mario holds
is a loss; the lesson completes anywhere on his floor).
`controller_teacher._dodge_in_area` jumps when the jump over the walker,
started `TRIGGER_SLACK` frames later (the next decision, then the 4-frame late
start every taught move tolerates), would no longer pass it; near the walker
trials decide, since the predicted paths are a frame or two hopeful and do not
see bricks overhead. Jumps whose predicted path goes through anything solid
are not considered. `_dodge_wait` finds the hold's length by holding on in a
probe until that rule jumps. Over 27 layouts the teacher won all, jumping when
the walker was 2 to 6 pixels away (more for faster walkers). About half the
medium and hard draws have no safe dodge (two walkers between a pit and a
pipe) and are redrawn. The family pictures draw each jump's trigger box
instead of a goal box, and each hold's length.

## 2026-10-13 — Fixes from the review of the family pictures (user decisions; built)

The user reviewed the picture of every layout and asked for these:

- **A run-up keeps its speed into the jump** ("approach keeps its speed into
  the jump"). Mario walked to a gap's edge, stopped, and jumped from rest,
  barely clearing it. An approach is now a run that does not brake: it
  ends at its destination still running (`SkillToken.carry`,
  `GroundMove._carry_press`), and the jump after it is planned from where it
  ends, at that speed (`controller_teacher._takeoff_search`, `_carry_run`).
  Only such a run is named "approach". A step that stops before a jump
  from rest (onto a pipe too tall to clear running from close up) is a plain
  run, searched last.
- **Jumps no higher than needed.** Among moves with enough room (16 pixels
  from every threat, 6 from a landing edge: `ENOUGH_ROOM`, `ENOUGH_LANDING`;
  more room counts for nothing), the lowest jump is taught (`_rank`). A hop
  that clears nothing is never taught (`_mere_hop`).
- **Action lessons are one action**, in every layout. The sampler drops a
  layout the teacher cannot finish in its one jump (`monte_carlo._one_action_fault`).
  `action_climb` and `action_climb_back` had walks before the jump.
- **No lesson starts in the air**: `landing_enemy` and the others start Mario
  on the floor.
- **`choice_hold_area`**: a larger area and walkers that pass through it.
  Mario stays in the area and jumps over each one (`_dodge_in_area`); the
  goal box covers the area. In the air above the area is not leaving it.
  (Replaced on 2026-10-15 by a trigger buffer around Mario, above.)
- **A taught move tolerates a late start** (`_tolerant`): it must still work
  started 2 or 4 frames later; after 3 that do not, the first is taught.
- **Starts and spawns:** `pipe_mount` starts 20 to 40 pixels from the pipe;
  overhead-climb lessons start clear of the ledge above.

Last checks before re-making the pictures found four more, now fixed:

- **Vision ended a jump on the way up.** Rising past the top corner of a pipe
  as tall as his jump, Mario's box touched it one row below his feet, and
  vision called that contact. A landing now needs his feet not to be rising,
  except on a moving platform (`LandingWatch`). This made `pipe_mount` hard
  walk to the pipe, walk back and then jump; it now takes a 14-pixel step
  and jumps.
- **Vision saw a landing on an enemy stomped earlier.** It stays drawn
  flattened. After the bounce, falling back past it ended the next action in
  the air. Contact only with enemies already defeated on the frame before is
  no landing (`LandingWatch`, `_only_old_defeated`).
- **A stomp completed in the next action was named a second stomp.** Vision
  ends a stomp when Mario's feet touch the enemy's top; the simulator kills
  it a frame later, once the boxes overlap. That kill no longer names the
  next action (`teacher_actions._classify`).
- **No progress past the goal.** A jump into the goal that failed when
  started late gave way to a longer one past the goal, against the world's
  end, where the walker trapped Mario. A move that ends past the goal box is
  no progress (`_overshoots_goal`), in plans as well as single moves.

The test suite found two more, also fixed:

- **Run-ups into jumps no command makes.** After a run-up, jumps landing
  more than REACH (128) pixels away were screened, and no command the
  teacher gives makes them; they used up the tries, and the teacher stomped
  an enemy it should pass. They are no longer screened (`_takeoff_search`).
- **Runs aimed under a raised block.** The run search took the floor under
  the block Mario stood on for a lower floor he could walk onto; every such
  run stopped at once. With hops no longer taught, the teacher had no move.
  A lower floor is now reached off the ends of his own only (`_screened_runs`).

The re-check of every layout found one more: on a ledge 33 pixels wide,
only a 3-pixel step to its end took Mario onto the platform above, and steps
were at least 8 pixels (against creeping, which a step whose jump is
remembered does not do). A step before a jump can now be any length, but
one shorter than 8 pixels only when no jump from where Mario stands works
(else a few pixels more room put tiny steps before jumps that worked). And an
episode where the teacher has no command now ends there
(`no_teacher_command`): it held and asked again every frame until the stall
limit, an hour and a half for one layout, which failed the check anyway.

The re-check's training rounds then lost two layouts in four rounds:

- **Stranded at a wide pit** (`single_gap`, a 56-pixel pit with a walker
  pacing the far floor). With no takeoff working at the start, the teacher
  walked Mario to the edge, from where no jump crosses, and held there until
  the episode stalled. When nothing crosses now, the teacher now waits where
  Mario stands for the shortest wait after which a run-up and jump across is
  screened to keep the most room (`_pit_wait`); the holds until it ends are
  taught without searching again (`_planned_wait`), and the run-up is then
  certified as usual. Mario waits 48 frames, runs up keeping his speed and
  jumps the pit and the walker.
- **A stomp only one jump height makes** (`action_stomp_up_back`, a walker
  standing on a small raised platform, which vision places 4 rows lower than
  the truth). Played in a batch of eight episodes, the teacher's jump missed
  and it walked under the platform. A one-jump lesson only one jump height
  wins is now redrawn (`one_height`; 2 of 176 one-jump lessons a round).
- **No holding when nothing moves.** After that walk the teacher held, with
  only a standing walker in the scene, until the stall limit. Holding is now
  taught only while something moves.

From the user's review of the pictures after that:

- **`enemy_patrol` is taught by the main teacher.** It alone had its own
  teacher (`patrol_teacher.py`, now removed), which took the most progress
  per frame: a hop along the floor before the first walker, clearing
  nothing. It now gets no route proposal and the teacher weighs every move,
  as for every family: a run-up into a stomp or a jump over each walker.
- **A route waypoint is not taught before a move that wins.** The route's
  proposal was taught whenever it progressed: `skill_lower_descent` walked
  16 pixels to a waypoint, then ran down to the goal. A proposal that only
  progresses is now taught when no move from here wins.
- **Wait for space, then a running jump over the pit** (user, reviewing a
  `single_gap` layout whose walker paced the far floor up to the pit's edge;
  the teacher walked to the edge and jumped from rest onto the lip, 5 pixels
  from the walker, with no name for the landing). A takeoff over a pit is
  taught at once only if it keeps `ENOUGH_ROOM` (16 pixels) from every
  enemy; otherwise, while an enemy moves, Mario waits where he stands, with
  room to run up, for the shortest wait after which a running jump across
  is screened to keep it (`_pit_wait`), then runs up keeping his speed and
  jumps; the walker is cleared by the same jump when it is near, or by a
  jump of its own after landing in front of it. Unless the lesson is played
  for points, a crossing stomps nothing (the screen had taken jumps onto the
  walker for clear ones). Over 32 `single_gap` and `enemy_gap` layouts of a
  round: 26 jump the gap and then the walker, 5 clear both at once, 3 wait
  first; all won. At most 4 waits are planned an episode
  (`MAX_PLANNED_WAITS`): the screen can keep promising a run-up the
  certified search refuses.
- **The pictures' decision lists** give a hold decided again and again as
  one line ("hold ×71"), named "hold" alone (it is aimed at nothing).

From the full re-check that followed (5 of about 10,000 layouts lost):

- **The teacher is omniscient** (user: "the teacher teaches and the LSTM
  learns, not the other way around"). A jump aimed at a walker was planned
  where the learner's memory forecast it; an older checkpoint's forecast of
  a standing walker made three stomp lessons miss. In training the
  executor, in the teacher's trials and in the episode itself, now aims at
  where the simulator will have moved the walker when the jump lands
  (`VisualTracks.exact`, `exact_positions`; `smb_trajectory.aimed_goal`
  plans the jump again until its landing time settles). As deployed, the
  agent aims by its own memory, as before. The re-check now plays with the
  policy training starts from, not an older one.
- **One-jump lessons are also replayed through vision** before they are
  used: a climb onto a step 56 pixels tall from 5 pixels away was won with
  labels, but under vision's steering every jump a destination names
  clipped the step's corner.
- **A run off a ledge aims 4 pixels past its end** (`DROP_MARGIN`): aimed
  just clear of it, a run stopped 3 pixels short through vision and Mario
  stood on the ledge's corner.
- **Of the moves that win, the one ending nearest the goal box's middle is
  taught** (user), and a one-jump lesson whose jumps onto a higher surface
  all land within 4 pixels of its edge is redrawn (`edge_landing`): a climb
  onto a wide step from too close landed 1 pixel inside it. Jumps across a
  gap or down are not checked: the widest gap a standing jump crosses lands
  just past its far lip, and with every landing checked the sampler gave up
  on hard action_jump_gap lessons.

Everything the skill learned from needs teaching again: the actions, their
names and where they end changed in most families.

## 2026-10-12 — After a stomp the teacher teaches the next action; no decisions in the air (user decision; built)

A stomp ends when Mario's falling feet meet the enemy; leaving the enemy is a
new action, and what it is (verb and object) is the teacher's to teach (user,
correcting the description of it as a bounce). The re-check of every layout
found three choices after a stomp with no name: the teacher had taught a
command aimed at the goal, a point in the air, which names no action. It
also judged a stomp safe by one fixed follow-up (falling straight down)
rather than by an action it would teach.

- **Only named actions are taught** (`controller_teacher._nameable`): the
  proposal, the ranked moves and the margin versions must each name an
  action (`teacher_actions.teacher_action`).
- **A stomp is safe only if a next action is** (`next_action_safe`, replacing
  `bounce_lands_clear`): from the moment of contact, a steer the teacher can
  teach (straight down first, every 4 pixels within 32) wins or lands at
  least 8 pixels from every live enemy with room to brake.
- **Names:** an action that reaches the goal before it ends is named by its
  destination; a point where the floor is cut at an enemy is on the nearest
  piece of that floor.

Pictures of every skill family (`stages/block_smb/family_atlas.py`, user
request: each drawn parameter at its lowest, middle and highest value, enemy
and moving-platform motion blurred, a heat spot at each decision) then
showed decisions in the air of two kinds, both fixed:

- **The first decision of every `landing_enemy` layout** came while Mario
  was still dropping onto the floor. The watched frames before an episode
  now continue until he stands (`env.LANDING_WATCH`, up to 64 more).
- **Walking off a low step** (`action_descend_back`), vision's contact went
  off and then flickered on for a frame at the height he left (his box at
  the step's corner): the agent took it for a landing, ended the run there
  and decided again in the air. A landing now needs his feet to have moved
  up or down at least 2 rows while he was in the air
  (`smb_agent.LandingWatch`, `LANDING_MOVE`); the deployed agent ends
  actions by the same check. Likewise a run ends on arrival only with him
  standing (`SpatialFeedback.arrived`): one that takes him off a ledge is
  ended by his landing.

## 2026-10-12 — One rule names a run-up; every teacher choice has a name (user decision; built)

The run with run-ups named by what follows
(`artifacts/block_smb/skill_approach_20261012`, stopped after round 1 at 93.8%)
still lost 16 of 36 validation layouts in `landing_enemy`, `enemy_hop`,
`enemy_patrol` and `single_gap`. A replay of them, choice by choice against
the teacher's: none died; every one ended at a choice the teacher could not
carry out, and the skill agreed with the teacher at 86 of 117 decisions.
Losses by cause:

- **A run-up named three ways (9).** Labels named it after the floor the
  jump lands on, often Mario's own floor even when the jump clears an enemy;
  the jump itself is named after the enemy. The teacher read "approach my
  own floor" as a run toward that floor's middle, often backwards, so it had
  no version. Now one rule names a run-up, in labels, in the plan the
  teacher remembers between decisions and when carrying out the skill's
  choice (`teacher_actions.jump_aim`, `run_up_slot`): what its jump is aimed
  at, in the order a jump is named (the enemy it stomps, else the enemy it
  clears, else the surface it lands on). A run before a jump back onto the
  surface the run started on is a plain run; approaching that surface has no
  version.
- **Choices impossible from Mario's state (5):** running across a gap at
  full speed, jumping while still in the air, jumping where the teacher
  runs, and aiming at the far floor instead of the near one. These are the
  skill's to learn.
- **The teacher's own choice had no name (2).** A landing with Mario's
  middle 3 pixels past a surface's edge matched no surface (now within half
  his width, `STANDING_REACH`), and a steer in the air that came down on a
  second enemy was named a run to a point in the air (now a stomp; in the
  air, every verb is carried out by a steer).

Enemy timing was a smaller part than expected: the enemy mattered in a few of
the 16 losses, and then as "run up first" against "jump now".

## 2026-10-12 — Run-ups named by what follows; enemies watched in every skill lesson (user decision; built)

The skill trained with the teacher carrying out its actions
(`artifacts/block_smb/skill_teacher_execution_20261011`: best validation
90.5%, 88.9% of 369 fresh layouts) failed in two ways:

- **Approach or run?** A run along the floor was labelled an approach only
  when the teacher had planned the jump before running; otherwise the same
  run was "run to my floor" and the jump was planned at the next decision.
  That plan is not in the scene, so similar scenes got different verbs, and
  a mix-up was often a choice with no safe version (`single_gap`,
  `action_jump_gap`). Now a teacher run its next command follows with a jump
  the same way is labelled an approach to what that jump stomps or lands on,
  named in the run's own scene (`layered_train._Lane._name_run_ups`,
  `teacher_actions.approach_slot`, using the jump's trial made before it
  starts). Carrying out the skill's "approach X", the teacher's own run
  toward X that stops before it counts (`teacher_actions._runs_up_to`).
- **Enemy timing** (`enemy_hop`, `landing_enemy`, `stomp_mount`,
  `enemy_patrol`): the skill chose stomps, landings and runs at moments with
  no safe version. Every layout of a skill-curriculum family with an enemy
  now starts with the 12 watched frames (one picture does not show how an
  enemy moves; before, only `enemy_stomp`, the bypasses and the action
  families had them), and these four families get 16 extra layouts every
  round (`LayeredTrainConfig.extra_layouts`).

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
  and under speed run a stomp was taught in `landing_enemy`). The teacher
  now wins that layout, in four commands: the direct stomp needs the longest
  jump, and no destination aimed at the enemy makes the executor choose it
  (aiming at its top picks a shorter jump that comes down on the platform
  edge beside it). Left as a known limit of aiming.
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
