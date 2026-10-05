# The layered SMB agent

One agent plays both Super Mario Bros games: Block SMB (a simulator drawn to
look like the NES game) and Full SMB (the real game in an emulator). It sees
each game only through that game's vision transformer, and it learns
everything first in Block SMB.

## What the agent sees

Each game has its own trained copy of one vision transformer
(`retroagi/core/vision.py`, `SceneVisionTransformer`). From a 256x240 screen it
outputs:

- every pixel's type: background, Mario, ground, brick, question block, pipe,
  coin, enemy, moving platform, power-up;
- for each 8x8 cell, the kind of enemy drawn there: walker, plant, other,
  defeated;
- Mario's facing; whether he is standing, in the air or on a moving
  platform; and whether his feet are on something (the land detector). Each
  game's land detector learns from its own engine's truth: the standing flag,
  plus stomps. In Full SMB the stomps are read from the emulator's memory; in
  Block SMB they come from the simulator's stomp handling.

The pixel types never reach a policy. `retroagi/core/smb_scene_labels.py` turns
them into one description of the screen, a `SceneObservation`:

- **Objects:** Mario and every enemy, coin, power-up and moving platform.
  Each is a group of touching pixels of its type; pieces at most one pixel
  apart join, and groups under 8 pixels are dropped as specks.
- **Blocks and pipes:** groups of their pixels.
- **Surfaces:** the top edge of everything Mario can stand on.
- **Gaps:** stretches of the floor with nothing to stand on.

Everything is reported as drawn. Nothing hidden is reported, such as a plant
inside its pipe or the game's collision boxes, and there is no speed: one
picture, one description.

The truth the vision transformers learn from uses the same rules.
- **Block SMB:** the simulator draws exact labels.
- **Full SMB:** the frame is rebuilt from game memory
  (`retroagi/stages/full_smb/pixel_labels.py`).

`scripts/vision/evaluate_block_vision.py` and
`scripts/vision/evaluate_full_vision.py` measure both models the same way.

`retroagi/core/smb_observer.py` packs the description into the policy's three
inputs:

- **C (352 numbers):** Mario, then fixed-length lists, each item "present"
  plus its box relative to Mario. The lists are enemies, coins, power-ups,
  moving platforms, pipes, blocks, surfaces and gaps.
- **A and B (8 and 16 numbers):** for each vertical band of the screen, the
  most important thing drawn in it.

## How it decides

`retroagi/core/layered_policy.py` holds the policy: two learned layers
(tactic and action), a shared scene encoder and two memories. Above them sits
the strategy switch.
- **Scene encoder:** one token per reported object, plus the band codes.
  Every position number also enters as 8 sine and 8 cosine waves, with
  wavelengths from 512 pixels (480 vertically) down to 4. A difference of one
  pixel is therefore a clear difference of input, while the long waves still
  tell near from far.
- **Memory**, a long short-term memory network, steps once per action, at
  the action's start, before anything decides:
  - **Input:** vision information only, never the action Mario is about to
    take: the scene encoder's summary of the latest picture, the frame the
    action starts on. What happened at earlier actions, including how things
    moved, it carries in its own state.
  - **Output:** its expected scene, a prediction of the scene numbers the
    vision transformer will report when the coming action ends, completed
    or interrupted.
  - **Who reads it:** the expected scene is encoded like a real one and given
    to the decision layers beside the current scene.
  - **Training:** action by action. It's scored against the scene actually
    reported when each action ended.
- **Strategy switch:** not a learned layer. Whoever runs the agent sets what
  the run is for and which side the goal is on (the agent can't see the
  goal):
  - speed run: finish in the least time;
  - max points: finish with the most points, a point for every coin collected
    and every enemy killed.

  A Block SMB layout sets it (its family's strategy, otherwise speed run, and
  the side its goal is on). In Full SMB it is set for the run, and the goal is
  always to the right. A strategy is an objective, a definition of what is
  rewarded, so there is nothing for a layer to learn in choosing one.
- **Tactic layer:** chooses one of six tactics, the way Mario is going, and
  holds it over many actions. Forward is right, the level's way; backward is
  left. Going left is never advancing:
  - advance: go right. Walking, jumping gaps, getting past or stomping
    enemies, and waiting for the way to clear (a moving platform to come, a
    plant to go back into its pipe, a monster to come close enough to jump)
    are all part of it; the action layer sees from the scene which is
    needed;
  - retreat: go back to the left: to a goal on the left, to coins or an
    enemy to stomp behind Mario, to the goal after a jump carried him past
    it, or away from danger (backing out of a dead end, keeping away from a
    monster);
  - climb forward / climb backward: go up onto something higher, to the right
    or to the left;
  - descend forward / descend backward: go down to something lower, to the
    right or to the left.

  It is an option-critic: a tactic is an "option", a behaviour that lasts
  many actions. Its transformer reads the current scene, the action memory's
  expected scene, the strategy switch, the tactic it holds and how long it has
  held it (actions and frames), its own memory's state, and that memory's
  expected scene when the tactic ends. At every action's start it gives:
  - **the end check:** the chance that the held tactic is finished;
  - **the critic:** for every tactic, the reward to expect from holding it
    from here;
  - **the choice:** which tactic it would choose now.

  If the end check says finished (in play, a chance over one half; while
  exploring, a draw), the tactic memory steps and the layer chooses the next
  tactic; otherwise it keeps the tactic it holds. It may choose the same one
  again, which starts it afresh.
- **Tactic memory**, the tactic layer's own long short-term memory network,
  steps once per tactic, when one starts (at the episode's start, and when
  the held tactic ends):
  - **Input:** the latest picture's summary, the tactic that just ended and
    how long it lasted, and the action memory's state, which carries what
    happened while that tactic was held. It is never told the next tactic.
  - **Output:** a prediction of the scene when the coming tactic will end,
    read by the tactic layer.
  - **Training:** tactic by tactic, against the scene reported when each
    tactic actually ended.
- **Action layer:** a transformer whose context holds:
  - the current scene;
  - the memory's expected scene;
  - the tactic token;
  - its own last 16 choices, each marked with how many decisions ago it was
    made. In play these are its own choices; in training they're whatever
    was actually used, including the teacher's, so keeping or changing
    course is something it learns.

  It emits a button action (nothing, right, right + jump, left, left + jump
  or jump) and how many frames to press it: any whole number from 1 to 32, for
  every action. It decides from the scene when to jump, stomp or stand still.

The executor (`retroagi/core/smb_executor.py`) takes only those two numbers
from the action layer and presses that button action on each of those frames;
it reads nothing else. An action ends in one of two ways, and only these two
make the layers choose a new one:
- **It ran its course:** it pressed all of its frames. A jump is the jump
  button held for its frames, so the layers may choose the next action with
  Mario still in the air.
- **Mario landed:** the vision transformer decides. Its land detector says
  whether Mario's feet are on something in the picture: the ground, a moving
  platform, or an enemy he is stomping. When it switches on after being off
  (he was in the air), the running action ends.

`retroagi/core/smb_agent.py` (`SMBAgents`) runs all of this from screens
alone, for both games.

## How it learns, in Block SMB

`retroagi/stages/block_smb/layered_train.py` trains one layer at a time, from the
bottom up, with the others frozen.
- **Action layer:** given the tactic.
- **Tactic layer:** reading the layout's strategy switch.

Which families each layer trains on (`layered_train.learner_families`):
- **Action layer:** the nine single-action families
  (`stages/block_smb/action_families.py`), one scene per action:

  | Family | The one action | Tactic | Parameters, every value swept |
  |---|---|---|---|
  | action_walk | walk right to the goal | advance | distance (24-200 px) |
  | action_walk_back | walk back left to a goal behind | retreat | distance (24-200 px) |
  | action_jump_gap | jump a pit from a standstill at its edge | advance | pit width (8-40 px), distance to the edge (0-4 px) |
  | action_climb | jump up onto a step ahead from a standstill | climb_forward | step height (8-56 px), distance to it (0-8 px) |
  | action_climb_back | jump up onto a step behind from a standstill | climb_backward | step height (8-56 px), distance to it (0-8 px) |
  | action_descend | walk off a ledge ahead down to the floor | descend_forward | drop (8-80 px), distance to the edge (0-24 px) |
  | action_descend_back | walk off a ledge behind down to the floor | descend_backward | drop (8-80 px), distance to the edge (0-24 px) |
  | action_stomp | land on an enemy walking toward Mario | advance | its distance (24-72 px), its speed (0.40-0.80 px a frame) |
  | action_wait | stand still on a moving platform that carries Mario to the goal | advance | its travel (48-112 px), its speed (0.50-1.00 px a frame) |

  Difficulty splits each family's main parameter into three ranges. The
  scenes going back to the left fit in one screen, because the camera never
  scrolls back.
- **Tactic layer:** the 12 strategy families (see below), one scene per
  tactic played under each strategy.

The action layer's training layouts are a full sweep
(`monte_carlo.block_smb_parameter_combinations`). Each family's generator is run
with every combination of every value of the parameters it draws: every whole
number of a range, every option of a choice, every order of a shuffle, and
fractional ranges (speeds) in steps of 0.01. This happens at every difficulty,
and only layouts whose teacher route wins are kept.
- **Action layer:** 10,375 layouts, all played every round, with each family
  weighing the same in learning however many layouts it has. Every
  combination has a winning teacher route.
- **Speed:** slow families are split across the workers by their first draws,
  and the made layouts are kept on disk, keyed by the code that makes them.
- **Test layouts:** held-out layouts are drawn at random from the same values
  (`monte_carlo.ParameterDraws`), so every test layout is one the sweep
  covered.
- **Limit:** the trainer refuses any family with more than 20,000
  combinations at a difficulty (`sweep_limit`) and names it.

The tactic layer's layouts are drawn at random, never swept: its families
insert coins and enemies at random, far too many combinations to play.

**Rule for each layer's families (don't forget this).** A layer trains only
on families at its own level:
- **Action layer:** scenes that each need a single action. The teacher labels
  exactly one tactic, the same at every decision from start to finish, and
  `test_a_single_action_family_asks_for_one_tactic_from_start_to_finish`
  checks it. Never scenarios that string several actions together; composing
  actions is the tactic layer's job. A tactic lasts until Mario lands (in the
  air he keeps the tactic he left the ground with), so a jump is one action
  from take-off to landing. Families such as stair_climb, platform_chain or
  stair_gap compose several actions, so they are not action families.
- **Tactic layer:** the strategy families only, once the action layer is
  trained. Each strings several actions together, and only the tactic layer
  reads the strategy switch.

The token from above comes from a teacher that reads the simulator
(`retroagi/stages/block_smb/teacher_tokens.py`). The teacher is used only in
training.

### Every layout states its tactics

Each layout carries its plan in order: a list of segments
(`retroagi/stages/block_smb/tactic_schedule.py`). A segment has a schedule
stance (advance, alternate route, hold area or retreat: the teacher's own
plan, not the tactic tokens), a direction, and optionally a route (platforms
to stand on, in order), forbidden platforms and an enemy to stomp. It ends at
a platform, at a line, after a number of frames, once a moving platform is
crossed, or once an enemy is passed or killed. The simulator follows the
segments during training:
- the goal counts only in the last segment;
- standing on a forbidden platform, or leaving a hold area early, ends the
  episode as a loss.

In the families no layer trains on now, three kinds of segment change their
stance inside, by their teacher's rule:
- moving platform: hold the area while waiting for it or riding it;
- plant: hold the area while waiting for it to go back into its pipe;
- monster: retreat from it, hold the area, then jump over it.

The teachers follow the segments (`teacher_tokens.teacher_tactic`). The
tactic is the next move under the current segment's stance:
- hold area: advance (standing still is the action layer's choice);
- backing out or keeping away (a retreat segment, or the plant and monster
  rules): retreat;
- advance or alternate route: the next step of that path, aimed at the
  segment's next route platform, else the nearest obstacle toward the goal.
  A platform higher than Mario's feet is climbed and a lower one (or a goal
  on lower ground) descended to, forward or backward by its side; anything
  else (walking, jumping a gap, getting past or stomping an enemy, boarding
  or leaving a moving platform) is advance to the right and retreat to the
  left.

A tactic lasts until Mario lands: in the air he keeps the tactic he left the
ground with, so a jump is labelled the same from take-off to landing.

The action teacher's route follows the same segments.

### Strategy families: the tactic layer's families

`retroagi/stages/block_smb/strategy_families.py` builds six scenes, one per
tactic, and plays each under both strategies: 12 families, named
`<strategy>_<tactic>` (speed_run_advance, max_points_climb_backward, ...). The
two families of a scene play the very same layouts and differ only in:
- the strategy switch the tactic layer reads;
- the route the teacher takes;
- what the episode pays and what counts as winning.

Each scene's main structure needs its tactic to reach the goal:

| Scene | Main structure |
|---|---|
| advance | a floor to the right, sometimes cut by a pit to jump |
| retreat | the goal back to the left |
| climb_forward | the goal on a step ahead |
| climb_backward | the goal on a step behind |
| descend_forward | Mario on a ledge, the goal on the floor ahead |
| descend_backward | Mario on a ledge, the goal on the floor behind |

The backward scenes fit in one screen, because the camera never scrolls back.
Difficulty sets the pit, step and drop sizes (the action families' ranges) and
how many enemies walk the way.

Coins and enemies are inserted at random:
- **on the way:** zero to three coins at Mario's height and zero to two enemies
  walking toward him, which every route meets;
- **off the way:** one or two detours, each one Mario may take or skip: coins
  on a floating platform above the way (climb onto it), coins behind him (go
  back for them), or an enemy patrolling behind him (go back and stomp it).
  A plan segment can name an enemy, and the teacher then goes to stomp it.

The teacher plays every combination of taking and skipping the detours and
measures each: frames to the goal, and points (coins collected plus enemies
killed). Each strategy takes its best combination:
- **speed run:** the fewest frames;
- **max points:** the most points, then the fewest frames.

A layout is kept only if max points' best route gathers more points than
speed run's. Each layout records what the teacher measured for every
combination (`route_results`).

What each strategy is paid for (`env.STRATEGY_REWARDS`) and how it wins:

| Strategy | Paid for | Wins when |
|---|---|---|
| Speed run | time only: a frame cost five times the usual, a time bonus at the goal for each frame left before its deadline, nothing for coins or kills | it reaches the goal within 10% of the fastest route's time |
| Max points | 25 for every coin and every kill (two and a half times the usual coin reward, five times the usual stomp reward) | it reaches the goal with at least three quarters of the extra points its route gathers beyond speed run's |

Which tactics the teacher labels in the strategy families, counted at the
decisions along its route on 6 layouts per family (2 per difficulty; every
one won; measured 2026-10-05):

| Family | Advance | Retreat | Climb forward | Climb backward | Descend forward | Descend backward |
|---|---|---|---|---|---|---|
| speed_run_advance | 41 | | | | | |
| speed_run_retreat | | 28 | | | | |
| speed_run_climb_forward | 26 | | 14 | | | |
| speed_run_climb_backward | | 19 | | 14 | | |
| speed_run_descend_forward | 25 | | 4 | | 8 | |
| speed_run_descend_backward | | 11 | | | | 11 |
| max_points_advance | 39 | 15 | 13 | | 3 | |
| max_points_retreat | 8 | 24 | | 15 | | 4 |
| max_points_climb_forward | 29 | 12 | 31 | | 3 | |
| max_points_climb_backward | 12 | 33 | | 16 | | 1 |
| max_points_descend_forward | 29 | 12 | 16 | | 13 | |
| max_points_descend_backward | 15 | 29 | | | | 10 |

Speed run walks straight to the goal; max points goes back for coins and
enemies behind (retreat, or advance in the backward scenes) and climbs onto
coin platforms, so the strategy switch changes which tactics are right.

### Families no layer trains on

These families have explicit plans but no layer trains on them now
(`retroagi/stages/block_smb/tactic_families.py`):
- **upper_route, lower_route, dead_end_retreat, monster_retreat:** a pit too
  wide to jump crossed by raised platforms; a ledge cut by an opening, crossed
  below; a dead end backed out of; a monster that can't be stomped, kept away
  from and jumped over.
- **The chained and sequence families,** composed from sections
  (`retroagi/stages/block_smb/compose.py`), whose tactics change along the way.
- **The clones** (choice_advance, choice_alternate_route, choice_hold_area,
  choice_retreat, low_choice_advance, low_choice_alternate_route): the very
  same layouts with different plans, made for the skill layer, which is gone.
- **The older basic families** (flat_run, single_gap, stair_climb, the enemy
  and moving-platform families, and the rest).

### Rounds

1. **Imitation rounds.** Workers play episodes through the vision transformer.
   At every decision the teacher says what it would do from that exact state;
   the learner trains on all such labels so far. In the first round every
   decision plays the teacher's choice, and the share falls to none by the
   last imitation round.
2. **Reward rounds.** The learner samples its own choices and learns from the
   rewards each family gives until the next decision. The teacher's labels
   still count.

The tactic layer learns, in its imitation rounds:
- **the choice:** at every decision, the teacher's tactic there;
- **the end check:** wherever a tactic is held, whether it is finished, which
  is when the teacher's tactic is no longer it. Those moments are rare, so
  they weigh four times as much;
- **its memory's prediction:** at each tactic's start, the scene when it
  ended;
- **the critic:** each decision's tactic is valued at the reward that
  followed, blended with the critic's own estimate at the next decision (the
  chance the tactic continues times its value there, plus the chance it ends
  times the value of choosing anew). Rewards are discounted per frame. The
  goal-distance shaping is made exactly potential-based for that discount,
  so it can't change which tactic is best.

Its reward rounds change only the tactic transformer, and only once its
critic predicts held-out returns well enough (explained variance at least
0.5, measured every round):
- **the choice**, where it chose by sampling: the clipped policy-gradient
  rule, crediting the return over the value of the situation;
- **the end check**, the option-critic rule: lower the chance of ending where
  holding the tactic is worth more than choosing anew, raise it where it is
  worth less, with a small cost for every switch so it doesn't flip back and
  forth.

A layer counts as trained when every family wins at least 90% of its
held-out layouts (`--validation-layouts-per-difficulty`, 3 per difficulty by
default). A strategy family's episode is won only with its strategy's
objective met: in time for speed run, with enough points for max points. The
tactic layer must also agree with the teacher on where tactics
change: each of its changes is matched to one of the teacher's within two
decisions, and at least 70% must match both ways. After the tactic layer,
the whole agent is also scored as deployed: every token is its own, under
each layout's strategy switch. The best round that passes is saved as
`passed.pt` (of equally good rounds, the latest), and the next layer starts
from it.

```bash
retroagi-block-smb train-layer --learner action --output artifacts/block_smb/action
retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/action/passed.pt --output artifacts/block_smb/tactic
```

## Full SMB

The layers trained in Block SMB play Full SMB unchanged, through the Full SMB
vision transformer. `retroagi/stages/full_smb/play.py` gives the agent screens;
game memory is read only to score how far Mario got.

```bash
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
```

The strategy switch is set for the run: `--strategy speed_run` or `max_points`
(speed run by default).
