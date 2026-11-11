# How general is it? Measurements on levels nothing was tuned on

The agent's rules, its simulator and its constants were written while
watching two levels of the game, World 1-1 and World 5-1. This page
measures what that work is worth on levels nobody looked at, and records
each step taken to make the agent general against that measure. Nothing
learns on the real game in any of it.

## How far from a working agent

Three numbers, on the held-out levels (`scripts/dev/scorecard.py FOLDER`):

| | First contact | After the second round | A working agent |
|---|---|---|---|
| Whole levels finished from the screen alone | 0 of 32 plays | 0 of 32 | most |
| Tries at an obstacle that fail, from the screen alone | 47% | 41% | about 2% |
| The same with exact positions in place of seeing | 35% | 24% | about 2% |

A level has about 35 obstacles, so half its plays finish only when about 2%
of tries at an obstacle fail (0.98 to the 35th is a half), and 9 in 10 at
about 0.3%. The agent now passes 2.4 obstacles between failures from the
screen and 4.2 with exact positions; it needs about 50. World 1-1, one of
the two levels it was built on, it finishes from the screen in every play.

"With exact positions" (elsewhere here: "from game memory") is a measuring
device, not the agent: the positions of Mario, the enemies and the ground
are read from the emulator's memory and handed to the agent in place of
what it would see. It shows what choosing and controlling lose by
themselves. The agent proper is given the screen's picture and nothing
else ("from vision").

## The measure

- **Three sets of levels.** *Tuned*: 1-1 and 5-1. *Development*: 2-1 and
  3-1, where single failures may be looked at to understand a kind of
  failure. *Held out*: 4-1, 6-1, 7-1 and 8-1, where only totals and counts
  by cause are read, never a single failure, and for which nothing is
  written. A fix that only helps the levels it was found on shows as a gap
  between the last two sets.
- **Obstacle benches** (`framework/bench.py`): each level's obstacles with
  a saved state shortly before each
  (`python -m retroagi.domains.smb.nes.obstacles --level Level2-1`). A
  bench covers the part of a level that a random player, set back after
  each death, reached while mapping it: all of 3-1, 4-1, 6-1 and 7-1, 2-1
  to x 2077 and 8-1 to x 3806.
- **Tries that differ.** Play is deterministic: the four tries of an
  obstacle used to be the same play four times. Each further try now lets
  the world run 6 frames longer before the agent acts
  (`agent.START_WAITS`), so the enemies are elsewhere in their walks.
  Obstacles that share a saved state are scored in one play, and results
  are also given by distinct start state (in brackets below): obstacles
  sharing a start are not independent. An obstacle not passed within 900
  frames has stalled (from the second round on, 900 frames after the
  obstacle before it in the play was passed).
- **Whole levels** (`framework/play.py --runs 4`): from the level's start,
  in both ways to play. One death anywhere ends a play, so this is far
  harder than the bench.
- **Three agents**: the outcome network's chooser with the state from game
  memory (perfect perception: what the controller and chooser are worth),
  the hand-written chooser from game memory (what the network adds), and
  the network's chooser from vision alone (the agent proper).

`scripts/dev/generalization_table.py FOLDER` makes the tables below from a
folder of results; `scripts/dev/bench_table.py` gives one result with an
interval.

## First contact, 2026-11-11

Network `outcome_20261109d`, the hand-written dynamics rules. Share of
obstacles passed (and of the plays from distinct starts); whole levels as
plays finished of 8 and the middle play's furthest point.

| Level | memory, network | memory, scripted | vision, network | whole level, memory | whole level, vision |
|---|---|---|---|---|---|
| 1-1 (tuned) | 0.98 (0.99) | 0.98 (0.97) | 0.95 (0.96) | 8/8, x 3161 | 4/8, x 3161 |
| 5-1 (tuned) | 0.94 (0.93) | 0.70 (0.72) | 0.74 (0.80) | 0/8, x 1816 | 2/8, x 1840 |
| 2-1 | 0.52 (0.61) | 0.60 (0.84) | 0.43 (0.57) | 0/8, x 345 | 0/8, x 466 |
| 3-1 | 0.70 (0.77) | 0.45 (0.54) | 0.39 (0.47) | 0/8, x 1800 | 0/8, x 371 |
| 4-1 | 0.69 (0.60) | 0.85 (0.80) | 0.48 (0.80) | 4/8, x 3593 | 0/8, x 1830 |
| 6-1 | 0.53 (0.72) | 0.19 (0.44) | 0.34 (0.42) | 0/8, x 1378 | 0/8, x 506 |
| 7-1 | 0.69 (0.65) | 0.45 (0.53) | 0.42 (0.53) | 0/8, x 637 | 0/8, x 998 |
| 8-1 | 0.68 (0.80) | 0.35 (0.51) | 0.75 (0.78) | 0/8, x 1217 | 0/8, x 1732 |
| **tuned, all obstacles** | **0.96** | **0.82** | **0.83** | | |
| **untouched, all obstacles** | **0.65** | **0.47** | **0.49** | | |

What it says:

- **The agent is tuned to its two levels.** With perfect perception it
  passes 96% of the obstacles it was built on and 65% of those it was
  not; from vision, 83% and 49%.
- **Almost every failure is a death, not a stall**, and perception is not
  the main cause: two thirds of the gap is there with the state read from
  game memory.
- **The earlier numbers flattered it.** With tries that differ, 5-1 from
  vision is 0.74, not the 0.87 measured when all four tries were the same
  play. From its start, 5-1 is finished in 2 of 16 plays; only 1-1 is
  finished reliably (and from vision only in a speed run).
- **The outcome network helps where the hand-written chooser was not
  tuned** (0.65 against 0.47), though not on every level.

### What kills it

`scripts/dev/bench_deaths.py` replays the bench and reads the game's own
account of each death. With game memory and the network's chooser, on the
six untouched levels: 114 deaths in 496 plays.

| What Mario met | Deaths | Where |
|---|---|---|
| Piranha Plant | 34 | 2-1, 3-1, 4-1, 7-1, 8-1 |
| Spiny (thrown by Lakitu) | 19 | 4-1, 6-1 |
| Green Koopa | 16 | 2-1, 3-1, 7-1 |
| Goomba | 14 | 2-1, 3-1, 8-1 |
| A fall | 11 | 3-1, 8-1 |
| Buzzy Beetle | 10 | 7-1 |
| Hopping Paratroopa | 5 | 3-1, 7-1 |
| Bullet Bill | 3 | 7-1 |
| Something in no enemy slot (a hammer) | 2 | 3-1, 7-1 |

On the development levels, where single deaths may be read
(`scripts/dev/trace_obstacle.py`):

- **A plant the agent could not see.** In 2-1 the agent stood by a pipe
  whose plant was down inside it, jumped onto the pipe and met the plant
  coming up. A plant in its pipe is in no state the controller is given,
  and nothing remembers that one went down there a moment ago.
- **A Goomba walking down a staircase** (2-1's first steps, every try): it
  dropped from step to step into Mario's jump. The rules had enemies
  falling three times faster than the game's do.
- **A Koopa falling from above** onto where Mario had just landed (3-1):
  every choice was infeasible, the chooser tried one a frame for 17 frames
  while Mario slid on, and turning back came too late.

## How well the controller's dynamics model predicts

A controller is as good as its model of how the other things move.
`python -m retroagi.domains.smb.nes.dynamics_check --bench DIR` scores a
model apart from the agent's play: at every frame of recorded play it
predicts each enemy's box 8, 16, 32 and 48 frames ahead from what the
controller was given then, and compares with where the game had put that
enemy. Below, the share of predictions within 4 pixels of the truth.

The hand-written rules (`planner.RULES`), 16 frames ahead: **0.91 on the
two levels they were written on, 0.68 on the four held-out ones.** Where
they have no rule for a kind (a Spiny 0.34, a Buzzy Beetle 0.48, Lakitu
0.51) or a rough one (a hopping Paratroopa 0.54), they are wrong half the
time. And one rule was simply wrong: a Goomba falling off a ledge gains
0.15 to 0.2 pixels a frame each frame up to about 3.2, not the 0.5 up to
4.5 written in the rules and in the simulator the outcome network learns
in.

## Step: a dynamics model identified from what is seen

`framework/observed_dynamics.py` (`--use dynamics=observed`) is told
nothing of what any thing is. It fits each thing's motion to its own track
(still, steady, or speeding up downward), reads from the solids whether it
rests on a top, is inside a solid or is in the air, and remembers by the
type label the state source gives what things of that type have been seen
to do: how hard they fall and how fast at most, how fast they leave the
ground when they bounce, how far below their box their feet are, how far
out of a solid they come and at what pace, whether they pace to and fro.
Its memory starts empty at each play.

It was developed on 1-1, 5-1, 2-1 and 3-1 only. Share of predictions within
4 pixels, at 8 / 16 / 32 / 48 frames ahead, a new model each play:

| Kind (predictions) | rules | observed |
|---|---|---|
| **Development levels, all (31,731)** | 0.94 / 0.87 / 0.83 / 0.79 | 0.93 / **0.90** / 0.85 / 0.81 |
| **Held-out levels, all (28,600)** | 0.81 / 0.68 / 0.57 / 0.51 | 0.86 / **0.74** / 0.62 / 0.56 |
| Held out: hopping Paratroopa (2,874) | 0.87 / 0.54 / 0.40 / 0.42 | 0.87 / **0.80** / 0.69 / 0.66 |
| Held out: Spiny (5,087), never met in development | 0.43 / 0.34 / 0.23 / 0.17 | 0.67 / **0.51** / 0.31 / 0.24 |
| Held out: Buzzy Beetle (898), never met in development | 0.64 / 0.48 / 0.31 / 0.32 | 0.70 / **0.71** / 0.65 / 0.58 |
| Held out: Goomba (3,784) | 0.97 / 0.95 / 0.91 / 0.86 | 0.98 / 0.94 / 0.91 / 0.86 |
| Held out: green Koopa (3,498) | 0.97 / 0.94 / 0.91 / 0.86 | 0.97 / 0.95 / 0.93 / 0.89 |
| Held out: Bullet Bill (4,263) | 0.95 / 0.95 / 0.94 / 0.94 | 0.96 / 0.95 / 0.95 / 0.95 |
| Held out: Piranha Plant (1,950) | 1.00 / 0.88 / 0.79 / 0.74 | 0.94 / 0.86 / 0.79 / 0.74 |
| Held out: Hammer Bro (913) | 0.85 / 0.67 / 0.39 / 0.16 | 0.88 / 0.67 / 0.37 / 0.15 |
| Held out: Lakitu (5,333) | 0.74 / 0.51 / 0.27 / 0.19 | 0.77 / 0.47 / 0.25 / 0.19 |

It predicts better than the rules on levels neither was written on, most
where the rules had none, and no worse than them where they were tuned. It
learned the Goomba's true fall and the Paratroopa's hop from watching one.
What it cannot predict is what nothing seen so far predicts: Lakitu and
the Hammer Bro move by what Mario does, and a plant's stay at the top of
its pipe has a length the model has not been given a way to learn.

### Playing with it

The same benches and whole-level plays as at first contact, the observed
model in the controller's place of the rules (the outcome network is the
same, though its outcomes were sampled with the rules). Share of obstacles
passed (and of the plays from distinct starts):

| Level | memory, network | vision, network | whole level, memory | whole level, vision |
|---|---|---|---|---|
| 1-1 (tuned) | 1.00 (1.00) | 0.98 (0.98) | 8/8, x 3161 | 7/8, x 3161 |
| 5-1 (tuned) | 0.94 (0.93) | 0.86 (0.82) | 0/8, x 1818 | 1/8, x 1904 |
| 2-1 | 0.51 (0.66) | 0.57 (0.73) | 0/8, x 337 | 0/8, x 1730 |
| 3-1 | 0.70 (0.79) | 0.54 (0.61) | 0/8, x 2429 | 0/8, x 1474 |
| 4-1 | 0.73 (0.60) | 0.45 (0.75) | 4/8, x 3593 | 0/8, x 1836 |
| 6-1 | 0.53 (0.70) | 0.44 (0.44) | 4/8, x 2969 | 0/8, x 1081 |
| 7-1 | 0.74 (0.74) | 0.39 (0.60) | 0/8, x 645 | 0/8, x 985 |
| 8-1 | 0.69 (0.78) | 0.81 (0.84) | 0/8, x 2516 | 0/8, x 1914 |
| **tuned, all obstacles** | **0.96** | **0.91** | | |
| **untouched, all obstacles** | **0.66** | **0.55** | | |

Against the rules, by set of levels (obstacles passed; by distinct start
with its 90% interval):

| State | Levels | rules | observed |
|---|---|---|---|
| Game memory | tuned | 0.96 (0.96, 0.94-0.99) | 0.96 (0.97, 0.94-0.99) |
| Game memory | development | 0.63 (0.71, 0.56-0.85) | 0.63 (0.74, 0.60-0.88) |
| Game memory | held out | 0.65 (0.73, 0.63-0.83) | 0.68 (0.73, 0.64-0.82) |
| Vision | tuned | 0.83 (0.89, 0.81-0.95) | **0.91** (0.91, 0.84-0.97) |
| Vision | development | 0.41 (0.51, 0.36-0.65) | **0.55** (0.65, 0.52-0.77) |
| Vision | held out | 0.53 (0.66, 0.56-0.75) | 0.55 (0.70, 0.61-0.78) |

- **From vision the agent plays better with it**: 8 points on the tuned
  levels and 14 on the development ones. Fitting a thing's motion to its
  whole track serves noisy boxes better than a rule picked by vision's
  guess of the thing's kind. On the held-out levels the gain is 2 points,
  inside the interval: not shown.
- **From game memory nothing changes** (114 deaths at first contact, 109
  with it): the better predictions are of things that were not what was
  killing the agent. Some obstacles are won (2-1's staircase Goomba: 0 of
  8 tries to 4) and others lost.
- **Whole levels**: 24 of 128 plays finished against 18 (6-1 is finished
  in 4 of 8 plays from game memory, where none was), and the middle play
  goes further on most levels.

(That was so when this was measured; the second round below gave the
simulator tracks and made this model the default.)

## Second round, 2026-11-11: following the deaths

Each step below was measured on its own, with the state from game memory
(the benches take two minutes that way) and the same network; the table
after them has the whole of it from vision too.

**The bench's frame limit counts again from each obstacle passed.** A play
that covers a dozen obstacles from one saved state had 900 frames for all
of them, so a patient wait at one pipe "stalled" every obstacle after it.
With the limit counted from the last obstacle passed, the first-contact
agent scores 0.96 on the tuned levels and 0.66 on the untouched ones (0.65
before), with 118 deaths in 496 plays there. The steps are measured against
that.

| Step (state from game memory) | Tuned | Untouched | Deaths in 496 plays on the untouched levels |
|---|---|---|---|
| The first-contact agent | 0.96 | 0.66 | 118 |
| The dynamics model fitted to what is seen | 0.97 | 0.68 | 110 |
| A refused choice answered in the same frame | 0.97 | 0.72 | 77 |
| A plant shown when its head shows, as vision sees it | 0.97 | **0.77** | **70** |
| (remembering what went out of sight: not kept) | 0.97 | 0.69 | 68 |
| The hand-written rules with the same fixes, for comparison | 0.94 | 0.73 | 85 |

- **A refused choice is answered in the same frame.** The chooser offered
  one choice a frame, and while the controller refused them the agent went
  on as it was: Mario slid into a pit of 5-1 past seven refused landings.
  The agent now asks for the next choice at once, up to 8 a frame
  (`agent.TRIES`). A third of the deaths on the untouched levels went with
  it.
- **Perfect perception was blinder than vision about plants.** The
  game-memory state source showed a piranha plant only once the box the
  game collides with (14 pixels below the plant's head) had cleared the
  pipe: 2 frames before it reached a Mario landing there. Vision sees the
  head 28 frames sooner, and the simulator shows it then too. Shown when
  its head shows, a plant coming up is seen in time to steer the jump
  away: plant deaths fell from 26 to 9. Most of what looked like a need
  to remember hidden plants was this.
- **Remembering what went out of sight was tried and not kept.** The model
  kept each thing last seen going into a solid, learned from the first one
  seen to return how long its type stays away, and gave the planner its
  return as a thing to keep clear of. It did what it was built to (a plant
  is away 120 frames, and the model found that), but the agent then stood
  beside the pipe waiting for a plant that, in this game, stays down as
  long as Mario stands there: 10% of its tries stalled, and no fewer died.
  Dropping the expectation after a long wait only moved the deaths (the
  plant comes up the moment Mario steps back for a run-up). To use such a
  memory the model must also learn what keeps a thing away. The experiment
  is kept as a patch outside the tree, not in the code.
- **The simulator gives tracks, and falls as the game does.** Each enemy's
  and mushroom's places while in sight are kept in the simulator's own
  state (so they are saved and put back with it), and outcome sampling
  plays each candidate with a controller that knows what the playing
  agent's model has learned. Under the game's rules the simulator's
  walkers and mushrooms now fall off a ledge as the game's do (0.2 pixels a
  frame gained each frame up to 3.4, fitted to Goombas and Koopas falling
  in the game: 1.6 pixels off on average, where the 0.5 up to 4.5 written
  first was 11 off). On 30 fresh random levels the hand-written chooser
  finishes 47 of 60 plays with the fitted model and 42 with the rules; the
  network's chooser 53 and 52.
- **The fitted model is the default** (`--use dynamics=rules` for the
  rules). The rules were corrected on the way (the fall, and the pipe of a
  plant whose box is still inside it), which is most of why they are now
  closer to it than at first contact.

### Where it stands

Network `outcome_20261109d` (its outcomes were sampled with the rules and
one choice a frame: a network sampled with the present agent is being
trained). Share of obstacles passed (and of the plays from distinct starts,
with its 90% interval):

| State | Levels | First contact | The rules, now | The default, now |
|---|---|---|---|---|
| Game memory | tuned | 0.96 (0.97, 0.94-0.99) | 0.94 (0.95, 0.91-0.98) | 0.97 (0.97, 0.94-0.99) |
| Game memory | development | 0.64 (0.71, 0.56-0.85) | 0.72 (0.71, 0.57-0.85) | **0.78** (0.83, 0.71-0.93) |
| Game memory | held out | 0.67 (0.75, 0.66-0.84) | 0.74 (0.79, 0.71-0.87) | **0.76** (0.80, 0.73-0.88) |
| Vision | tuned | 0.83 | 0.87 (0.92, 0.85-0.97) | 0.85 (0.89, 0.82-0.94) |
| Vision | development | 0.41 | 0.56 (0.64, 0.48-0.77) | **0.58** (0.69, 0.57-0.81) |
| Vision | held out | 0.53 | 0.57 (0.71, 0.61-0.80) | **0.59** (0.72, 0.63-0.80) |

(First contact from vision was measured under the old frame limit.)

| Level | memory | vision | whole level, memory | whole level, vision |
|---|---|---|---|---|
| 1-1 (tuned) | 1.00 (1.00) | 0.95 (0.95) | 8/8, x 3161 | 8/8, x 3161 |
| 5-1 (tuned) | 0.96 (0.93) | 0.78 (0.82) | 3/8, x 1905 | 0/8, x 1906 |
| 2-1 | 0.88 (0.93) | 0.60 (0.79) | 4/8, x 3193 | 0/8, x 1731 |
| 3-1 | 0.71 (0.78) | 0.56 (0.64) | 0/8, x 1963 | 0/8, x 1471 |
| 4-1 | 0.86 (0.80) | 0.52 (0.78) | 2/8, x 2408 | 0/8, x 1911 |
| 6-1 | 0.54 (0.67) | 0.42 (0.48) | 2/8, x 2290 | 0/8, x 509 |
| 7-1 | 0.87 (0.86) | 0.55 (0.62) | 0/8, x 647 | 0/8, x 1365 |
| 8-1 | 0.75 (0.83) | 0.77 (0.84) | 0/8, x 2703 | 0/8, x 2491 |
| **tuned, all obstacles** | **0.97** | **0.85** | | |
| **untouched, all obstacles** | **0.77** | **0.58** | | |

- **With perfect perception the gap between tuned and untouched levels has
  gone from 0.96 against 0.66 to 0.97 against 0.77**, on the held-out
  levels as on the development ones, and the deaths there from 118 to 70.
  Whole levels finished: 27 of 128 plays (18 at first contact), 1-1 in all
  16, and 2-1, 4-1 and 6-1 for the first time (from game memory).
- **Perception is now the larger loss**: 0.77 with the state from game
  memory, 0.58 from vision, and from vision twice as many tries end in a
  death (38% against 18%). Nothing in this round was aimed at it.
- **From vision, the fitted model and the corrected rules are level**
  (0.58 and 0.57 on the untouched levels, the tuned levels within their
  intervals); with perfect perception the fitted model is ahead (0.77
  against 0.73, 70 deaths against 85).
- **What kills it now** (70 deaths on the untouched levels, game memory):
  Lakitu's Spinies 25 (all in 6-1 and 4-1), Goombas 10, piranha plants 9,
  Koopas 7, hopping Paratroopas 7.

How well each model predicts, re-measured with plants shown from their
heads (within 4 pixels at 8 / 16 / 32 / 48 frames ahead): on the tuned
levels the rules 0.95 / 0.90 / 0.85 / 0.81 and the fitted model 0.94 / 0.91
/ 0.86 / 0.81; on the held-out levels 0.83 / 0.69 / 0.56 / 0.51 and 0.87 /
**0.76** / 0.63 / 0.56 (hopping Paratroopas 0.53 and 0.80 at 16 frames,
Spinies 0.41 and 0.58, Buzzy Beetles 0.68 and 0.80, plants 0.80 and 0.88).

## What seeing costs, 2026-11-11

Vision was trained on frames of all eight levels, and it is about as
accurate on the untouched ones as on 1-1
(`python -m retroagi.domains.smb.nes.perception_check --bench DIR` lays its
state over the exact one on the same frames): Mario within 2 pixels at the
median and 5 at the 90th percentile, 85% to 99% of each kind of enemy
given, within 1 to 2 pixels (3 to 8 at the 90th percentile), the ground
agreed in 96% to 98% of 8-pixel columns around Mario. Small errors, yet
from the screen the agent fails 41% of its tries where with exact
positions it fails 24%.

`scripts/dev/perception_parts.py` plays the bench from the screen with some
parts of the state replaced by the exact ones:

| Exact in place of seen | Tuned | Untouched |
|---|---|---|
| Nothing (the agent proper) | 0.85 | 0.58 |
| Mario's own body and motion | 0.89 | 0.65 |
| The enemies and items | 0.89 | 0.67 |
| The ground, pipes and blocks | 0.80 | 0.55 |
| All three | 0.95 | 0.78 |

What vision loses is in Mario's own position and motion and in the
enemies, about equally, and not in the ground. (Exact ground under a
Mario placed a few pixels off is no better than seen ground.)

## Tried and not kept, 2026-11-11

Measured with exact positions, against 0.97 on the tuned levels and 0.77
on the untouched ones:

- **Room around a thing by the model's own record of error** (the model
  scored its predictions 8, 16 and 32 frames on, and things that cannot be
  stomped were kept clear of by what three quarters of the predictions for
  their type had been off by): 0.97 and 0.75. Fewer deaths on 4-1 and none
  fewer on 6-1, the level of Spinies it was meant for.
- **A run or wait ended when the model's paths bring a thing from another
  level onto Mario** (a Koopa falling from a ledge, a thrown Spiny): 0.97
  and 0.77, 67 deaths for 70. Too little to keep.
- **A wait credited with no progress** (the chooser took short waits for
  the ground Mario slid over during them): 0.97 and 0.77, some levels up
  and others down.

What the deaths show instead (`scripts/dev/trace_obstacle.py --why`): on
the held-out levels half of them happen during a run or a wait, and at
the decisions before one the outcome network gives every choice no chance
of death with a Goomba a few steps away. Its lessons call a choice safe
while any way out is left after it, so the chooser waits, safely, and
waits again, until none is left. Outcomes are now sampled with a second
answer, **danger** (dead soon after the choice if the agent then does
nothing, a way out or not), the network predicts it, and the chooser
prefers choices that do not leave Mario so (`--use chooser=plain` for the
chooser without it). The network sampled this way, with the present agent,
is in training (outcomes_20261111k).

## What the measurements say to do next

1. **Mario's own position and motion, and the enemies, as seen.** The loss
   from the screen is now the largest, and it is there (above): each made
   exact gives back 7 to 9 points. Then outcome sampling with perception's
   kind of error, so the chooser learns with it.
2. **Things thrown from above.** Lakitu's Spinies are a third of the
   deaths: a thing first seen falling, from a thrower that follows Mario.
   The model predicts their fall better than the rules (0.58 against 0.41)
   but the chooser has no choice made for them.
3. **The outcome network sampled with the present agent, with danger** (in
   training), and after it the same check for the simulator's enemies as
   for the game's: where the practice world moves a thing as the game does
   not.
4. **Margins that follow the model's own error**: the planner keeps every
   enemy 2 pixels wider than its box, whatever the model's record with its
   type (Lakitu and the Hammer Bro are within 4 pixels half the time).
5. **What a model may keep from one play to the next** is the user's to
   decide: its memory starts empty at each play, and what it learns of a
   type in one play would serve the next.
