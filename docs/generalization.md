# How general is it? Measurements on levels nothing was tuned on

The agent's rules, its simulator and its constants were written while
watching two levels of the game, World 1-1 and World 5-1. This page
measures what that work is worth on levels nobody looked at, and records
each step taken to make the agent general against that measure. Nothing
learns on the real game in any of it.

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
  sharing a start are not independent.
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

The rules stay the default for now: the simulator gives no tracks, so the
outcome network cannot yet be given outcomes sampled with the observed
model.

## What the measurements say to do next

In the order the deaths rank them:

1. **Remember what went out of sight.** A plant that went down its pipe is
   still there. The model should keep things that disappeared into a
   solid, learn from the first one seen to return how long they stay away,
   and give the planner their return as a predicted hazard. This is the
   largest single cause of death and no model of visible things can touch
   it.
2. **When nothing is feasible, act at once.** The chooser tries one choice
   a frame and Mario slides on meanwhile; the choices should be planned in
   one frame and the safest taken.
3. **Margins that follow the model's own error.** The planner keeps every
   enemy 2 pixels wider than its box. The model can measure how wrong its
   predictions for each type have been (Lakitu, Spinies in the air) and
   keep away by that much instead.
4. **The simulator's rules against the game's.** The same check, run on
   the simulator's enemies, would show where the practice world teaches
   the outcome network something the game does not do (the Goomba's fall
   is one).
5. **Tracks in the simulator**, so that the observed model can be the one
   outcomes are sampled with, and the default.
