# How general is it? Measurements on levels nothing was tuned on

The agent's rules, its simulator and its constants were written while
watching two levels of the game, World 1-1 and World 5-1. This page
measures what that work is worth on levels nobody looked at.

## The rule (user, 2026-11-11)

The project stands in for a robot, which cannot be trained or debugged in
its real situation: a trained, working system must exist before it meets
the real environment. So the real game may be looked at for three things
only:

1. **to train seeing** (the vision network: there is no way around it);
2. **to make Block SMB behave as the game does** (measured from the game's
   memory: copying how a system behaves is a far simpler problem than
   learning to play in it). Its behaviour, not its graphics and not its
   levels: Block SMB is meant to be fast, cheap and drawn as plain blocks,
   and no model may learn a real level's layout;
3. **to score the finished system.**

All of the owner's requirements, word for word, are in
[requirements.md](requirements.md).

The policy, everything that decides and controls, is learned from Block
SMB alone. A failure in the real game may only be turned into the
question "what does Block SMB lack, or do differently?"; Block SMB is
corrected, the policy learns again there, and it is debugged on fresh
Block SMB levels.

**The first two rounds below broke this rule.** Deaths on real levels were
traced and the agent changed to answer them; a dynamics model was shaped
against recordings of real levels; "development levels" of the real game
were declared for the purpose. What they measured stands as measurement.
What they changed in the policy was put back the same day, each change
first scored where it should have been, on 100 fresh Block SMB levels
(plays finished of 200, network `outcome_20261109d`):

| The policy | Finished |
|---|---|
| As before the rounds (hand-written motion rules, one choice a frame) | 188 |
| The dynamics model fitted to what is seen | 187 |
| A refused choice answered in the same frame | 184 |
| Both | 185 |

None earns its place in Block SMB, so none is kept: the rules are the
default again, a choice is tried once a frame, and outcomes are sampled as
before (the "danger" answer, drawn from a death on a real level, is gone;
the resampling begun with it was stopped). The fitted model remains as a
named alternative (`--use dynamics=observed`), an experiment until Block
SMB gives it a reason. What stays from the rounds is what the rule allows:
the measuring tools, and corrections to Block SMB itself (its walkers fall
as the game's do, a standing enemy stands exactly, every thing has its
track). With those corrections the hand-written chooser finishes 192 of
the 200 plays and the network's 183: the network learned in the Block SMB
of before, and is due to learn again in this one.

The real-game numbers further down are of the policies of those rounds,
not of the present one.

## How far from a working agent

Three numbers, on the held-out levels (`scripts/dev/scorecard.py FOLDER`):

| | First contact | After the second round | Learned in Block SMB alone (2026-11-18) | A working agent |
|---|---|---|---|---|
| Whole levels finished from the screen alone | 0 of 32 plays | 0 of 32 | 2 of 32 | most |
| Tries at an obstacle that fail, from the screen alone | 47% | 41% | 17% | about 2% |
| The same with exact positions in place of seeing | 35% | 24% | 20% | about 2% |

(The first two columns are of policies changed by hand while watching real
levels, which the rule below forbids and which were put back. The third is
of the present policy: its deciding network learned in Block SMB alone,
`outcome_sided_20261117`.)

A level has about 35 obstacles, so half its plays finish only when about 2%
of tries at an obstacle fail (0.98 to the 35th is a half), and 9 in 10 at
about 0.3%. The agent now passes 5.9 obstacles between failures from the
screen and 5.0 with exact positions; it needs about 50. World 1-1, one of
the two levels it was built on, it finishes from the screen in 6 of its 8
plays.

"With exact positions" (elsewhere here: "from game memory") is a measuring
device, not the agent: the positions of Mario, the enemies and the ground
are read from the emulator's memory and handed to the agent in place of
what it would see. It shows what choosing and controlling lose by
themselves. The agent proper is given the screen's picture and nothing
else ("from vision").

## Learning from Block SMB alone: how it is set up (2026-11-11)

Each piece has its own data and its own test:

| Piece | Made or learned from | Tested by |
|---|---|---|
| Seeing (the vision network) | frames of the real game | frames of the real game not trained on |
| Block SMB (how it behaves) | the real game's memory | the same situation in both, compared (`nes/clone.py --check`, `nes/dynamics_check.py`) |
| Deciding (the outcome network) | every choice tried in Block SMB | the Block SMB exam, on worlds not trained on |
| The whole agent | nothing more | the Block SMB exam; then the real game, to score |

**Which real levels may be looked at.** Worlds 2-1, 3-1 and 5-1 are for
building Block SMB: what pieces they are made of is measured, and Block
SMB's behaviour is checked against them. Worlds 1-1, 4-1, 6-1, 7-1 and 8-1
are for scoring only: the agent is played on them and the score read, and
nothing of them goes into Block SMB. No level's layout, of either set, is
ever a practice world.

**Where the policy's data comes from** (`framework/sampling.py`): in Block
SMB every choice open at a decision is played out from a saved copy of that
moment and its outcome kept. The worlds are

- *skill scenes* (`sim/levels.skill_scene`): a short stretch calling for
  one skill, one piece of the game (a pipe, a pit, enemies, bricks, stairs,
  question blocks, a cannon) or two pieces side by side, with no flag;
- *whole random levels* of the same pieces, where skills follow one another.

Both are drawn at random from the game's pieces. (For a few hours on
2026-11-11 the three building levels, laid out in Block SMB, were a third
kind of practice world. The owner ruled it out the same day, "I don't want
the models learning the full SMB layout", and they were taken out before
any network had trained on them.)

The play that visits the states is an outcome network's own (`--driver`),
with a share of its decisions made at random: a policy learns from where
its own play takes it, and the hand-written chooser is out of the data (it
remains as a yardstick).

**The Block SMB exam** (`framework/practice.py --levels N --scenes N`):
worlds not trained on, counted by kind, so a weak skill shows by name.
Before any new training (100 levels, 100 scenes of each kind, both ways to
play):

| | Whole levels finished | Skill scenes passed |
|---|---|---|
| The hand-written chooser (the yardstick) | 192 of 200 | 1552 of 1600 |
| The outcome network `outcome_20261109d` | 183 of 200 | 1572 of 1600 |

Every scene failed was an enemy scene, and every one of those was the
scene's own fault: its enemy started on top of Mario (since put right: no
piece starts nearer than 5 tiles). Of scenes with a fair start both passed
all.

**What real levels have that Block SMB's generator did not**
(`scripts/dev/piece_catalogue.py`, on the three building levels): pipes 5
and 6 tiles tall (it drew 2 to 4), pieces side by side or one or two tiles
apart (it left 3 to 8 between them), a pit of 8 tiles under a bridge of
floating tiles, and Hammer Bros. The pair scenes now put pieces 0 to 2
tiles apart; the rest is still to be added to the generator, as pieces
drawn at random (a tall pipe with a way up beside it, a bridge of floating
tiles over a long pit, pieces touching in whole levels too).

**Checking that Block SMB behaves as the game does** (`nes/clone.py
--check`): a building level's layout is laid out in Block SMB's format,
Block SMB's Mario is put where the game's Mario is, moving as he moves, and
the same buttons pressed for 16 frames, at a few hundred moments along the
level. He ends within 2 pixels of the game's Mario in 93% (2-1), 96% (3-1)
and 92% (5-1) of them. The layouts lack lifts, what question blocks hold,
blocks that are invisible until hit, the last staircase's upper steps (the
game's memory could not be read once the castle is in view), and the two
Hammer Bros of 3-1. They are kept in `data/full_smb_layouts` for this check
alone.

**Round 1** (`outcome_20261111k`): 2.08 million choices tried in 4,800
skill scenes and 1,000 plays of random levels, the old network's play
visiting the states.

| The Block SMB exam (worlds not trained on; plays got through) | Random levels | Skill scenes |
|---|---|---|
| The hand-written chooser (the yardstick) | 192 of 200 | all with a fair start |
| The old network `outcome_20261109d` | 183 of 200 | all with a fair start |
| Round 1 | 185 of 200 | 1593 of 1600 |

Each count is of plays, not of worlds: 100 random levels played both ways
(for speed, for points) are 200 plays.

*Withdrawn.* While the three laid-out building levels were practice
worlds, the exam also had 200 plays on them: 100 starts, each one of the
three levels entered at its start or at a random spot along it, played
both ways. (Three levels, 200 plays: not 200 levels.) 13 of those plays
reached the flag with the hand-written chooser, 14 with the old network,
17 with round 1. It showed that the policy which passes Block SMB's random
levels does not pass arrangements real levels have, at 2-1's first
staircase, 3-1 before its bridge and 5-1's tall pipe. That is put right by
teaching Block SMB's generator those pieces, not by practising on the
levels: the exam on them is gone with them.

**Round 2 was stopped** (2026-11-11): it had begun to sample outcomes on
the laid-out levels. What it had sampled on them was deleted; what it had
sampled in random levels and skill scenes, with round 1's network playing,
is kept (`outcomes_20261111l`, unfinished).

**The generator widened** (2026-11-11, `sim/levels.py`). What the three
building levels are made of, beyond what the generator drew, was measured
and added as pieces drawn at random:

| In the building levels | Now drawn |
|---|---|
| Pipes 5 and 6 tiles tall, each with something lower to jump from before it | Pipes 2 to 6 tiles; one over 4 has a way up: a lower pipe, a step, or a ledge with a pit between |
| Staircases that only go up (to 5 and 6 tiles), Goombas standing on their steps, a pit at the foot | Staircases up and down, up only (3 to 6 tiles, Goombas on the steps, sometimes a pit at the foot), or a wall with steps down |
| Lone steps, and two of unlike heights side by side (3 then 6) | A step of 1 to 4 tiles, half the time with a second beside it |
| A walkway of floating tiles over a pit 8 tiles wide, five Goombas on it | A bridge: steps up to a walkway over a pit 5 to 8 tiles wide, up to three Goombas on it |
| Brick rows over pits; rows 5 tiles up; rows 11 tiles long; Koopas walking on rows | Pits roofed with bricks a quarter of the time; rows 4 or 5 tiles up and 1 to 11 long, an enemy on one in four |
| Koopas, Goombas and Paratroopas in twos and threes | Files of one to three |
| Pieces touching, or one or two tiles apart | 0 to 8 tiles of floor between pieces in whole levels (it was 3 to 8) |

Lifts and blocks invisible until hit came later (below).

**Hammer Bros** (2026-11-12, from World 3-1's memory: 116 throws, three
jumps, the walk). Block SMB's Hammer Bro has the game's 8 by 24 box (4 in
from each side of its 16 by 24 drawing), walks 8 pixels either way at a
quarter pixel a frame, jumps every 215 or 241 frames up to the floor 64
pixels above (starting 6.25 pixels a frame upward, through the row over
it) or down through what it stands on (3.25 upward), under 0.25 a frame,
and throws a hammer every 49 or 50 frames, now and then one left out. A
hammer is held 13 frames over its head, then thrown toward Mario at 1
pixel a frame, starting 2 upward and falling a sixteenth of a pixel a
frame faster each frame, through everything (in the game: 1 pixel a frame,
gravity 0.0625 within a thousandth, every throw toward Mario). The
generator has a piece of them (3-1's two rows of bricks 4 and 8 tiles up,
a Bro under them and perhaps one on the lower row) and a skill scene of
it. The planner predicts a hammer as a "thrown" thing (falling, through
solids); the game-memory state reads hammers from the game's loose-item
slots, and vision's already tells them (their own type). Never trained
on, the hand-written chooser passes 107 of 200 Hammer Bros scenes and
round 2's network 125.

**Can every world be passed at all?** `sim/passable.py` searches button
presses for a way through a world with its enemies taken out, knowing
nothing of the agent. All 3,000 skill scenes and all 100 whole levels
checked can be passed, so what the agent fails in them is the policy's to
learn, not an arrangement no player could pass.

**The exam on the widened worlds**, before any learning in them (100 random
levels and 100 scenes of each of the ten kinds, both ways to play; counts
are plays):

| | Random levels | Skill scenes | Weakest kind of scene |
|---|---|---|---|
| The hand-written chooser (the yardstick) | 83 of 200 | 1,868 of 2,000 | bridge, 105 of 200 |
| The old network `outcome_20261109d` | 151 of 200 | 1,962 of 2,000 | bridge, 173 of 200 |
| Round 1 | 153 of 200 | 1,987 of 2,000 | two pieces close together, 191 of 200 |

Two things show. The difficulty the real levels had is now in Block SMB's
own random worlds: whole levels fell from 185 of 200 to 153 for the same
network. And the learned chooser carries over to pieces neither it nor the
hand-written one had met far better than the hand-written one does (153
against 83).

**Round 2** (`outcome_20261111m`, 2026-11-11): 2.55 million choices tried in
600 plays of widened random levels and 4,000 skill scenes, round 1's
network playing; trained from nothing on rounds 1 and 2 together (and on
the 0.5 million choices of the stopped first attempt).

| The Block SMB exam on the widened worlds (plays) | Random levels | Skill scenes | Weakest kind of scene |
|---|---|---|---|
| Round 1 | 153 of 200 | 1,987 of 2,000 | two pieces close together, 191 of 200 |
| Round 2 | 170 of 200 | 1,993 of 2,000 | two pieces close together, 193 of 200 |

Bridges are passed every time (200 of 200, from 196). Of the 30 level
plays still failed, 28 are deaths, most where a pit stands by another piece
with enemies about. On outcomes of fresh worlds the network ranks a choice
that kills above one that does not 98% of the time.

Scored on the real game (nothing learned there, nothing looked into):

| | 1-1 from the screen: obstacles | 1-1 from the screen: whole plays | Held-out levels from the screen | Held-out, exact positions | Held-out whole plays from the screen |
|---|---|---|---|---|---|
| The old network | 0.95 | 4 of 8 | 0.59 | 0.74 | 1 of 32 |
| Round 1 | 0.97 | 6 of 8 | 0.59 | 0.68 | 1 of 32 |
| Round 2 | 0.88 | 4 of 8 | 0.66 | 0.65 | 1 of 32 |

On the building levels from the screen: 2-1 0.77 (round 1: 0.76), 3-1 0.51
(0.56), 5-1 0.86 (0.86), where 4 of 8 whole plays now finish (2 of 8).

What it says: learning in the widened worlds raised the score on levels
nobody looked at, from the screen, for the first time (41% of tries at an
obstacle failing, now 34%; a working agent about 2%), and lowered it on
1-1 (3% failing, now 12%). The score with exact positions has not risen
with it and is now no better than from the screen. Three trainings have
given 4, 6 and 4 whole plays of 1-1 out of 8: a retraining does not yet
pass 1-1 reliably.

**Round 3** (`outcome_20261112n`, 2026-11-12): 1.38 million more choices
in the widened worlds, round 2's network playing; trained on rounds 1 to 3.
In Block SMB no better than round 2 (165 of 200 level plays, 1,990 of
2,000 scenes; round 2: 170 and 1,993), in the real game the best so far:

| Scored on the real game | Round 1 | Round 2 | Round 3 |
|---|---|---|---|
| Tries at an obstacle failing, held-out levels, from the screen | 41% | 34% | 29% |
| The same with exact positions | 32% | 35% | 23% |
| Whole held-out levels finished from the screen | 1 of 32 | 1 of 32 | 1 of 32 |
| 1-1 whole plays finished, from the screen / exact positions | 6 / 0 of 8 | 4 / 4 of 8 | 6 / 8 of 8 |
| 4-1 whole plays finished with exact positions | 1 of 8 | 0 of 8 | 6 of 8 |

**From nothing, in one run** (`outcome_scratch_20261112`, 2026-11-12):
`scripts/learn_in_block_smb.sh` with the larger network below, three rounds
of sampling, training and the exam in today's Block SMB (Hammer Bros among
the pieces), nothing carried over from earlier networks; the choices
without the care variants (the baseline the care run is compared with).
About 7.7 hours. In Block SMB: 159 of 200 level plays finished, 2,170 of
2,200 skill scenes passed, 174 of 200 Hammer Bros scenes (round 2: 148).
On the real game, against round 3 (built up over three runs):

| Scored on the real game | Round 3 | From nothing |
|---|---|---|
| Tries at an obstacle failing, held-out levels, from the screen | 29% | 36% |
| The same with exact positions | 23% | 25% |
| Whole held-out levels finished from the screen | 1 of 32 | 2 of 32 |
| 1-1 whole plays finished, from the screen / exact positions | 6 / 8 of 8 | 4 / 8 of 8 |
| 4-1 whole plays finished with exact positions | 6 of 8 | 2 of 8 |

One run from nothing comes near three rounds built up, with exact
positions; from the screen it loses more (36% against 29%), and it passes
1-1 from the screen in half its plays. Not yet a system that passes 1-1
reliably.

**From nothing with the care variants** (`outcome_care_20261112`,
2026-11-13): the same run with each choice in its three variants (bold,
usual, careful: the controller's berth around enemies, how long a landing
must stay clear, how soon a coming enemy ends a run), the outcome network
picking among them. In Block SMB: 152 of 200 level plays finished (round
2 had 162; the last round's network is not always the best), 2,152 of
2,200 skill scenes, 175 of 200 Hammer Bros scenes. On the real game:

| Scored on the real game | From nothing | With the care variants |
|---|---|---|
| Tries at an obstacle failing, held-out levels, from the screen | 36% | 38% |
| The same with exact positions | 25% | 26% |
| Whole held-out levels finished from the screen | 2 of 32 | 1 of 32 |
| 1-1 whole plays finished, from the screen / exact positions | 4 / 8 of 8 | 7 / 8 of 8 |
| 4-1 whole plays finished with exact positions | 2 of 8 | 0 of 8 |

Obstacle by obstacle with exact positions it does better on 1-1, 2-1, 3-1
and 6-1 and worse on 4-1, 5-1 and 8-1. Taken together the care variants
change nothing measurable yet: the difference between the two runs is
within what one training to the next gives. The three sets are still
written by hand; learning the settings themselves is the next step for
the controller.

**From nothing in the widened Block SMB** (`outcome_widened_20261113`,
2026-11-14): the same command, care variants included, after Block SMB
gained everything the levels not scored on hold except water (red Koopas,
Lakitu, flying Paratroopas, frenzies, lifts of every kind, springboards,
fire bars, lava, Bowser with his flames and hammers, blocks invisible until
hit). About 18.7 hours: the widened levels are longer, so round 1's
sampling alone took 5.7 hours (9.1 million choices tried). Examined on the
same 100 random levels and 4,200 skill scenes, none trained on:

| Round | Speed runs finished | Points runs finished | Skill scenes passed | Hammer Bros scenes |
|---|---|---|---|---|
| 1 | 60 of 100 | 54 of 100 | 4,076 of 4,200 | 166 of 200 |
| 2 | 66 of 100 | 67 of 100 | 4,092 of 4,200 | 166 of 200 |
| 3 | 66 of 100 | 68 of 100 | 4,106 of 4,200 | 171 of 200 |

The hand-written chooser finishes 7 of 30 speed runs on such levels. The
scenes still failed most are Hammer Bros (171 of 200), two obstacles
together (186), ledges (187), looping lifts (187) and bridges (192); almost
every level not finished ends in a death. Scored on the real game
(score only), against the care run:

| Scored on the real game | With the care variants | Widened Block SMB |
|---|---|---|
| Tries at an obstacle failing, held-out levels, from the screen | 38% | 46% |
| The same with exact positions | 26% | 25% |
| Whole plays finished with exact positions, held-out levels | 2 of 32 | 15 of 32 |
| Whole plays finished with exact positions, all eight levels | 11 of 64 | 33 of 64 |
| Whole plays finished from the screen, all eight levels | 9 of 64 | 5 of 64 |
| 1-1 whole plays finished, from the screen / exact positions | 7 / 8 of 8 | 5 / 6 of 8 |

With exact positions the agent now finishes whole levels it never
finished before: 4-1 and 6-1 in 7 of 8 plays, 5-1 in 5, 2-1 in 4, 3-1
in 3 (8-1 in none). Choosing and controlling learned in the widened
Block SMB carry over to the real game's levels far better. From the
screen it does worse than before (46% of tries failing against 38%):
what the vision network reports is not what the network learned from,
and that gap is now the larger loss. By the rule, that is seeing's
problem, worked on with seeing's own data.

**A larger network learns more of the same data.** Trained on rounds 1 and
2's outcomes, a network twice as wide (192 numbers a token), one layer
deeper (4) and trained twice as long (40,000 steps) against round 2's
(96, 3, 20,000), both examined in today's Block SMB (Hammer Bros among the
random levels' pieces, met by neither in training):

| | Random levels | Skill scenes | Hammer Bros scenes |
|---|---|---|---|
| Round 2's size | 140 of 200 | 2,115 of 2,200 | 125 of 200 |
| The larger network | 162 of 200 | 2,138 of 2,200 | 149 of 200 |

The from-nothing run (`scripts/learn_in_block_smb.sh`) is made with the
larger network.

**What still fails in Block SMB** (round 2's 30 failed level plays of
200, each played again in Block SMB, 2026-11-12):

| What happened | Plays |
|---|---|
| Trapped: the choice before, rated safe, left Mario where every choice was rated likely to kill (Goombas coming down a staircase onto him, enemies on both sides); the least bad, running back, killed him | 14 |
| A pass or a landing rated under 3% likely to kill ended in a pit | 8 |
| Waiting or landing by a plant | 5 |
| Out of time | 2 |

The outcome network already has the answer for a trap: a choice after
which no choice gets Mario on safely is "stuck", and the chooser avoids
it. On these fresh levels it did not see it. Every group is the network's
prediction wrong on a world it had not met, which more data and a larger
network address, not a change by hand.

**Tried and not kept: the controller's enemy rules made to match Block SMB.**
Measured in Block SMB alone, the rules the controller plans with put a
falling Goomba within 4 pixels of where Block SMB put it 16 frames later
only 39% of the time (a walker already in the air was kept at its speed,
without gravity or landing) and a hopping Paratroopa 86% (it hopped
before it had landed, and rose through brick rows). Corrected, 100% and
99%. Block SMB play did not gain (round 2's network: 166 of 200 level
plays, 170 before; the hand-written chooser 81, 83 before), so the
correction is not kept. Seen on the way, measured in the game's memory on
the building levels: the game's hopping Paratroopa hops as Block SMB's
does (about 44 pixels up every 56 frames; Block SMB's 44 every 54), and in
the game's state an enemy often hides the ground under it (a fifth of the
still Goombas had no solid under them), which Block SMB's state never
does.

**One command** learns the deciding network from nothing in Block SMB,
round after round (`scripts/learn_in_block_smb.sh NAME`): sample, train,
take the exam, and let the new network's play visit the next round's
situations. Seeing is trained apart, on real frames. A full run from
nothing was made on 2026-11-12 (above: "From nothing, in one run"). A sampled play is now given up after 900 frames
without progress (`framework/sampling.py`): in the widened worlds the
hand-written chooser, whose play the first round samples, stalls in a
third of the levels, and a stalled play took over four minutes to sample.

Scored on the real game (nothing learned there), obstacles passed and whole
levels finished:

| | 1-1, from the screen | 1-1, whole level from the screen | Held-out levels, from the screen | Held-out, exact positions |
|---|---|---|---|---|
| The old network | 0.95 | 4 of 8 plays | 0.59 | 0.74 |
| Round 1 | 0.97 | 6 of 8 plays | 0.59 | 0.68 |

## Every kind of thing the game has (2026-11-13)

Block SMB had been measured in four levels. To find what else the game
holds, `scripts/dev/game_catalogue.py` flies Mario through every level not
scored on (27 of 32: 1-1, 4-1, 6-1, 7-1 and 8-1 are not read) and reads
the game's memory as he goes. He gets to a level by writing its numbers
into memory and letting the timer run out, is held near the top of the
screen, made invincible and pushed through walls. Every frame it records
which kind of thing is in each of the game's enemy slots (what the level
lists place and what the game makes as it goes: Cheep Cheeps, a Lakitu's
Spinies, frenzies), the level's blocks and its area type. The three
castles whose corridors loop (4-4, 7-4, 8-4) are not flown to their ends.
`scripts/dev/measure_enemy.py` then reads how one kind moves: Mario is
held still or at a set speed, unhurtable, or set down to ride a lift, and
the thing's place, speed and box are read every frame.
`scripts/dev/level_shapes.py` measures the levels' shapes from the block
maps.

The game has 34 kinds of things in those levels; Block SMB had 6. Added,
each measured in a level not scored on:

| Kind | Measured in | What Block SMB now does |
|---|---|---|
| Buzzy Beetle | 4-2, 5-2, 6-2 | Walks at half a pixel a frame and leaves a shell, as a Koopa: to an agent without fireballs it is one |
| Red Koopa | 1-3, 3-3 | A Koopa turned back where the floor under its middle would end; red Koopas in enemy files and on brick rows and ledges. A new kind for the planner (a pacer) |
| Lakitu and Spinies | 8-2 | Flies 40 from the top at a base speed less a quarter of his distance from Mario (base 20, 47 or 63 by how Mario scrolls the screen), turns back 60 past him, throws an egg every 128 frames while there is room, leaves at the end of his stretch; an egg rises at 3.06, falls under 0.125 and hatches into a Spiny walking at Mario, which cannot be stomped. A new kind for the planner (an egg) |
| Flying Paratroopas | 1-3 (red), 7-3 (green) | The red one swings 104 down and back every 400 frames, the green one 95 across every 320, bobbing 8 (sines, 1 to 3 pixels off the game's); stomped, a walking Koopa of its colour |
| Bullet Bill frenzy | 5-3 | A Bill from the right every 240 frames at three heights in turn |
| Leaping Cheep Cheeps | 2-3 | From below near Mario, up at 4.97 under 0.052, across at one of four speeds; over some bridges. A new kind for the planner (a leaper) |
| Lifts | 1-3, 3-3 (Mario riding) | 48 wide; across (53 pixels, 240 frames), up and down (137, 340), falling while stood on (2.25 a frame); they carry Mario both ways |
| Springboards | 2-1 (Mario set down on it, and walked into it) | On a half block that stops a walking Mario (one solid with it in Block SMB), its top 20 over the block. Standing on it presses him down 20 and lets him up over 19 frames, then throws him up at 7 pixels a frame, or at 12 if the jump button is held on the 16th to 18th of those frames (the game takes no other press there); gravity in the throw is 40/256. The plain throw is within a pixel of the game's all the way up; the strong one goes 457 up, as the game's. A piece: one at the floor's end before a wall 10 tiles tall (2-1) or a pit (3-1, 5-2, 8-2). The planner throws Mario over the wall from it (standing still to the throw, then steering), and the simulator lands him where it planned |
| Fire bars | 1-4, 5-4 (the bar's turn from memory, its fireballs as drawn) | Six fireballs (twelve on a long bar) 8 apart out from the middle of a block, turning a 32-step circle once in 205 frames (5-4's fast one in 146: the game's turns, 40/256 and 56/256 of a step a frame, read from its program), the fireballs at the whole step reached (within a pixel of where the game draws them); they hurt, cannot be stomped, and shells pass them. A piece: one to three bars on blocks at the heights of the game's 47 (in the floor, on it, or 2, 3, 4, 6 or 8 tiles over it), a quarter turning clockwise, one in fifty fast (the game's one of 46). A new kind for the planner (a turner: it goes round its hub at its turn, which the state gives with it). The hand-written chooser passes 58 of 80 fire bar scenes |
| Lava fireballs | 2-4 | Jumps out of its pit from under the screen at 5.95 to 6.35 pixels a frame (peaking 70 to 92 from the top: the game's two measured leaps peaked at 73 and 90), under 28/256, falling at most 3.25, and jumps again 21 times 6 to 15 frames later; the game's 10 by 6 box; it hurts and cannot be stomped. A piece: one to three lava pits 1 to 3 tiles wide (the castles'), a fireball in each; one under a fifth of the bridges. A new kind for the planner (a fountain). The hand-written chooser passes 74 of 80 lava scenes |
| Bowser and his flames | 1-4, 6-4 (his place, his flames', the lift over him), and his routine in the game's program | He paces about where he starts at a quarter of a pixel a frame (two at first and while following), turning back 17, 33, 49 or 65 either way (drawn each time he passes his start); he jumps 17, 33, 49 or 65 frames after landing, 2 a frame up under 15/256, 37 pixels high and 70 frames in the air (takeoff to takeoff 86, 102, 118 or 134 frames, the game's); 223 frames after he comes on screen his mouth opens for 32 frames (he stands still) and as it shuts he breathes a flame, then again after 191 or 64 frames in the game's order (16 fewer, and faster flames, in the hard castles of worlds 5 to 8); once Mario is behind him he turns, follows at half a pixel a frame and stops 89 right of his start, neither jumping nor breathing. His two halves are one box here, 32 by 27; he cannot be stomped. A flame flies left at 1.25 (1.375) and a pixel a frame up or down until level with 112, 128 or 144 from the top of a castle (Block SMB: 160, 176, 192), its box the game's 4 by 4. Free flames come from the right before he is on screen (36 to 42 tiles before his bridge, every 223 or 96 frames, level at once) and stop when he comes. A piece: the castles' bridge, 13 tiles level with the floor over lava, Bowser 7 tiles in, a lift 32 wide swinging across 4 tiles over it (the game's, 53 pixels every 240 frames, in all five castles read), a lava fireball under it two times in five, and the axe's step after it. Two new kinds for the planner: a jumper (walking; in the air, falling to a top: when he will jump is not known) and a flame (to its level, which the state gives with it). Every Bowser scene checked can be passed; the hand-written chooser passes 56 of 80 (Bowser coming down on Mario as he runs under is most of the rest). Not here: the axe |
| Bowser's hammers | 6-4 (the game's nine loose-item places read every frame) and his routine in the game's program | In worlds 6 to 8, while he is in the air, once in 4 frames he tries to throw: one of nine places is drawn (two of them half as often as the others), and if no hammer of his is in it and its place is on screen, a hammer appears 2 right of his place and 10 up. It is held there 15 frames, moving with him, then thrown the way he faces, 1 pixel a frame sideways and 2 up, falling as a Hammer Bro's does. Its hurt box is the game's for a hammer, 8 by 8 from its place. In 6-4 he threw 6 to 9 a jump (Mario was mostly behind him, so he jumped less often); the same draw with as many jumps gives 7.5, and in front of Mario about 6. The castle a Bowser piece stands for is now one of the game's eight, each as often: from world 5 the hard castles' flames, from world 6 hammers, in worlds 6 and 7 no flames of his own. Every hammer castle checked can be passed; the hand-written chooser passes 17 of 30 (31 of 50 castles without hammers) |
| Looping lifts | 1-2 and 2-4 (each lift's place every frame) | Lifts going up or down 15/16 of a pixel a frame without end (still one frame in sixteen), coming round from the other end of the game's 256 rows every 273 frames; they carry Mario, but not round. The large ones are 48 wide (1-2, 4-2), the castles' short ones 24 (2-4, 5-4): the game's boxes. A piece: a bottomless pit 6 or 7 tiles wide with a column of one or two lifts (96 to 128 pixels apart) all going up or all down, 12 to 28 pixels in from its near edge; or, a third of the time, a castle's 8-tile pit with a short lift going up and one going down. Every scene checked can be passed; the hand-written chooser passes 28 of 40 |
| Balance lifts | 3-3 (Mario riding one), their routine in the game's program, and the nine pairs of 3-3, 4-3 and 6-3 | Two lifts on one rope: the one Mario stands on sinks and the other rises as much, 5/256 of a pixel a frame faster each frame to at most 3 (the game's: 82 pixels in the first 91 frames); left alone they slow by as much to a stop. A lift that rises to 45 from the screen's top stops there, unless Mario is on the other: then the rope breaks and both fall, an eighth of a pixel a frame faster each frame to 3 (92 frames after Mario stepped on here, about 91 in the game). A piece: a bottomless pit with the higher lift (its top 78 or 94 from the screen's top) and the lower one (126 or 142) 48 to 128 to its right; Mario comes to the higher lift as in the game, from about its height: a ledge 48 or 64 below it, 16 to 48 before it (with a way up when it is taller than a jump). Every scene checked can be passed; the hand-written chooser passes 39 of 40. The game's 1000 points for breaking the rope are not counted |
| Blocks invisible until hit | 2-1, 3-1, 4-2, 5-1, 5-2, 6-2 and 1-4's last room (the block maps), and the game's program | Until Mario's head hits one from below it is not seen and nothing else meets it: he falls and walks through it, enemies and mushrooms too (the game's code leaves them out of every other test). Hit, it shows as a used block, solid all round, and gives its coin or lets out a 1-up (worth nothing here: Block SMB has one life). A piece, each way as often as the game has it: one alone 4 tiles up (a 1-up one time in five), a 1-up 7 tiles up over a brick or a hidden coin, four side by side 4 to 6 up (4-2), or 1-4's six in a zigzag. The state does not show it until it is hit, as vision cannot. Every scene checked can be passed; the hand-written chooser passes 40 of 40 |

With the castles and the lifts, a random level is much harder than it was:
the hand-written chooser finishes 7 of 30 as speed runs (seeds 0 to 29).
The search that tells whether a world can be passed counts a lift as part
of a situation only within 128 pixels of Mario: a lift going round without
end would otherwise make every later situation new, and long levels ran
out of tries.

The levels' shapes, from the block maps:

| In the game | Block SMB before | Now |
|---|---|---|
| Sky levels (1-3, 3-3, 4-3, 5-3, 6-3): stretches of 24 to 51 tiles with no floor, crossed on 16 to 20 ledges each, 1 to 12 tiles wide, 1 to 10 up, 0 to 8 apart | None | A piece of 3 to 8 ledges over a stretch with no floor, of the game's widths, heights and gaps, each within a jump of the one before; red Koopas on some, red Paratroopas and lifts in some gaps |
| Pits 1 tile wide (25, 14 of them in 8-2) | 2 to 4 | 1 to 4 |
| Brick rows 3 tiles up (36) | 4 or 5 | 3 to 5 |
| Bridges over pits of 8 to 15 (2-3, 7-3) | to 8 | to 15 |

The planner now takes each solid's box frame by frame: a lift moves at
its speed and carries Mario standing on it; still solids plan exactly as
before. From a springboard it plays the squash and the throw (looking 160
frames ahead, not 96: a throw comes down to the floor's height in about
120). A jump none of its plans meets is tried once more pressed toward
the goal for a while and then released (dropping from the top of 2-1's
wall with speed, every jump pressed on overshot the floor in sight). Every
new skill scene checked can be passed (the passability search counts
where the lifts are, and a springboard's point of its squash, as part of a
situation). The hand-written chooser, never trained on them, passes in 80
plays of each kind (40 scenes, both ways of playing): 62 Lakitu scenes,
44 ledge scenes, 78 springboard scenes; the released jumps raised the
bridges from 52 to 67, and the scenes both Block SMBs have are otherwise
the same as before.

Still missing: in the castles, the game's own test of a fire bar's touch (Block SMB
takes the fireballs' drawn 8 by 8); water levels (swimming, Bloopers,
swimming Cheep Cheeps). In the real game, the
state from memory and from vision show lifts without their speed, and
show no springboard, no fire bar going round, no lava fireball and no
Bowser. None of the networks has learned from the widened Block SMB yet:
that is the next run from nothing.

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

### What seeing cost the widened network, and the fall that never stopped (2026-11-15)

The network learned in the widened Block SMB (`outcome_widened_20261113`)
passed 0.74 of tries at obstacles on the six untouched levels with exact
positions and only 0.53 from the screen. The same bench played from the
screen with one part of the state exact at a time
(`scripts/dev/perception_parts.py`):

| Exact in place of seen | Untouched levels |
|---|---|
| Nothing (vision) | 0.53 |
| Where Mario's body is | 0.54 |
| Whether he stands | 0.59 |
| How he moves | 0.66 |
| All of Mario | 0.78 |
| The enemies | 0.60 |
| The ground, pipes and blocks | 0.60 |
| Mario, enemies and ground | 0.76 |

Nearly all the loss was in Mario's own motion. Compared field by field
with the game's (`scripts/dev/motion_check.py`), vision's motion model
held Mario's vertical speed at 4 pixels a frame, the fastest fall, in
most frames he stood: the model's own movement rules never stop a fall,
the game's landing does, and Block SMB copies the landing every frame he
stands. The outcome network reads his vertical speed, so from the screen
a standing Mario looked to it like one falling fast, in every situation
it decided in. Vision's estimate now stops the fall when Mario is seen
standing and the model has him falling or level (a jump just started is
kept: the screen shows the takeoff a frame late). With the same network:

| Share of tries passed, from the screen | Before | Fall stopped | Exact positions |
|---|---|---|---|
| 1-1 and 5-1 (tuned) | 0.77 | 0.79 | 0.98 |
| The six untouched levels | 0.53 | 0.71 | 0.74 |

1-1 went from 0.82 to 0.96, 4-1 from 0.46 to 0.67, 7-1 from 0.50 to 0.81.
5-1 fell from 0.73 to 0.67: one start, where Mario jumps into a Piranha
Plant rising from the next pipe, is shared by nine of the bench's
obstacles and counted in each.

The next error the field-by-field check showed: run into a wall, Mario
is stopped by the game (speed 0, pushed back a pixel a frame) while
vision's model ran on at full speed, because each change in his running
sprite's width threw away its speed samples before they could correct
it. The estimate now stops him as the game's wall contact does when, over
four frames, his leading edge has moved on at most a pixel while the
model has him running, and a seen solid's side is just ahead (without
that condition a scroll measured wrong stopped him in open running, in 8
of 46 stops on 2-1 and 7-1; with it, in none). On the ground of 6-1 his
speed now agrees with the game's in 1,636 frames of about 1,720 (1,593
before). In play it changes little: 0.82 of tries passed on the tuned
levels (0.79) and 0.70 on the untouched ones (0.71), within what one
bench to the next varies.

Scored again with both corrections (same network, score only):

| On the real game | Before | Both corrections | Exact positions |
|---|---|---|---|
| Tries at an obstacle failing, held-out levels, from the screen | 46% | 24% | 25% |
| Whole held-out levels finished from the screen | 0 of 32 | 3 of 32 | 15 of 32 |
| Whole plays finished from the screen, all eight levels | 5 of 64 | 8 of 64 | 33 of 64 |
| 1-1 whole plays finished from the screen | 5 of 8 | 4 of 8 | 6 of 8 |

Obstacle by obstacle, seeing now costs the agent almost nothing on the
held-out levels. Whole levels still finish far less often from the
screen than with exact positions: a level is about 35 obstacles in a row,
and what vision gets wrong now is rarer but still enough to end most
plays somewhere.

## What ends plays in Block SMB itself (2026-11-16)

Since the corrections to vision, an obstacle tried from the screen fails
about as often as with exact positions; what is left is in choosing and
controlling, and those are worked on in Block SMB only (requirements.md,
section 3). The widened network (`outcome_widened_20261113`) finished 134
of 200 plays of the exam's 100 fresh random levels (both modes) and died in
65. What killed Mario (`scripts/dev/block_smb_deaths.py`): pits 24, Hammer
Bros and their hammers 13, fire bars, lava, Spinies, Bullet Bills and the
rest fewer each; in 44 of the 65 deaths the network had given the fatal
choice less than a 5% chance of death.

**The jump planner against Block SMB.** Every jump in those plays, the
plan in force as Mario left the ground against what then happened
(`scripts/dev/jump_check.py`): of 5,832 jumps, 91.5% landed where planned,
4.1% elsewhere on the planned solid (some by a correction in the air as an
enemy moved), 3.1% on another solid, and 37 (0.6%) died where a landing had
been planned; at about 29 jumps a level, those alone end one level in six.
The landings planned with only a pixel or three of Mario's 12-pixel width on
the solid failed to happen as planned 37% of the time (2.4% with all of him
on it): a pixel of drift in the air, a wall's corner, a lift. The planner
now prefers a **firm** landing, with at least 4 pixels of him on the solid
(all of a narrower one), at takeoff and among its corrections in the air
(planner.FOOTING).

**A one-tile pit, as the game has it.** Runs ended in pits too: Mario ran
at a one-tile pit, dipped a pixel over it, met the far ledge's side and was
stopped by it as by a wall, and fell. Measured in the game (World 1-1 with
one column of its floor blocks cleared in the game's memory,
`scripts/dev/gap_rule.py`): walking or running, Mario crosses a one-tile pit;
the game sets him on top of a block he meets with his feet up to 3 pixels
below its top, and stops him at 4; stopped over the pit, he falls in; a
two-tile pit he falls into. Block SMB stopped him at any overlap, the
planner at 3 pixels. Both now follow the game (sim/env.py STEP_UP, for
blocks; lifts, the game's objects, were not measured so and keep their
sides; planner.WALL_DEPTH).

With the same network, unchanged:

| Fresh Block SMB levels (200 plays) | Before | Firm landings | And the game's step onto a ledge |
|---|---|---|---|
| Finished | 134 | 141 | 144 |
| Died | 65 | 59 | 56 |
| ... in a pit | 24 | 22 | 13 |
| Jumps dying where a landing was planned | 37 | 33 | 32 |

**Hammers still held.** Of the 15 deaths by Hammer Bros, 6 were hammers,
and 4 of those had been in sight when the fatal jump was chosen, most still
held over the Bro's head: the planner took a held hammer for one already
thrown and foresaw it falling from there. It now tells a held one by what
is seen (it has moved only with the Bro, never up or down) and foresees it
kept there until the game's 14 frames of hold are up, then thrown toward
Mario (planner.held): 32 of 36 held hammers in Hammer Bro scenes are then
foreseen to the pixel for 60 frames. It changes little by itself (144
levels finished; Hammer Bro scenes 169 of 200 passed against 171): most of
those deaths are Mario touching the Bro himself, as it jumps up to a ledge
he stands on or as a run starts beside it, and when a Bro will next jump is
not seen.

Scored on the real game with these corrections (score only; the network
unchanged), World 1-1's four starts in each of the two modes finish in 8 of
8 plays with exact positions (6 of 8 before) and in 5 of 8 from the screen
(4 of 8 before).

**Things falling through the air.** The planner kept a walker already in
the air (a Paratroopa stomped out of the air, a Goomba, a Koopa or a Spiny
falling, a Hammer Bro jumping) at the speed it was falling at, and planned a
second bounce off a stomped Paratroopa that never came. It now lets one fall
as the game's do, faster each frame to the same limit as Block SMB, onto
the first top under it (planner.enemy_paths). Measured against Block SMB in
25 fresh levels, the place it foresees for a falling Goomba, Koopa or Spiny
16 frames on is now at the pixel at the median (12 to 16 pixels off before);
a Hammer Bro's still drifts (7 pixels at 16 frames, 22 before: it falls a little
faster, and when it jumps is not seen). On 400 plays of 200 fresh levels the
network unchanged finishes 286 against 280.

**Lifts, as the game has them.** A lift coming down onto a rising Mario
pushed him aside in Block SMB (13 pixels, off the jump), and one he fell
into from below, deeper than its top, set him on it. Measured in the game
(World 1-3's lift going up and down, Mario put beside, under and over it in
memory, `scripts/dev/lift_contact.py`): rising into its underside, he falls
from where he is, neither pushed down nor aside, his run kept; falling, he
is set on it only with his feet at most 4 pixels into its top, and passes
through it deeper; met from beside, it stops him as a wall. Block SMB and
the planner now follow (sim/env.py and planner.py LIFT_*; met from beside near its top, as measured
afterwards: below). With both, the
same 400 plays finish 284 (286 with the first alone: no change beyond what
a different course through a level gives); the network learned with the
old lifts.

## Learned again in the corrected Block SMB (2026-11-17)

The deciding network learned from nothing again (`scripts/learn_in_block_smb.sh
corrected_20261116`, three rounds, 7.8 hours) in Block SMB with the firm
landings, the game's step onto a ledge and held hammers above, but not yet
the falling and the lifts, which came while it ran. The network is kept as
`data/outcome/outcome_corrected_20261116.pt`. The exam after each round
(100 fresh random levels, both modes, and 4,200 skill scenes):

| Round | Fresh levels finished (of 200) | Skill scenes passed | The widened run's round |
|---|---|---|---|
| 1 | 114 | 96.9% | 114 |
| 2 | 142 | 98.2% | 133 |
| 3 | 142 | 97.6% | 134 (144 with the corrected controller) |

Both networks played with the present code: in Block SMB, 400 plays of 200
fresh levels (`scripts/dev/block_smb_deaths.py`, first seeds 100000 and
100100), and on the real game (score only), every bench in both ways to
play and every level whole, from four starts in each of the two modes:

| | The widened network | Learned again |
|---|---|---|
| Fresh Block SMB levels finished (of 400 plays) | 284 | 291 |
| ... died in a pit | 32 | 38 |
| ... by Hammer Bros and their hammers | 23 | 26 |
| ... in lava or by fire bars | 22 | 16 |
| Held-out real levels: tries at an obstacle that fail, exact positions | 26% | 18% |
| ... from the screen | 27% | 30% |
| Held-out real levels finished, exact positions (of 32) | 10 | 2 |
| ... from the screen | 2 | 4 |
| World 1-1 finished (of 8), exact positions / from the screen | 8 / 7 | 8 / 7 |
| All eight scored levels finished (of 64), exact positions / from the screen | 27 / 9 | 16 / 12 |

Per level (share of tries at an obstacle that fail, and whole plays
finished of 8; W the widened network, A the one learned again):

| Level | Exact positions, W | A | From the screen, W | A | Whole, exact, W | A | Whole, screen, W | A |
|---|---|---|---|---|---|---|---|---|
| 1-1 | 2% | 5% | 6% | 8% | 8 | 8 | 7 | 7 |
| 5-1 | 3% | 4% | 29% | 31% | 2 | 6 | 0 | 1 |
| 2-1 | 16% | 40% | 35% | 35% | 7 | 0 | 0 | 0 |
| 3-1 | 29% | 31% | 42% | 44% | 0 | 0 | 0 | 0 |
| 4-1 | 15% | 15% | 36% | 18% | 3 | 2 | 2 | 2 |
| 6-1 | 50% | 28% | 32% | 48% | 5 | 0 | 0 | 2 |
| 7-1 | 32% | 28% | 37% | 34% | 2 | 0 | 0 | 0 |
| 8-1 | 13% | 6% | 12% | 24% | 0 | 0 | 0 | 0 |

Learning again in the corrected Block SMB made no clear difference. In
Block SMB it finishes 7 more plays of 400, about what a different course
through the levels gives. On the real game the two differ level by level
in both directions. A whole play is deterministic and the four starts
differ by a few frames only, so each level's eight whole plays are close
to two samples, one for each mode: one change of course swings a level
from 7 of 8 to 0. The benches, which try each obstacle from its own
start, are the finer measure. With exact positions the network learned
again fails 18% of its tries on the held-out levels, against 26%; from
the screen it fails 30%, against 27%. What seeing costs it is that
difference, largest on 6-1 and 8-1, and is worked on with seeing's own
data. Its deaths in Block SMB are those of before: pits first, then Hammer
Bros, and in 73 of its 108 deaths the network had given the fatal choice
less than a 5% chance of death.

### Pits, traced again (2026-11-17)

Pits were still the first killer of the network learned again (38 of 109
deaths in the 400 plays). Three causes were found by replaying them, each
where the planner or the controller and what Mario then did came apart.
With all three, the network unchanged finishes 588 of 800 plays (from first
seeds 100000 and 100200) against 567 with the lifts' first correction alone,
and 57 of them end in pits against 77.

**Lifts and ledges met from beside.** Jumping up to a balance lift, Mario
reached its side still rising with his feet 2 pixels below its top; Block
SMB stopped him there and he fell past it into the pit, while the plan had
him land on it. Measured in the game (World 1-3's lift, Mario run right into
its side at every depth and vertical speed, `scripts/dev/lift_contact.py
--side`), by where his feet are after the frame's moves of Mario and lift:
from 1 pixel above its top to 4 into it, falling or level, he is set on it;
rising with them there or deeper, or falling with them deeper than 4, his
run is stopped; higher, he passes clear. Block SMB stopped him at any depth
(wrong when falling); the planner let him through whenever his feet were
within 3 pixels of the top (wrong when rising), and when falling let him
through a lift's side, or a block's within its 3-pixel step, and on down to
the floor under it. Both now follow the game, and from the same standing
starts the planner foresees every jump at a raised block or lift where
Block SMB lands him (`scripts/tests/test_lifts.py`). The same 400 plays:
295 finished against 291, 32 deaths in pits against 38.

**Letting go once down.** An in-flight correction kept Mario clear of a
piranha plant rising out of the pipe he was to land on by landing him on
its last pixel; the planner allowed it because, buttons released, he would
slide off onto the ground beside it. The controller instead kept pressing
on, as it does after a plan's last button until it has seen him stand two
frames, and carried him over that ground into a pit. Now, once Mario has
been down on something after the plan's buttons are done, it lets go, as
the planner foresees. Over 800 plays (the 400 and another 400 from first
seed 100200) it finishes 571 against 567, 72 deaths in pits against 77: no
clear difference, kept because the controller now does what the planner
expects of it.

**A gap before a step up.** A run ended where Mario could still stop, asked
only of the spot where he would stop: beyond the end of the ground there was
a one-tile gap and then a ledge 16 pixels higher, and the run took the
ledge for a wall that would stop him, as a pipe does. He ran off the ground,
met the ledge's side over the gap and fell in. A run now asks of the whole
stretch to where he would stop: a gap he would fall into (as wide as he is
or wider, and not a one-tile gap whose far side is at most a step above his
feet, which he crosses as in the game) is a pit (controller.pit_ahead). Over
the same 800 plays: 588 finished against 571, 57 deaths in pits against 72.

**A springboard springing back.** Thrown up by a springboard with his run
held, Mario was shoved a whole width back off it: the springboard snaps back
to its rest in a frame, into his feet as he rises, and Block SMB took its
side for a wall. It now meets a Mario already over it only by his feet, and
not at all as he rises off it, as the game throws him straight up
(`scripts/tests/test_springboards.py`). On the 800 plays it changes little
(583 finished against 588, 56 deaths in pits against 57); it is kept as a
correction of Block SMB to the game.

**Hammers not yet thrown.** Hammer Bros and their hammers became the next
killer (51 of the 800 plays). Of the deaths by hammers during a jump, nearly
all were by a hammer that did not exist when the jump was chosen: it
appeared over the Bro's head as Mario jumped over or onto it and met him
while still held, 6 to 15 frames old. A Hammer Bro throws on a beat (one
every 49 frames, measured in the game), so the planner now foresees each
one's next hammers from the last one seen of it (its age is how long it has
been in sight), held over its head as the Bro moves and then thrown toward
Mario; one with none of its hammers in sight keeps one over its head all
the while, as a hammer may appear any frame (planner.coming_throws; a
Hammer Bro is told by the state, Block SMB's record or vision's type). The
800 plays: 584 finished against 583, deaths by hammers 25 against 30; the
Hammer Bro scenes: 187 of 200 passed against 182.

**A Paratroopa just stomped.** In a quarter of the deaths by Koopas, Mario
had stomped the Koopa out of its wings the frame before: bounced up but
still touching it, Block SMB had it hurt him. Measured in the game (World
1-3's red Paratroopas, big Mario dropped onto one from a range of heights
and speeds, `scripts/dev/stomp_contact.py`): stomped deep, he still touches
it one or two frames after, and is never hurt. Block SMB now leaves a Koopa
just stomped out of its wings harmless while he still touches it (the
planner already took a stomped enemy for harmless). The 800 plays: 587
finished against 584, 16 deaths by Koopas against 21.

**A choice refused, answered in the same frame.** Mario landed on a
springboard, and the network, wrong about what could be done from it (it
called landing back on the springboard possible and the lands beyond the
wall it serves impossible, the reverse of the planner), picked one landing
after another that the controller could not plan. Each refusal left its
frame idle, the springboard's squash ran out with nothing pressed, and it
threw him up drifting back into the pit behind. The agent now answers a
choice refused as it starts with the chooser's next in the same frame, up to
eight (framework/agent.py; it did so before 2026-11-11, and was put back
then only because it had come from tracing a death in the real game). The
800 plays: 607 finished against 587, 45 deaths in pits against 55.

**A Hammer Bro's own path.** The planner foresaw a Hammer Bro as a walker,
which walls turn back: one dropping through the brick row it had stood on
was turned by the row every frame, foreseen 1.5 pixels from where it fell,
and the jump past it grazed it. A Hammer Bro (Block SMB's, as measured in
the game) meets no walls, stays at its height while it stands, falls under
its own gravity (0.25 a frame, not a walker's 0.2) and passes through the
top it drops from. The planner now gives it that path, the top dropped from
told by its speed: a drop starts with a hop up at 3.25 pixels a frame, so
the top under it where such a hop would have started is passed through
(planner.BRO; the chooser still sees it as a walker, so the network's inputs
are unchanged). Over every flight of a Bro jumping between floors its
foreseen height is Block SMB's exactly (`scripts/tests/test_hammer_bros.py`).
The 800 plays: 610 finished against 607, 13 deaths by Hammer Bros against
26; the Hammer Bro scenes: 185 of 200 passed against 181.

Tried and not kept: a run that asks whether Mario could still stop short of
a pit after this frame's press, not as if released now (a run chosen again
each frame pressed on a frame and braked the next, and slid off a ledge):
574 of the 800 finished, but 90 deaths in pits against 72. Nor: each foreseen
fireball's box kept wider by how far off it may be (half a step of its
bar's turn at its distance from the hub): 593 finished against 587 and the
fire bar scenes 190 of 200 against 186, but as many deaths by fire bars (23
against 21), so not what kills Mario at them. Nor: a jump's re-plans (every
4 frames of its run-up) kept able to carry on the plan they replace. A step
down from a block, pressed 4 frames then let go, could not be re-planned:
walking off was no longer allowed once the plan's takeoff (never, for a
walk-off) was counted down, and letting go at once was not among the
pressings tried (4 to 32 frames); the plan put in its place overshot onto
the last pixels of the next step, and Mario slid into a pit. With walking
off still allowed and every pressing from 0 to 32 frames in fours, 596 of
the 800 finished against 610 (602 with the first alone, 606 with the
second): no better, so neither is kept.

## Learned again with the day's corrections (2026-11-18)

The deciding network learned from nothing once more (`scripts/learn_in_block_smb.sh
sided_20261117`, three rounds, 22 hours with Block SMB tracing at its side)
in Block SMB with the lifts and ledges met from beside, the controller
letting go once down, the gap before a step up, and the springboard
springing back. Its rounds 2 and 3 sampled with the code merged during the
run too (a Hammer Bro's hammers and its path foreseen, the Paratroopa just
stomped, a refused choice answered in the same frame): what it learned from
is two versions' play. The exam after each round (100 fresh random levels,
both modes, and 4,200 skill scenes):

| Round | Fresh levels finished (of 200) | Skill scenes passed |
|---|---|---|
| 1 | 156 | 97.7% |
| 2 | 161 | 98.1% |
| 3 | 152 | 98.0% |

The third round is kept, as `data/outcome/outcome_sided_20261117.pt`, by
Block SMB alone: on 800 fresh plays (first seeds 100000 and 100200) with the
present code it finishes 618 against the earlier network's 610 (the second
round, measured with the re-plan change tried above, finished 599 where the
third finished 617 and the earlier network 596). Both networks played with
the present code, on Block SMB and then on the real game (score only):

| | The earlier network | Learned again |
|---|---|---|
| Fresh Block SMB plays finished (of 800) | 610 | 618 |
| ... died in a pit | 46 | 38 |
| ... by Hammer Bros and their hammers | 34 | 41 |
| ... by Spinies | 22 | 8 |
| ... in lava or by fire bars | 32 | 30 |
| Held-out real levels: tries at an obstacle that fail, exact positions | 13% | 20% |
| ... from the screen | 31% | 17% |
| Held-out real levels finished, exact positions (of 32) | 5 | 7 |
| ... from the screen | 1 | 2 |
| World 1-1 finished (of 8), exact positions / from the screen | 8 / 5 | 8 / 6 |
| All eight scored levels finished (of 64), exact positions / from the screen | 16 / 8 | 20 / 10 |

Per level (share of tries at an obstacle that fail, and whole plays
finished of 8; E the earlier network, A the one learned again):

| Level | Exact positions, E | A | From the screen, E | A | Whole, exact, E | A | Whole, screen, E | A |
|---|---|---|---|---|---|---|---|---|
| 1-1 | 0% | 2% | 5% | 6% | 8 | 8 | 5 | 6 |
| 5-1 | 13% | 6% | 25% | 11% | 3 | 3 | 2 | 2 |
| 2-1 | 8% | 2% | 34% | 30% | 0 | 2 | 0 | 0 |
| 3-1 | 28% | 31% | 54% | 40% | 0 | 0 | 0 | 0 |
| 4-1 | 3% | 0% | 32% | 7% | 4 | 5 | 0 | 1 |
| 6-1 | 18% | 25% | 41% | 23% | 1 | 0 | 0 | 1 |
| 7-1 | 25% | 36% | 35% | 17% | 0 | 2 | 1 | 0 |
| 8-1 | 10% | 20% | 21% | 20% | 0 | 0 | 0 | 0 |

The new network plays from the screen about as well as from exact
positions: on the held-out levels it fails 17% of its tries from the
screen against the earlier one's 31%, though 20% with exact positions
against 13%. Each level's eight whole plays are still close to two
samples, so the benches are the finer measure. In Block SMB it falls into
fewer pits and is caught by fewer Spinies, and dies more often by hammers;
in 102 of its 175 deaths it had given the fatal choice less than a 5% chance.

What is left, by what the deaths show:

- **Lava pits.** In 10 of 14 deaths by lava fireballs (the earlier network,
  800 plays) the fireball had never been seen before the fatal choice: it
  leapt from a pit no one could tell from a plain one, as nothing in the
  state marked a pit as lava. Now (2026-11-18) Block SMB's pits that hold
  lava (those a lava fireball jumps from, and the one under Bowser's bridge)
  are seen: the state gives each as a surface of kind "lava" across the
  pit, its top where the game draws lava, which the network reads and the
  planner never stands on (State.lava). A network trained before reads the
  surfaces' numbers it knew and plays on. The network is learning again with
  it (`lava_20261118`). Vision does not yet see lava in the real game (none
  of the eight scored levels has any); teaching it is seeing's own work.
- **Springboards.** The network is about 90% right on what can be done
  from one, but Mario stands on one in 1% of the sampled states, and the
  rest of its errors there are costly (a run chosen as the springboard
  throws him, landings named that cannot be planned).
- **Choices rated safe that are not**: caught between two Spinies, each
  run toward one was given under 4%.
- **Cannons.** Half of the deaths by Bullet Bills (6 of 12, the kept
  network, 800 plays) are by one fired as the choice was made or just
  after, from a cannon the state gives as ground (as the game's labels call
  it). Told apart (a cannon surface kind, from Block SMB's record and from
  vision), the network could learn what stands by one and the planner
  could foresee its shots by their beat; as with lava, it changes the
  network's inputs and so goes with a retraining.

Hammers, traced with the kept network (2026-11-18). Of its deaths by
hammers, a third were by one not yet thrown when the choice was made, and
others by one held when it was made. Two errors of foresight were behind
them:

- **A hammer's age past what tracks keep.** The planner tells when a
  Hammer Bro throws next from the age of its last hammer, the time it has
  been in sight; every source keeps a thing's places 32 frames
  (planner.TRACK_FRAMES, now one value for all), so a hammer 47 frames old
  read as 31, and the next throw was foreseen 16 frames late. A thrown
  thing seen that long now tells neither its age nor its thrower, and a
  Bro with no hammer of known age in sight holds one over its head all the
  while.
- **A hammer still held.** One counted as held only if it had never moved
  up or down, so a hammer over Bowser jumping, or over a Hammer Bro
  jumping, was taken for thrown and foreseen falling, 56 pixels below where
  it went. A held thing is now told by its holder: a Hammer Bro or Bowser
  near it whose last moves it has made exactly; it is foreseen going where
  its holder goes until thrown (planner.holder). Over every hammer held by a
  Bro jumping between floors its foreseen height is now Block SMB's exactly
  (`scripts/tests/test_hammer_bros.py`; when a standing Bro will jump is not
  known).

The 800 plays with the kept network: 621 finished against 618, 21 deaths by
hammers against 29.

**A Paratroopa hopping under bricks.** Block SMB stops anything rising
into a solid there (not yet checked against the game); the planner's hop
did not, and foresaw a Paratroopa hopping on up through the brick row over
it, while it fell back early into Mario's jump. Nor did it wait for one
falling the last pixels onto the floor: taken for standing, it hopped a
frame early from where it was. Both now as Block SMB has them: from every
frame of a Paratroopa hopping under bricks its foreseen height is Block
SMB's exactly (`scripts/tests/test_flying_paratroopas.py`). The 800 plays:
621 finished, as before, 15 deaths by Koopas against 16; kept as the
planner foreseeing what Block SMB does.

**Fire bars.** The fireballs that killed Mario were foreseen within a few
pixels, but two things came apart:

- **A slide off a step, unwatched.** Landed on a step's last pixel with
  speed, Mario slides off onto the floor below; the planner allowed it, as
  there was floor to catch him, but stopped watching the enemies as soon
  as he was off the step, and he dropped through a fire bar's sweep beside
  it. He is now watched on the way down too, falling released until he
  lands (`scripts/tests/test_smb_planner.py`).
- **Fireballs step.** Block SMB, as the game, puts a fire bar's balls at
  the whole step of its turn reached (32 a round, one every 4.6 or 6.4
  frames); the planner foresaw them turning smoothly from half a step on,
  half a step off either way (8 pixels a step far out on a bar), and half a
  step more for a bar turning anticlockwise. How far into its step a ball
  is shows when it last stepped (its track), so it is now foreseen at the
  whole steps from there: forecasts up to 40 frames ahead in the fire bar
  scenes are exact 94% of the time, against 41% (`scripts/tests/test_fire_bars.py`;
  a ball not yet seen stepping is foreseen as before).

The 800 plays: 623 finished against 621, 10 deaths by fire bars against 17.

**A slide off a step, followed down.** Watched on the way down, the slide
was still taken for caught once the watch ended (12 frames) with Mario in
the air: under a step's end a one-tile column three tiles lower caught him
only if he was still over it when he got down to it, and landed slowly he
slid off past it into the pit beyond (fast, he is carried over the pit:
Block SMB has both so). His fall is now followed down to what catches him,
or off the screen, a fall (`scripts/tests/test_smb_planner.py`). The 800
plays: 617 finished against 623, 34 deaths in pits against 40: no clear
difference, kept as the planner foreseeing what Block SMB does.

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

(Written before the rule and the requirements, and kept as a record. What
is done next is decided by [requirements.md](requirements.md) and the
section "Learning from Block SMB alone" above: the generator is widened,
the controller's parameters are learned in Block SMB, and seeing is
improved with its own data.)

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
