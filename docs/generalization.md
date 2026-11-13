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

| | First contact | After the second round | Learned from Block SMB alone, round 2 | A working agent |
|---|---|---|---|---|
| Whole levels finished from the screen alone | 0 of 32 plays | 0 of 32 | 1 of 32 | most |
| Tries at an obstacle that fail, from the screen alone | 47% | 41% | 34% | about 2% |
| The same with exact positions in place of seeing | 35% | 24% | 35% | about 2% |

(The first two columns are of policies changed by hand while watching real
levels, which the rule below forbids and which were put back. The third is
of the present policy: its deciding network learned in Block SMB alone.)

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

Still missing: lifts, blocks invisible until hit.

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
| Fire bars | 1-4, 5-4 (the bar's turn from memory, its fireballs as drawn) | Six fireballs (twelve on a long bar) 8 apart out from the middle of a block, turning a 32-step circle once in 205 frames, the fireballs at the whole step reached (within a pixel of where the game draws them); they hurt, cannot be stomped, and shells pass them. A piece: one to three bars on blocks at the heights of the game's 47 (in the floor, on it, or 2, 3, 4, 6 or 8 tiles over it), a quarter turning clockwise. A new kind for the planner (a turner: it goes round its hub at its turn, which the state gives with it). The hand-written chooser passes 58 of 80 fire bar scenes |
| Lava fireballs | 2-4 | Jumps out of its pit from under the screen at 5.95 to 6.35 pixels a frame (peaking 70 to 92 from the top: the game's two measured leaps peaked at 73 and 90), under 28/256, falling at most 3.25, and jumps again 21 times 6 to 15 frames later; the game's 10 by 6 box; it hurts and cannot be stomped. A piece: one to three lava pits 1 to 3 tiles wide (the castles'), a fireball in each; one under a fifth of the bridges. A new kind for the planner (a fountain). The hand-written chooser passes 74 of 80 lava scenes |

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

Still missing: lifts that loop up or down (1-2, 4-2) and balance lifts
(3-3, 4-3, 6-3); in the castles, the one fast fire bar (5-4), Bowser and
his flames (measured: flames fly left at 1.25 at 112, 128 or 144 from the
top), and the game's own test of a fire bar's touch (Block SMB
takes the fireballs' drawn 8 by 8); water levels (swimming, Bloopers,
swimming Cheep Cheeps); blocks invisible until hit. In the real game, the
state from memory and from vision show lifts without their speed, and
show no springboard, no fire bar going round and no lava fireball. None of the networks has learned from the widened Block SMB yet:
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
