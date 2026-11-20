# What this project must be

The project owner's standing requirements, in the owner's own words with
the day they were said, each followed by what it means for the work. This
page is the authority: where another document or the code disagrees with
it, they are wrong and are corrected. A new requirement is written here the
day it is said ("make sure it is recorded in the documents, I don't want to
have to keep repeating myself", 2026-11-11).

## 1. The goal: a trained, working system before it meets the real thing

> "the whole point of this project is to have a trained working system
> BEFORE it sees the real environment because a robot can't be trained in
> real situations." (2026-11-11)

> "the framework should be general so that it can be applied to any game
> and eventually any robotic system, but it should be a combination of
> adaptive control and interchangeable models. for this reason I think the
> system needs to be composable." (2026-11-11)

> "the point is to make a composable system where the problem is broken
> into digestible pieces" (2026-11-11)

- Super Mario Bros stands in for a robot's real situation. The present
  target: "We want to pass full smb 1-1 but we don't know the layout"
  (2026-11-11).
- Each piece has its own data and its own test, behind a declared
  interface, and can be exchanged for another that answers the same
  question ([the framework](framework.md)).

## 2. What learns, and from what

> "the goal of this project is to build a system that can learn. the models
> should learn from data. the controller is adaptive and can be passed in
> learned parameters to adapt to any situation. the decision making should
> come from a deep learning model that has learned to estimate the outcomes
> of actions. there are block smb skills that can produce all the data we
> need." (2026-11-11)

> "the learned policy that makes decisions and controls the agent must be
> learned from block smb" (2026-11-11)

- Deciding is a deep network that has learned to estimate what each action
  leads to. It is not hand-written tracking and hand-written choices.
- The controller adapts through parameters passed to it, and those
  parameters are learned, in Block SMB.
- The training data is made in Block SMB, by its skills (short scenes each
  calling for one skill) and random levels.

## 3. What the real game may be used for

> "you can observe the real game to build block-smb, but you can't observe
> the real game to inform ANY of the control or inference models that are
> learning to play the game." (2026-11-11)

> "yes, observe full smb to train the vit (we need vision and theres no way
> around that), and measuring it to make block smb accurate." (2026-11-11)

> "if you are able to read the memory file of full smb and capture the look
> and behavior directly I think that is valid and should speed up
> development" (2026-11-11)

> "I don't want the models learning the full SMB layout" (2026-11-11)

> "A general intelligence that plays smb doesn't learn long-term
> consequences. If I know how to play smb, I don't need to remember a level
> to play it. The levels could be randomly generated and I could still pass
> them. That is the part that is general." (2026-11-07)

The real game is looked at for three things and nothing else:

1. **to train seeing** (the vision network for the real game);
2. **to measure how the game behaves**, so that Block SMB behaves the same:
   how Mario and each enemy move, what pieces levels are made of and how
   big and how far apart they are. Reading the game's memory for this is
   fine;
3. **to score the finished system**, once it is trained.

What this rules out:

- **No model learns a real level's layout.** Every practice world, for
  training and for the Block SMB exam alike, is an arrangement drawn at
  random from the game's pieces. A real level laid out in Block SMB's
  format (`nes/clone.py`) is a measuring device only: the same situation is
  set up in the game and in Block SMB and what follows is compared. Nothing
  is sampled, trained or examined on it.
- **A failure in the real game never changes the policy.** It may only be
  turned into the question "what does Block SMB lack, or do differently?".
  Block SMB is corrected (a piece added, a range widened, a rule measured
  again), the policy learns again there, and it is debugged on fresh Block
  SMB worlds.
- No value over long-term consequences, no memory of a level.
- Nothing learns while playing the real game ("dont train on full smb for
  now, just run it and evaluate", 2026-10-28).

## 4. Block SMB: fast, cheap, and not the game's assets

> "The point of block smb is that it should be fast and cheap" (2026-11-11)

> "Block smb is intentionally not full smb assets" ... "Don't use full game
> assets" (2026-11-11)

> "The vision transformers are different on purpose, we shouldn't need the
> full resolution vit, in fact it's completely fine if block smb isn't
> learned at all but just a fast lookup" (2026-11-11)

- Block SMB copies how the game **behaves** (its rules, its motion, its
  pieces and their sizes). It does not copy the game's graphics or its
  levels: it is drawn as plain blocks and loads nothing of the game's
  pictures, levels or program.
- **In Block SMB the state is a fast lookup** from the simulator, not the
  output of a network. The full-resolution vision network belongs to the
  real game and is never run in Block SMB. That the policy learns from
  looked-up positions and then plays the real game from what vision reports
  is the design. What vision's errors cost in the real game is seeing's own
  problem, worked on with seeing's own data.
- Anything that would make Block SMB slow or expensive (rendering, a
  network in its loop, the game's emulator) is kept out of it.

## 5. The controller: much faster than the deep models

> "The point of the adaptive controller is that it should be much faster
> than the deep models. It should be capable of running on high speed
> robotics controllers, but it could also enable faster than real time
> simulation and data collection" (2026-11-11)

- No network runs inside the controller. Its work each frame is plain
  arithmetic on the state and its parameters, small enough for a robot's
  control loop.
- The deep models are consulted seldom (a decision every 25 frames or so);
  the controller acts every frame between decisions.
- Because the controller and Block SMB are both cheap, training data is
  collected many times faster than the game runs.

## 6. The controller's settings are chosen, by what they lead to

> "make them part of each choice" (2026-11-12, choosing between two ways
> the controller's hand-set safety margins and timings could be learned:
> as part of each choice, or as fixed values found by search in Block SMB)

- The settings that decide how carefully a choice is carried out (in Super
  Mario Bros: how wide a berth enemies are given, how long a landing must
  stay clear of them) become part of the choice. Each candidate choice
  comes in its variants of those settings; the outcome network predicts
  what each variant leads to, and the chooser picks among them as among
  any choices. The parameters the controller is passed are then picked per
  situation by what was learned in Block SMB, not set by hand.

## 7. How the work is done

- No training run is started unless asked for; a failed check is fixed in
  what failed, never bypassed.
- Networks run on the GPU, simulation on the processor.
- Plain language in every document and report: what a part reads, computes
  and gives; no unexplained code names.
- Tested work is committed and pushed promptly; functions are changed in
  place, without forked variants.

> "I don't think you have to talk about the "current design", you can just
> call it the design, and remove references to the earlier design notes.
> also give the readme some explanatory diagrams so it isn't all text"
> (2026-11-13)

- The documents describe the design as it is, called the design. Retired
  designs are not described in them; retired code keeps its own documents
  beside it (`retroagi/legacy/docs`).
- The README explains with diagrams as well as text, and is kept up to
  date with the code.

## Where the code stands against this (2026-11-19)

| Requirement | Where it stands |
|---|---|
| Trained before it meets the real game | The outcome network has learned in Block SMB only. The hand-written parts (the controller's rules and constants, the list of choices, the chooser's priorities, the tracking of things seen) were adjusted while watching Worlds 1-1 and 5-1 before the rule was said; they have not yet been learned or set again from Block SMB alone. |
| Deciding by a deep model of outcomes | The outcome network predicts every candidate choice's outcome in one pass and fixed priorities pick among them. The list of candidate choices is still written by hand. |
| The controller takes learned parameters | Its berth around enemies, how long a landing must stay clear and how soon a coming enemy ends a run are part of each choice since 2026-11-12 (section 6): bold, usual or careful, picked by the outcome network's predictions. The three sets, and the controller's other constants, are still set by hand. A network learned with them from nothing (2026-11-13) scores about as the baseline does: 38% of tries at obstacles failing on held-out levels from the screen against 36%, 1-1 finished in 7 of 8 plays from the screen against 4. |
| The controller is much faster than the deep models | Yes, measured below. Nothing it imports is a deep-learning library (kept so by a test). |
| Block SMB is fast and cheap | Yes, measured below. |
| Practice worlds are random, of the game's pieces | Random levels and skill scenes drawn from pieces and ranges measured in the game's memory. Since 2026-11-13 every level not scored on has been read for what it holds (`scripts/dev/game_catalogue.py`: 34 kinds of things, Block SMB had 6) and its shapes: Block SMB now also has red Koopas, Lakitu and Spinies, flying Paratroopas, Bullet Bill and Cheep Cheep frenzies, lifts, the sky levels' ledges, one-tile pits and longer bridges, springboards, the castles' fire bars (the fast one too) and lava fireballs, Bowser, his flames and his hammers, looping and balance lifts, blocks invisible until hit. Still missing: water levels. A network has learned in it from nothing (2026-11-14): 66 of 100 fresh random levels finished in speed runs, 98% of skill scenes passed. Every world checked can be passed by some sequence of buttons (`sim/passable.py`). Its deaths are now traced in Block SMB itself (2026-11-16, [generalization.md](generalization.md)): the planner prefers landings with Mario's feet well on the solid, and Block SMB and the planner set Mario onto a ledge met with his feet up to 3 pixels below its top, as measured in the game (a one-tile pit is walked across); the same network then finishes 144 of 200 fresh levels instead of 134. Since 2026-11-17 the planner lets things knocked off a ledge fall as the game's do, and lifts meet Mario as the game's do (measured at World 1-3's lift): his head bumps them from below, he lands on one only with his feet within 4 pixels of its top, and its side stops him only when met from beside: rising, or with his feet more than 4 pixels below its top (level or falling with them nearer, he is set on it). The planner now foresees every jump at a raised block or lift where Block SMB lands him, the controller lets go once Mario is down after a jump's last button, and a run counts a gap before a step up as a pit: the network unchanged finishes 588 of 800 fresh plays against 567, with 57 deaths in pits against 77. Then the planner foresees a Hammer Bro's coming hammers and its own path (it meets no walls and drops through the floor it stood on), Block SMB leaves a Koopa just stomped out of its wings harmless as the game does, and a choice the controller refuses is answered with the next in the same frame: 610 of the 800 finished, 46 deaths in pits, 13 by Hammer Bros against 26. On 2026-11-18 the planner was made to foresee what Block SMB does where it did not (a springboard's squash and throw, a stomp's bounce, lifts at whole pixels and below the screen, a jump at a block's side, a slide down a staircase, a walk to the left) and a run to stop for an edge and for things falling onto it: the kept network, unchanged, finishes 675 of 800 fresh plays against 617, 15 deaths in pits against 34. |
| No game assets, no real layouts in Block SMB | Block SMB draws plain boxes and loads no picture. Since 2026-11-11 no practice world has a real level's layout (kept so by a test). Three real levels had been laid out as practice worlds that day; the second training round, which had begun to sample them, was stopped and what it had sampled on them deleted. No kept network learned from them. |
| In Block SMB the state is a lookup | Yes. |
| One command retrains the deciding network from nothing | `scripts/learn_in_block_smb.sh NAME` (rounds of sampling, training and the Block SMB exam). A full run from nothing (2026-11-12, 7.7 hours) finished 4 of 8 whole plays of World 1-1 from the screen and 8 of 8 with exact positions; earlier trainings gave 4, 6 and 4 from the screen: a retraining does not yet pass 1-1 reliably. The run in the widened Block SMB (2026-11-14, 18.7 hours) finishes 33 of 64 whole plays of the eight scored levels with exact positions (the care run: 11), 15 of 32 on the held-out ones, but only 5 of 64 from the screen (care run: 9); 1-1 in 5 of 8 from the screen. Since vision's motion estimate stops Mario's fall when he stands (2026-11-15) and stops his run at walls, 24% of its tries at obstacles on the held-out levels fail from the screen, against 25% with exact positions (46% before), and 3 of 32 whole held-out plays finish from the screen. Learned again from nothing in the corrected Block SMB (2026-11-16, 7.8 hours, `data/outcome/outcome_corrected_20261116.pt`), it makes no clear difference ([generalization.md](generalization.md)): 291 of 400 fresh Block SMB plays finished against 284; on the held-out levels 18% of tries at obstacles fail with exact positions against 26%, 30% from the screen against 27%; 1-1 in 7 of 8 from the screen, as the widened network now does. Learned again with the next day's corrections (2026-11-18, `data/outcome/outcome_sided_20261117.pt`, kept by Block SMB alone): 618 of 800 fresh Block SMB plays finished against 610; on the held-out levels 17% of tries at obstacles fail from the screen against 31%, 20% with exact positions against 13%; 1-1 in 6 of 8 from the screen against 5 ([generalization.md](generalization.md)). Learned again with pits holding lava seen (2026-11-19, 27.5 hours), it is not kept: with the present code 657 of the 800 plays finished against the kept network's 675, as many deaths by lava fireballs (15 against 14: a fireball's next leap is not known while it is under the lava), and on the held-out levels 24% of tries at obstacles fail with exact positions against 17%, 27% from the screen against 26%. Since vision's motion estimate keeps a rising Mario in the air whatever the screen shows (2026-11-19; it let him speed up in the air where the game does not) a hopping Paratroopa's speed is read through its bounces and a thing whose sprite changes shape keeps its track, the kept network fails 18% of its tries at obstacles on the held-out levels from the screen (26% before; 17% with exact positions), and finishes 11 of 64 whole plays of the eight levels from the screen against 13. |
| Composable, interchangeable parts | Interfaces for every part; so far one part has an alternative (how other things are predicted to move). |
| Any game, later any robot | Two games play. The framework imports nothing of either (kept so by a test). Pong, the second, has been measured from its memory and Block Pong built from it (2026-11-15, [pong.md](pong.md)): frame by frame it agrees with the game on our paddle, theirs, the serve and the score always, and on the ball but where it grazes our paddle's rim (17 of 200,000 frames). Its outcome network learned from nothing in Block Pong alone with the framework's commands unchanged (2026-11-16, 45 minutes) and wins the real game 21 to 0 in 4 of 4 plays from the screen, as from memory: seeing is a vision network trained on the real game's frames and a tracker that keeps every exact state the screen allows. No choice in Pong decides whether a point is lost, so its chance of death has nothing to learn. |

**Speed, measured 2026-11-11** on one processor core, in milliseconds:

| | Each frame | Each call | How often |
|---|---|---|---|
| The controller, pressing on with a choice | 0.03 to 0.05 | | every frame |
| The controller, starting a choice (a jump planned) | | 0.7 to 1.0 | once in about 25 frames |
| Block SMB's step | 0.07 | | every frame |
| Block SMB's state lookup | 0.02 | | every frame |
| The outcome network, one decision | | 1.4 on the processor, 2.3 on the GPU | once in about 25 frames |
| Vision in the real game | 4.0 | | every frame |
| The real game's emulator | 0.5 | | every frame |

The whole agent plays Block SMB at 4,200 frames a second on one core with
the outcome network choosing (6,300 with the hand-written chooser): 70 to
105 times the game's 60 frames a second. The controller's work each frame
is about a hundredth of vision's.
