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

## 6. How the work is done

- No training run is started unless asked for; a failed check is fixed in
  what failed, never bypassed.
- Networks run on the GPU, simulation on the processor.
- Plain language in every document and report: what a part reads, computes
  and gives; no unexplained code names.
- Tested work is committed and pushed promptly; functions are changed in
  place, without forked variants.

## Where the code stands against this (2026-11-11)

| Requirement | Where it stands |
|---|---|
| Trained before it meets the real game | The outcome network has learned in Block SMB only. The hand-written parts (the controller's rules and constants, the list of choices, the chooser's priorities, the tracking of things seen) were adjusted while watching Worlds 1-1 and 5-1 before the rule was said; they have not yet been learned or set again from Block SMB alone. |
| Deciding by a deep model of outcomes | The outcome network predicts every candidate choice's outcome in one pass and fixed priorities pick among them. The list of candidate choices is still written by hand. |
| The controller takes learned parameters | Not yet: its margins and timings are set by hand. |
| The controller is much faster than the deep models | Yes, measured below. Nothing it imports is a deep-learning library (kept so by a test). |
| Block SMB is fast and cheap | Yes, measured below. |
| Practice worlds are random, of the game's pieces | Random levels and skill scenes drawn from pieces and ranges measured in Worlds 1-1, 2-1, 3-1 and 5-1, with every enemy the building levels have (Hammer Bros since 2026-11-12); every one checked so far can be passed by some sequence of buttons (`sim/passable.py`). |
| No game assets, no real layouts in Block SMB | Block SMB draws plain boxes and loads no picture. Since 2026-11-11 no practice world has a real level's layout (kept so by a test). Three real levels had been laid out as practice worlds that day; the second training round, which had begun to sample them, was stopped and what it had sampled on them deleted. No kept network learned from them. |
| In Block SMB the state is a lookup | Yes. |
| One command retrains the deciding network from nothing | `scripts/learn_in_block_smb.sh NAME` (rounds of sampling, training and the Block SMB exam). Checked end to end at a tiny size only; a full run from nothing has not been made. Three trainings so far gave 4, 6 and 4 whole plays of World 1-1 finished out of 8 from the screen: a retraining does not yet pass 1-1 reliably. |
| Composable, interchangeable parts | Interfaces for every part; so far one part has an alternative (how other things are predicted to move). |
| Any game, later any robot | One game. The framework imports nothing of it (kept so by a test). |

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
