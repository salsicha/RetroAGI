# The four-layer SMB agent

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
- Mario's facing, and whether he is standing, in the air or on a moving
  platform.

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
  plus its box relative to Mario, and the current skill's target. The lists
  are enemies, coins, power-ups, moving platforms, pipes, blocks, surfaces
  and gaps.
- **A and B (8 and 16 numbers):** for each vertical band of the screen, the
  most important thing drawn in it.

## How it decides

`retroagi/core/layered_policy.py` holds the policy: four layers, a shared scene
encoder, a two-picture window and a memory.
- **Scene encoder:** one token per reported object, plus the band codes.
  Every position number also enters as 8 sine and 8 cosine waves, with
  wavelengths from 512 pixels (480 vertically) down to 4. A difference of one
  pixel is therefore a clear difference of input, while the long waves still
  tell near from far.
- **Memory**, a long short-term memory network, steps once per action, at
  the action's start, before anything decides:
  - **Input:** vision information only, never the action Mario is about to
    take. It is the window's summary of the two latest pictures: the frame
    the action starts on and the frame before. In that summary, each object
    is marked "this frame" or "one frame before", so motion and speed are
    visible.
  - **Output:** its expected scene, a prediction of the scene numbers the
    vision transformer will report when the coming action ends, completed
    or interrupted.
  - **Who reads it:** the expected scene is encoded like a real one and given
    to the decision layers beside the current scene.
  - **Training:** action by action. It's scored against the scene actually
    reported when each action ended.
- **Strategy layer:** emits progress, careful or collect, and a direction.
- **Tactic layer:** emits advance, alternate route, hold area or retreat, and a
  direction.
- **Skill layer:** emits a skill and a direction, says whether enemy contact is
  required, and points at a target object. The skills are advance, clear a
  gap, mount a platform, wait for something to pass, clear an enemy, and
  retreat and recover.
- **Action layer:** emits an action and how many frames it lasts. Walks and
  waits choose from 1 to 96 frames; jumps choose how long the jump button is
  held, from the NES menu.

The executor (`retroagi/core/smb_executor.py`) plays one plan at a time.
- **Walks and waits** last their frame count.
- **A jump** holds the button for its count, then keeps the direction until
  the vision transformer reports a landing.
- **A plan is cut short** when the vision transformer reports Mario touching
  an enemy, Mario missing, or Mario walking off an edge.

The layers decide only when a plan ends, top layer first.
`retroagi/core/smb_agent.py` (`SMBAgents`) runs all of this from screens
alone, for both games.

## How it learns, in Block SMB

`retroagi/stages/block_smb/layered_train.py` trains one layer at a time, from the
bottom up, with the others frozen.
- **Action layer:** given the skill.
- **Skill layer:** given the tactic.
- **Tactic layer:** given the strategy.

The token from above comes from a teacher that reads the simulator
(`retroagi/stages/block_smb/teacher_tokens.py`). The teacher is used only in
training.

1. **Imitation rounds.** Workers play episodes through the vision transformer.
   At every decision the teacher says what it would do from that exact state;
   the learner trains on all such labels so far. In the first round every
   decision plays the teacher's choice, and the share falls to none by the
   last imitation round.
2. **Reward rounds.** The learner samples its own choices and learns from the
   rewards each family gives until the next decision. The teacher's labels
   still count.

A layer counts as trained when every family passes the bar on held-out
layouts. After the tactic layer, the whole agent is also scored as deployed:
every token is its own, and the strategy is "progress" until the strategy
layer has learned.

```bash
python -m retroagi.stages.block_smb.cli train-layer --learner action --reward-rounds 8
```

```bash
python -m retroagi.stages.block_smb.cli train-layer --learner skill --init artifacts/block_smb/layered/best.pt
```

## Full SMB

The layers trained in Block SMB play Full SMB unchanged, through the Full SMB
vision transformer. `retroagi/stages/full_smb/play.py` gives the agent screens;
game memory is read only to score how far Mario got.

```bash
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/layered/best.pt
```

The strategy layer learns only from playing Full SMB.
