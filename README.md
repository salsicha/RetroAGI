# RetroAGI
General purpose machine learning agent for retro games.

RetroAGI trains an agent for Super Mario Bros that decides in three layers and
sees only through a vision transformer:

1. **Strategy** — what the run is for: speed run (finish in the least time) or
   max points (finish with the most points: a point for every coin collected
   and every enemy killed). It is a switch set by whoever runs the agent, not
   a learned layer.
2. **Tactic** — the way Mario is going: advance, retreat, climb forward, climb
   backward, descend forward or descend backward. Forward is right, the
   level's way. A tactic is held over many actions; the layer is an
   option-critic with its own memory, so it decides when the tactic is
   finished.
3. **Action** — which buttons to press and for how many frames.

Each game has its own vision transformer (same model, its own weights). It
reads the screen and reports the objects on it: Mario, the ground and
platforms, gaps, pipes, blocks, coins and enemies. The layers read only
that report; game state is used for training labels and scores, never as an
input.

Training happens in two games:

- **Block SMB** is a simplified Mario game with the real game's physics. Its
  layouts are generated in families: single-action scenes for the action
  layer, and for the tactic layer one scene per tactic played under each
  strategy, with coins and enemies inserted at random. A teacher that can
  look ahead in the simulator labels every decision. The action and tactic
  layers train here, in that order; the tactic layer starts only after the
  action layer passes on held-out layouts.
- **Full SMB** is the original game in the stable-retro emulator. The layers
  trained in Block SMB play it unchanged, through the Full SMB vision
  transformer, under the strategy switch set for the run.

[docs/layered-agent.md](docs/layered-agent.md) explains what the agent sees,
how each layer decides and how it learns.

## Architecture

```text
 screen
   │
   ▼
 vision transformer (one per game) ──► objects on the screen ──► scene numbers
                                                                     │
                                                         scene encoder (tokens)
                                                                     │
                                  ┌──────────────────────────────────┤
                                  ▼                                  ▼
                       action memory (steps once      every layer reads the scene
                       per action; predicts the       and the action memory's
                       scene when the action ends)    prediction
                                  │
 strategy switch ─────────────────┼──────────────┐
 (set from outside:               │              ▼
  strategy, goal side)            │     tactic layer (option-critic) ◄──► tactic memory
                                  │     holds a tactic over many actions;  (steps once per
                                  │     end check, critic, choice           tactic; predicts the
                                  │              │ tactic                   scene when it ends)
                                  │              ▼
                                  └───► action layer: button action and frame count
                                                 │
                                                 ▼
                                   executor: presses it for its frames, or until
                                   the vision transformer sees Mario land
                                                 │
                                                 ▼
                                              buttons
```

**What the agent sees.** Each game has its own copy of one vision transformer
(`core/vision.py`).
- **Its output:** for every pixel of the 256×240 screen, a type: background,
  Mario, ground, brick, question block, pipe, coin, enemy, moving platform or
  power-up. For each 8×8 cell, the kind of enemy drawn there. Mario's facing,
  and whether his feet are on something (its land detector).
- **Objects:** groups of touching pixels of one type become the objects on
  the screen: Mario, each enemy, coin, power-up, moving platform, pipe and
  block, every surface Mario can stand on, and every gap.
- **The report:** the objects are packed into the numbers the layers read
  (`core/smb_observer.py`), every position measured from Mario.

Nothing else about the game reaches the agent: no speed, nothing hidden, no
game memory.

**How it decides.** The layers decide only when an action has ended, top to
bottom (`core/smb_agent.py`, `core/layered_policy.py`):
- **Scene encoder:** one token per reported object; every position also
  enters as sine and cosine waves, so a pixel's difference is a clear
  difference.
- **Action memory:** a long short-term memory network. It steps once at the
  start of every action, from the picture alone, and predicts the scene when
  the action will end. Every layer reads that prediction beside the current
  scene; it is how the agent knows about motion.
- **Strategy switch:** what the run is for (speed run or max points) and
  which side the goal is on. It is set from outside, not learned.
- **Tactic layer:** an option-critic transformer. It holds a tactic over many
  actions:
  - advance: go right, the level's way. Walking, jumping gaps, getting past
    or stomping enemies and waiting for the way to clear are all part of it;
  - retreat: go back to the left;
  - climb forward / climb backward: go up onto something higher, to the right
    or to the left;
  - descend forward / descend backward: go down to something lower, to the
    right or to the left.

  At every action start it checks whether the held tactic is finished,
  values each tactic (its critic), and says which it would choose. When the
  tactic ends, its own memory network steps once and predicts the scene when
  the next tactic will end, and the layer chooses the next tactic. Going left
  is never advancing.
- **Action layer:** a transformer that reads the tactic and its own last 16
  choices. It gives a button action (nothing, right, right + jump, left,
  left + jump or jump) and a frame count from 1 to 32. It decides from the
  scene when to jump, stomp or stand still.
- **Executor:** presses that button action on each of those frames and reads
  nothing else. The action ends when its frames are pressed, or when the
  vision transformer's land detector sees Mario land.

**How it learns** (`stages/block_smb/layered_train.py`). In Block SMB, one
layer at a time, bottom-up, with everything else frozen. A teacher that reads
the simulator gives each learner the token from the layer above, and labels
what the learner should do at every decision (training only).
- **Action layer:** given the teacher's tactic. It also trains the scene
  encoder and the action memory.
- **Tactic layer:** reads each layout's strategy switch. It learns the
  teacher's tactics and where they end, its memory's predictions and its
  critic. Then reward rounds improve its choices and its end check by the
  option-critic rules, once the critic predicts held-out returns well enough.

Which families each layer trains on (each only on families at its own level):
- **Action layer:** nine single-action families
  (`stages/block_smb/action_families.py`). Each is a scene that needs one
  action under one tactic and nothing else: walk to the goal, walk back to a
  goal behind, jump a pit, climb onto a step ahead or behind, walk off a
  ledge ahead or behind, stomp an enemy coming toward Mario, and wait on a
  moving platform that carries him to the goal.
- **Tactic layer:** 12 strategy families (`stages/block_smb/strategy_families.py`):
  one scene per tactic (advance, retreat, and climbing or descending forward
  or backward), each played under both strategies, on the same layouts.
  Coins and enemies are inserted at random: on the way, and off it in one or
  two detours Mario may take or skip (coins on a floating platform, coins
  behind him, an enemy behind him to stomp). The teacher plays every
  combination of detours; speed run takes the fastest and is paid only for
  time, max points takes the one with the most points and is paid for coins
  and kills. Its layouts are drawn at random, not swept.

The action layer trains on a full sweep: every combination of every value of
each family's parameters.
- **What counts as a value:** every whole number of a range, every option,
  every order; speeds in steps of 0.01 pixels a frame.
- **Coverage:** all of them are played every round, at every difficulty, and
  each family weighs the same.
- **Testing:** their held-out test layouts are drawn at random from the same
  values, so every test layout is one the sweep covers.
- **Size:** the nine action families make 10,375 layouts. The trainer refuses
  a family with more than 20,000 combinations at a difficulty and names it.

**Rule for each layer's families.** A layer trains only on families at its
own level. The action layer's families must be scenes that each need a single
action. The teacher labels one tactic, and only that tactic, from start to
finish, and a test checks every family. They must never be scenarios that
string several actions together, because composing actions is the tactic
layer's job. A tactic lasts until Mario lands, so a jump is one action from
take-off to landing.

A layer passes when every family wins at least 90% of its held-out layouts
(3 per difficulty by default). A tactic-layer episode is won only with its
strategy's objective met: in time for speed run, with enough points for max
points. The tactic layer must also agree with the teacher on where tactics
change. The next layer starts from the passing checkpoint (`passed.pt`). In
Full SMB the trained layers play unchanged, through the Full SMB vision
transformer, under the strategy switch set for the run.

## Project Layout

```text
retroagi/
  core/
    vision.py, scene_vision.py   # the vision transformer, its training and measurements
    smb_pixel_types.py           # the pixel types and the objects built from them
    smb_scene_labels.py          # true objects for training labels
    smb_observer.py              # turns a screen into the report the layers read
    tokens.py                    # the strategy and tactic vocabularies
    layered_policy.py            # the tactic and action layers and their memories
    smb_agent.py                 # the agent: screens in, button actions out
    smb_executor.py              # plays an action for its frames
    smb_physics.py               # Mario's NES motion, shared by both games
  stages/
    block_smb/                   # the Block SMB game, layout families, teachers
                                 # and the layer trainer (layered_train.py)
    full_smb/                    # emulator play, frame labels from game memory,
                                 # and the Full SMB evaluation of trained layers
scripts/
  vit/                           # train the two vision transformers
  vision/                        # measure the two vision transformers
  tests/                         # the test suite
```

## Supported Platforms

RetroAGI supports Linux x86-64 and macOS Apple Silicon with Python 3.12
through 3.14. It pins PyTorch 2.9.1 with torchvision 0.24.1. CPU-only
execution is the baseline; CUDA and Apple Metal/MPS acceleration are selected
automatically when available.

Reference training runs were performed on an Intel NUC with a 12th-generation
Intel processor and 64 GB of RAM, connected over Thunderbolt to a Razer Core X
V2 external GPU enclosure housing an NVIDIA Tesla V100 with 32 GB of memory.

See the [compatibility matrix and installation commands](docs/compatibility.md)
before creating an environment. Full SMB needs the game imported into
stable-retro; see [docs/full-smb-content.md](docs/full-smb-content.md).

## Usage

The [reproducibility procedure](docs/reproducibility.md) goes from a clean
checkout to a trained agent. In short:

1. Train and measure the vision transformers:
   ```bash
   python scripts/vit/train_block_vit.py --epochs 40 --samples-per-epoch 40000
   python scripts/vision/evaluate_block_vision.py
   python scripts/vit/train_full_vit.py --epochs 40 --samples-per-epoch 40000
   python scripts/vision/evaluate_full_vision.py
   ```
   [scripts/vit/README.md](scripts/vit/README.md) describes both models,
   their trainers and their measurements.
2. Train the layers in Block SMB, each starting from the run below it:
   ```bash
   retroagi-block-smb train-layer --learner action --output artifacts/block_smb/action
   retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/action/passed.pt --output artifacts/block_smb/tactic
   ```
   `passed.pt` is the best round that met the layer's bar on every family;
   `best.pt` is the best round overall (of equally good rounds, the latest).
   `retroagi-block-smb train-layer --help` lists every setting.
3. Examine a saved agent on fresh held-out Block SMB layouts, and play Full SMB
   with it:
   ```bash
   retroagi-block-smb exam-layer --checkpoint artifacts/block_smb/tactic/passed.pt
   python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
   ```
4. Run the test suite:
   ```bash
   python -m pytest scripts/tests
   ```

Other tools:

- `python scripts/mario_scenario_env.py` plays Block SMB by keyboard.
- `python scripts/semantic_segmentation.py` shows Block SMB beside its true
  pixel types.
- `python scripts/smb_physics_audit.py --output <file>` checks that Block SMB
  moves Mario exactly as the emulator does.

## Earlier design notes

These describe ideas from before the layered agent; the code they mention
has been removed:

- [AI teaching curriculum](docs/ai-teaching-curriculum.md)
- [Universal retro oracle roadmap](docs/universal-retro-oracle.md)
- [Hierarchical self-supervised planning plan](docs/hierarchical-self-supervised-planning.md)
- [Universal embodied framework design](docs/universal-embodied-framework.md)
