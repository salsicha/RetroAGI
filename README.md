# RetroAGI
General purpose machine learning agent for retro games.

RetroAGI trains an agent for Super Mario Bros that decides in four layers and
sees only through a vision transformer:

1. **Strategy** — what the run is for: finish fast, collect the most coins, or
   take the fewest risks. It is a switch set by whoever runs the agent, not a
   learned layer.
2. **Tactic** — what to do in this part of the level: advance, take another
   route, hold the area, or retreat. A tactic is held over many actions; the
   layer is an option-critic with its own memory, so it decides when the
   tactic is finished.
3. **Skill** — the next move: advance, jump a gap, climb, descend, stomp,
   retreat or wait, with its target.
4. **Action** — which buttons to press and for how many frames.

Each game has its own vision transformer (same model, its own weights). It
reads the screen and reports the objects on it: Mario, the ground and
platforms, gaps, pipes, blocks, coins and enemies. The layers read only
that report; game state is used for training labels and scores, never as an
input.

Training happens in two games:

- **Block SMB** is a simplified Mario game with the real game's physics. Its
  layouts are generated in families (gaps, stairs, enemies, moving platforms,
  piranha plants, alternate routes, retreats, and composed scenes whose tactic
  changes along the way). A teacher that can look ahead in the simulator
  labels every decision. The action, skill and tactic layers train here, in
  that order; each layer starts only after the one below it passes on
  held-out layouts.
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
                                  │     skill layer: skill, direction, target
                                  │              │ skill
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
- **Strategy switch:** what the run is for (speed run, max coins or careful)
  and which side the goal is on. It is set from outside, not learned.
- **Tactic layer:** an option-critic transformer. It holds a tactic (advance,
  alternate route, hold area or retreat) over many actions. At every action
  start it checks whether the held tactic is finished, values each tactic
  (its critic), and says which it would choose. When the tactic ends, its own
  memory network steps once and predicts the scene when the next tactic will
  end, and the layer chooses the next tactic. A tactic's direction follows
  from the goal's side.
- **Skill layer:** a transformer that reads the tactic and its own last 16
  choices, and gives the next move. The move is a skill (advance, jump gap,
  climb, descend, stomp, retreat or wait), a direction, and a target: a
  surface, enemy or moving platform on the screen.
- **Action layer:** a transformer that reads the skill, the target's box and
  its own last 16 choices. It gives a button action (nothing, right, right +
  jump, left, left + jump or jump) and a frame count from 1 to 32.
- **Executor:** presses that button action on each of those frames and reads
  nothing else. The action ends when its frames are pressed, or when the
  vision transformer's land detector sees Mario land.

**How it learns** (`stages/block_smb/layered_train.py`). In Block SMB, one
layer at a time, bottom-up, with everything else frozen. A teacher that reads
the simulator gives each learner the token from the layer above, and labels
what the learner should do at every decision (training only).
- **Action layer:** given the teacher's skill. It also trains the scene
  encoder and the action memory.
- **Skill layer:** given the teacher's tactic.
- **Tactic layer:** reads each layout's strategy switch. It learns the
  teacher's tactics and where they end, its memory's predictions and its
  critic. Then reward rounds improve its choices and its end check by the
  option-critic rules, once the critic predicts held-out returns well enough.

The action and skill layers train on every family except the strategy
courses. A course is about its strategy's objective (a deadline, a coin
count), which those layers can't see, so the courses are used only by the
tactic layer, once the actions and skills are trained. The tactic layer
leaves out the clone families, whose tactic is given rather than decided by
the scene.

A layer passes when every family wins at least 90% of its 18 held-out
layouts. The tactic layer must also agree with the teacher on where tactics
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
    tokens.py                    # the strategy, tactic and skill vocabularies
    layered_policy.py            # the tactic, skill and action layers and their memories
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
   retroagi-block-smb train-layer --learner skill --init artifacts/block_smb/action/passed.pt --output artifacts/block_smb/skill
   retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/skill/passed.pt --output artifacts/block_smb/tactic
   ```
   `passed.pt` is the best round that met the layer's bar on every family;
   `best.pt` is the best round overall. `retroagi-block-smb train-layer --help`
   lists every setting.
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

These describe ideas from before the four-layer agent; the code they mention
has been removed:

- [AI teaching curriculum](docs/ai-teaching-curriculum.md)
- [Universal retro oracle roadmap](docs/universal-retro-oracle.md)
- [Hierarchical self-supervised planning plan](docs/hierarchical-self-supervised-planning.md)
- [Universal embodied framework design](docs/universal-embodied-framework.md)
