# RetroAGI

A vision-based agent for Super Mario Bros.

## Current design: choose and control

Since 2026-11-07 the agent is being rebuilt around sparse decisions and a
fast controller ([the plan](docs/choose-and-control-plan.md)):

1. **The executor** (`core/smb_controller.py`, `core/smb_planner.py`): a
   model-based controller run every frame. It takes one choice at a time:
   jump onto a thing, stomp or pass an enemy, run to a place, wait. It plans
   the jump (run-up frames and how long the button is held) with the NES
   motion model against what it sees, re-plans before takeoff and corrects
   the flight in the air, and reports how the choice ended (landed,
   stomped, arrived, a threat, infeasible).
2. **The outcome network** (to be built): from the state, one pass predicts
   the consequences of every candidate choice at once (the chance of dying,
   progress right, coins and power-ups).
3. **The chooser**: the game's three rules on those predictions, in order:
   don't die, progress to the right, get coins and power-ups. Until the
   outcome network exists, a scripted chooser
   (`stages/full_smb/scripted_chooser.py`) stands in for it.

Nothing learns on Full SMB: training data comes from random levels built
in Block SMB from the real game's pieces and rules.

Where it stands (2026-11-08), the scripted chooser and the executor:

| | 1-1 obstacle bench (24 obstacles) | Full 1-1 | Random Block SMB levels (100) |
|---|---|---|---|
| Real game, game memory | 24/24 | max points finished; speed run died at x 1676 | |
| Real game, vision only | 0.92 | both died near x 2020 | |
| Block SMB, simulator state | | | 99 speed run, 98 max points |

Full-level results shift from change to change: earlier on 2026-11-08 all
four 1-1 playthroughs finished, and vision passed 0.98 of the bench.

```bash
# score a player on the 24 obstacles of 1-1
python -m retroagi.stages.full_smb.obstacle_bench --bench data/full_smb_obstacles/Level1-1 --player scripted_vision
# the scripted chooser on random levels of the real game's pieces
python -m retroagi.stages.block_smb.game_levels --levels 100
```

## Earlier design: the layered agent

This is the design the rebuild replaces; its code and trainers are still in
the repository. The hierarchy is **strategy → tactic → skill → predictive
executor**:

1. **Strategy:** externally selected `speed_run` or `max_points` objective.
2. **Tactic:** persistent option-critic choosing advance, retreat, climb or
   descend in either direction, or hold ground. There is no direction number
   in its categorical token.
3. **Skill:** reads the tactic, strategy context, ViT scene, its last 16
   actions and the executor's feedback; no prediction. Chooses an action: a
   verb (run, jump or wait), then for a run or a jump where Mario ends up,
   one of the objects the vision reports (a surface, or an enemy he lands
   on).
4. **Executor:** carries the action out to its end: predicts movement,
   chooses buttons and jump duration, brakes on arrival, and holds a spot
   relative to a moving platform. A wait is one action, as long as until a
   move is needed.

Beside them are two prediction networks, trained apart from recorded ground
truth and not used while the skills and tactics train (until then each
"prediction" is the simulator's truth):

- **World model:** an LSTM run once at the start of each action, after the
  skill has chosen it, predicting that action's outcome: where Mario and its
  object end, how many frames it takes (for a wait, how long), the chance
  it succeeds, and whether a run keeps its speed into the jump after it.
  Its prediction is the executor's destination; a landing on a still thing
  vision sees takes that thing's height from vision.
- **Target trackers:** run every frame, each predicting when and where its
  enemy and Mario will collide (a stomp or a hit). The executor aims stomps
  at the collision point and ends a wait or walk when a collision calls for
  a move.

Evaluation is Full SMB played many times, learning continuing on every
playthrough (the world model and the trackers included), in speed mode
(time to each checkpoint) and points mode; the measure is how few
playthroughs it takes before Mario plays the level successfully.

Each game has its own weights for the shared vision model. Simulator state
supplies teacher labels, ground truth and scores in training, never deployed
network inputs.

```text
screen → ViT → scene encoder
                 │
strategy → tactic (option-critic)
                 │
                 └──→ skill ← current scene + last 16 actions + executor feedback
                       │
                       ↓ action: verb + object
             world model (LSTM) → outcome (destination; a wait's length)
                       │            [Block SMB training: the ground truth]
                       ↓
                     predictive executor ← per-frame vision
                       │  ↑ target trackers: when and where each enemy collides
                       ↓ buttons            [Block SMB training: the ground truth]
```

Training proceeds **skill → tactic**, freezing lower layers. The executor has
no learned weights. Every family belongs to exactly one stage:

- **Skill: 44 families**, each played under one supplied tactic from start to
  finish: the 20 isolated maneuvers (walk, jump a gap, climb a step, jump onto
  a raised platform, walk off a ledge, jump down and stomp, each right and
  left, plus stomping an enemy on a raised platform both ways, holding a
  spot on a moving platform, climbing onto a pipe, stomping down from one
  onto a walker below, and jumping a pit reached running), two enemy-bypass families, nine local route
  maneuvers, and 13 older local families (flat runs, single gaps, a pit
  before an enemy, enemy hops, stomps and patrols, pipe and stomp mounts, pit
  leaps, platform hops, landing on an enemy, holding an area, the direct
  route of the choice scene).
- **Tactic: 49 families**, for strategy-dependent selection and termination:
  18 strategy families (seven tactics, a Goomba under a block and a pen seen
  from a pipe, under each strategy), ten composed
  levels, five scene-driven route/response families, three alternate-route
  families, three waiting/proceeding families, and 10 families whose tactic
  changes partway through. These are excluded from skill training and
  evaluation.

Both learned stages sample their scene families; an optional parameter sweep
supports selected small skill families.

Legacy spatial-skill checkpoints warm-start by discarding the action network
and preserving skill, tactic, scene encoder, and memory weights. No action
training or action checkpoint is required.

See [the architecture and training guide](docs/layered-agent.md) for the
layered agent's exact inputs, outputs, timing, curriculum, qualification
gates and known limits, and [design decisions](docs/design-decisions.md)
for what was decided, why, and what is still to be built.

## Project Layout

```text
retroagi/
  core/
    vision.py, scene_vision.py   # the vision transformer, its training and measurements
    smb_pixel_types.py           # the pixel types and the objects built from them
    smb_scene_labels.py          # true objects for training labels
    smb_observer.py              # turns a screen into the report the layers read
    tokens.py                    # strategy, tactics and spatial skill commands
    layered_policy.py            # the tactic and skill layers and their memories
    smb_agent.py                 # the agent: screens in, buttons out
    smb_executor.py              # turns a skill command into buttons, frame by frame
    smb_ground_control.py        # run commands: 8-frame predictive control with braking
    smb_trajectory.py            # jump commands: hold-length search and flight correction
    smb_spatial_feedback.py      # tracks the command's target and reports its progress
    smb_physics.py               # Mario's NES motion, shared by both games
    smb_planner.py               # the executor's jump planner (compiled)
    smb_controller.py            # the executor: one choice at a time, every frame
  stages/
    block_smb/                   # the Block SMB game, layout families, teachers
                                 # and the layer trainer (layered_train.py);
                                 # game_levels.py: random levels of the game's pieces
    full_smb/                    # emulator play, frame labels from game memory,
                                 # the Full SMB evaluation of trained layers, the
                                 # 1-1 obstacle bench and the scripted chooser
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
   retroagi-block-smb train-layer --learner skill --output artifacts/block_smb/skill
   retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/skill/passed.pt --output artifacts/block_smb/tactic
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
