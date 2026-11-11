# RetroAGI

A vision-based agent for Super Mario Bros, built as a game-neutral framework
with the game as its first domain: an adaptive controller plus
interchangeable models, composed through declared interfaces
([the framework](docs/framework.md)).

## Current design: choose and control

Since 2026-11-07 the agent is built around sparse decisions and a fast
controller ([the plan](docs/choose-and-control-plan.md)). Five parts, each
behind an interface in `retroagi/framework/interfaces.py`:

1. **A state source** turns the world into the state the others read: in
   the real game from vision alone (`domains/smb/perception/`,
   `domains/smb/nes/sources.py`), in the simulator from its truth.
2. **The choice space** (`domains/smb/choices.py`) lists what can be done
   now: jump onto a thing, stomp or pass an enemy, run to a place, wait,
   bump a block, collect an item.
3. **The outcome network** (`framework/outcome_network.py`): from the
   state, one pass predicts the outcome of every candidate choice at once
   (the chance of dying, of being infeasible, of leaving Mario stuck, of
   reaching the flag, and the progress, frames and points). It learns from
   outcomes measured, not taught: in practice worlds every candidate is
   played from saved states by the controller (`framework/sampling.py`).
4. **The chooser** (`framework/chooser.py`): fixed priorities on those
   predictions, in order: don't die, progress to the right, get coins and
   power-ups. A hand-written chooser (`domains/smb/scripted_chooser.py`) is
   kept to compare against.
5. **The controller** (`domains/smb/controller.py`, `planner.py`): a
   model-based controller run every frame. It takes one choice at a time,
   plans the jump (run-up frames and how long the button is held) with the
   NES motion model against what it sees, re-plans before takeoff and
   corrects the flight in the air, and reports how the choice ended
   (landed, stomped, arrived, a threat, infeasible).

Nothing learns on the real game: training data comes from random levels
built in the Block SMB simulator from the real game's pieces and rules
(`domains/smb/sim/`).

Where it stands (2026-11-11), with outcome_20261109d, on the two levels the
agent was built on and on six nobody looked at
([the measurements](docs/generalization.md)). Share of a level's obstacles
passed, four tries each that start at different moments:

| State, dynamics model | 1-1 and 5-1 (tuned), 59 obstacles | 2-1, 3-1, 4-1, 6-1, 7-1, 8-1 (untouched), 206 obstacles |
|---|---|---|
| Game memory, hand-written rules | 0.96 | 0.65 |
| Game memory, model fitted to what is seen | 0.96 | 0.66 |
| Vision only, hand-written rules | 0.83 | 0.49 |
| Vision only, model fitted to what is seen | **0.91** | **0.55** |

From its start, only 1-1 is finished reliably. On fresh random levels in
the simulator the network's chooser finishes 87 of 100 in a speed run and
91 for points (network h, trained with 5-1's enemies: 93 and 93).

The outcome network was trained only on Block SMB random levels (question
blocks, mushrooms, big Mario, Koopa Troopas, piranha plants, hopping
Paratroopas, Bullet Bill cannons); nothing learns on the real game.

```bash
# score an agent on the 24 obstacles of 1-1, from vision
python -m retroagi.framework.bench --domain smb --bench data/full_smb_obstacles/Level1-1 --source vision --model data/outcome/outcome_20261109d.pt
# the same with the dynamics model fitted to what is seen
python -m retroagi.framework.bench --domain smb --use dynamics=observed --bench data/full_smb_obstacles/Level1-1 --source vision --model data/outcome/outcome_20261109d.pt
# play a whole level
python -m retroagi.framework.play --domain smb --level Level1-1 --source vision --model data/outcome/outcome_20261109d.pt
# the scripted chooser, or the outcome network's, on fresh random levels
python -m retroagi.framework.practice --domain smb --levels 100 --first-seed 100000 [--model data/outcome/outcome_20261109d.pt]
# sample outcomes, then train the outcome network
python -m retroagi.framework.sampling --domain smb --levels 2000 --workers 14 --output data/datasets/block_smb/outcomes_NEW
python -m retroagi.framework.training --domain smb --data data/datasets/block_smb/outcomes_NEW --fresh data/datasets/block_smb/outcomes_fresh_NEW --output artifacts/block_smb/outcome_NEW
```

## Earlier design: the layered agent

This is the design the rebuild replaces; its code and trainers are still in
the repository (`retroagi/legacy`). The hierarchy is **strategy → tactic → skill → predictive
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
  framework/                     # game-neutral: imports no domain
    interfaces.py                # what a domain supplies; what the parts ask of each other
    agent.py                     # the loop: state, chooser, controller, one action a frame
    outcome_network.py           # predicts every choice's outcome in one pass
    chooser.py                   # the fixed priorities on those predictions
    observed_dynamics.py         # a dynamics model fitted to what is seen of each thing
    registry.py                  # domains by name; their parts' alternatives (--use PART=NAME)
    sampling.py, training.py     # outcomes measured in practice worlds; the network trained on them
    practice.py, bench.py, play.py   # practice worlds, obstacle benches, whole levels
  domains/smb/                   # Super Mario Bros
    domain.py                    # the game as the framework sees it
    controller.py, planner.py    # one choice at a time, every frame; the jump planner (compiled)
    physics.py                   # Mario's NES motion
    choices.py                   # the candidate choices and the state's tokens
    scripted_chooser.py          # the hand-written chooser
    perception/                  # the vision transformer, tracks, motion estimate, vision to state
    sim/                         # Block SMB: the simulator and random levels of the game's pieces
    nes/                         # the real game: emulator, game memory, frame labels, obstacle benches,
                                 #   the dynamics models scored against the game (dynamics_check.py)
  legacy/                        # the retired layered agent: its layers, teachers and curriculum
scripts/
  vit/                           # train the two vision transformers
  vision/                        # measure the two vision transformers
  dev/                           # refactoring check; result tables; what killed the agent; one obstacle traced
  tests/                         # the test suite (tests/legacy: the retired agent's)
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
   python -m retroagi.legacy.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
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
