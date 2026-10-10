# RetroAGI

A vision-based hierarchical agent for Super Mario Bros.

The hierarchy is **strategy → tactic → skill → predictive executor**:

1. **Strategy:** externally selected `speed_run` or `max_points` objective.
2. **Tactic:** persistent option-critic choosing advance, retreat, climb or
   descend in either direction, or hold ground. There is no direction number
   in its categorical token.
3. **Skill:** reads the tactic, strategy context, ViT scene, LSTM prediction and its last 16
   spatial commands. Chooses a run/jump/hold mode and a destination relative
   to Mario's feet.
4. **Executor:** predicts movement, chooses buttons and jump duration, brakes
   on arrival, tracks moving stomp targets, and holds a spot
   relative to a moving platform. Waits are reconsidered every frame so a
   departure window cannot disappear inside a long hold plan.

Each game has its own weights for the shared vision model. Simulator state
supplies teacher labels and scores in training, never deployed network inputs.

```text
screen → ViT → scene encoder → scene memory (LSTM prediction)
                 │                      │
strategy → tactic (option-critic)        │
                 │                      │
                 └──→ skill ←───────────┘
                       ↑ current scene + last 16 skill commands
                       │
                       ↓ relative spatial destination + movement mode
                     predictive executor ← per-frame vision + button history
                       ↓ buttons
```

Training proceeds **skill → tactic**, freezing lower layers. The executor has
no learned weights. Every family belongs to exactly one stage:

- **Skill: 38 families**, each played under one supplied tactic from start to
  finish: the 17 isolated maneuvers (walk, jump a gap, climb a step, jump onto
  a raised platform, walk off a ledge, jump down and stomp, each right and
  left, plus stomping an enemy on a raised platform both ways and holding a
  spot on a moving platform), two enemy-bypass families, nine local route
  maneuvers, and ten older local families (flat runs, single gaps, enemy hops
  and stomps, pipe and stomp mounts, pit leaps, platform hops, landing on an
  enemy, holding an area).
- **Tactic: 45 families**, for strategy-dependent selection and termination:
  14 strategy families (seven tactics under each strategy), eight composed
  levels, four scene-driven route/response families, three alternate-route
  families, three waiting/proceeding families, and 13 families whose tactic
  changes partway through. These are excluded from skill training and
  evaluation.

Both learned stages sample their scene families; an optional parameter sweep
supports selected small skill families.

Legacy spatial-skill checkpoints warm-start by discarding the action network
and preserving skill, tactic, scene encoder, and memory weights. No action
training or action checkpoint is required.

See [the architecture and training guide](docs/layered-agent.md) for exact
inputs, outputs, timing, curriculum, qualification gates and known limits.

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
