# RetroAGI
General purpose machine learning agent for retro games.

RetroAGI trains an agent for Super Mario Bros that decides in four layers and
sees only through a vision transformer:

1. **Strategy** — what the run is for: finish fast, collect the most coins, or
   take the fewest risks.
2. **Tactic** — what to do in this part of the level: advance, take another
   route, hold the area, or retreat.
3. **Skill** — the next move: advance, jump a gap, climb, descend, stomp,
   retreat or wait, with its target.
4. **Action** — which buttons to press and for how many frames.

Each game has its own vision transformer (same model, its own weights). It
reads the screen and reports the objects on it: Mario, the ground and
platforms, gaps, pipes, blocks, coins and enemies. The four layers read only
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
  transformer. The strategy layer learns only here.

[docs/layered-agent.md](docs/layered-agent.md) explains what the agent sees,
how each layer decides and how it learns.

## Project Layout

```text
retroagi/
  core/
    vision.py, scene_vision.py   # the vision transformer, its training and measurements
    smb_pixel_types.py           # the pixel types and the objects built from them
    smb_scene_labels.py          # true objects for training labels
    smb_observer.py              # turns a screen into the report the layers read
    tokens.py                    # the strategy, tactic and skill vocabularies
    layered_policy.py            # the four decision layers and their memory
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
