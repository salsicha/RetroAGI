# Reproducibility Procedure

This procedure goes from a clean checkout to a trained layered agent, and
records what is needed to repeat the run.

## 1. Start From A Clean Checkout

```bash
git clone https://github.com/salsicha/RetroAGI.git
cd RetroAGI
git status --short
git rev-parse HEAD
```

`git status --short` must be empty. Record the `git rev-parse HEAD` value in
any experiment note or issue.

## 2. Create A Supported Environment

Use one Python 3.12, 3.13, or 3.14 environment. Python 3.14 is the CI default.
Install exactly one PyTorch wheel variant for the target machine.

CPU-only:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  torch==2.9.1 torchvision==0.24.1 \
  --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test,vision]'
```

macOS Apple Silicon:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 torchvision==0.24.1
python -m pip install -e '.[test,vision]'
```

CUDA users should use the CUDA 12.8 or CUDA 13.0 install commands in
[compatibility.md](compatibility.md), then install `.[test,vision]`.

The `full-smb` extra installs stable-retro for the emulator. It is needed for
the Full SMB vision transformer and for playing Full SMB, not for Block SMB:

```bash
python -m pip install -e '.[full-smb]'
```

Then import the game as [full-smb-content.md](full-smb-content.md) describes.

Verify the selected runtime:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.backends.mps.is_available())'
```

## 3. Run The Test Suite

```bash
python -m pytest scripts/tests
```

## 4. Train The Vision Transformers

Block SMB frames come from every layout family, played with teacher, perturbed,
delayed and random routes; their true labels are drawn exactly by the
simulator. Full SMB frames come from the training levels in the emulator; their
true labels are read from game memory, and a frame is used only when memory
explains every visible pixel.

```bash
python scripts/vit/train_block_vit.py --epochs 40 --samples-per-epoch 40000
python scripts/vision/evaluate_block_vision.py
python scripts/vit/train_full_vit.py --epochs 40 --samples-per-epoch 40000
python scripts/vision/evaluate_full_vision.py
```

The checkpoints are `data/block_vit/block_vit_scene.pth` and
`data/full_vit/full_vit_scene.pth`, each with a `.json` summary beside it that
records the settings, measurements, code revision and runtime. The Full SMB
measurements use the test levels `Level1-1` and `Level5-1`, which the model
never trains on.

## 5. Train The Layers In Block SMB

Skill trains first, then tactic, each from the layer below as it is (no
gate stops what follows; user, 2026-10-18). In Block SMB training the
teacher carries out the skill's actions; the world model (the policy's
LSTM) and the collision trackers learn alongside from the same teacher
episodes, and apart from the saved datasets. Every round's episodes are
saved one file per family (`episodes/round_NN/`) and read back instead of
played again; `--generate-only true` makes and saves a dataset without
learning.

```bash
python -m retroagi.stages.block_smb.collision_train --source <policy> --output artifacts/block_smb/collision
retroagi-block-smb train-layer --learner skill --init <policy> --output artifacts/block_smb/skill
retroagi-block-smb train-layer --learner tactic --init artifacts/block_smb/skill/best.pt --output artifacts/block_smb/tactic
python -m retroagi.stages.block_smb.dataset_train --source artifacts/block_smb/skill/best.pt --tracker artifacts/block_smb/collision/tracker.pt --rounds artifacts/block_smb/skill/episodes artifacts/block_smb/tactic/episodes --collision artifacts/block_smb/collision/dataset --output artifacts/block_smb/predictions
```

- **Skill layer:** trains on 41 local maneuver families, each under one
  supplied tactic, learning actions (a verb aimed at an object) from vision,
  the executor's motion estimates, memory, tactic and its previous 16
  actions. All tactic-training families are excluded. The collision
  trackers first learn from the collision lessons (`collision_train`: every
  way Mario and an enemy touch in one action, and misses), which train
  nothing else; `dataset_train` then trains the world model and the
  trackers from every saved dataset.
- **Tactic layer:** trains on 42 families: 14 strategy families, eight composed
  levels, four scene-driven route/response families, three alternate-route
  families, three waiting/proceeding families and 10 families whose tactic
  changes partway through, with layouts drawn at random. The complete grouping is documented
  in [the layered agent guide](layered-agent.md).

Example larger runs add `--validation-layouts-per-difficulty 6 --workers 15` to
each, and `--reward-rounds 4 --train-layouts-per-family 32` to the tactic
layer. `retroagi-block-smb train-layer --help` lists every setting.

Each run folder holds `history.json` (every round's
scores per family), `last.pt`, `best.pt` and, once the bar is met, `passed.pt`.
Every checkpoint records its settings, the layers it trained, and the vision
checkpoint it was trained with (with that file's SHA-256).

## 6. Examine The Agent

```bash
retroagi-block-smb exam-layer --checkpoint artifacts/block_smb/tactic/passed.pt
python -m retroagi.stages.full_smb.layered_eval --checkpoint artifacts/block_smb/tactic/passed.pt
```

The first plays fresh held-out Block SMB layouts with the whole agent; the
second plays the Full SMB test levels and reports how far Mario got.

## 7. Preserve The Run

Keep, for each run: the commit hash, the exact commands, the run folders under
`artifacts/`, and the vision checkpoints with their `.json` summaries. Never
commit or upload the game ROM.
