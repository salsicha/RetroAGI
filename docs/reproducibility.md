# Reproducibility Procedure

This procedure goes from a clean checkout to a trained agent scored on the
real game, and records what is needed to repeat the run.

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

## 4. Train The Vision Network

The real game's vision network learns from its frames, labelled from its
memory: a frame is used only when memory explains every visible pixel. It
trains on the levels set aside for building Block SMB and is measured on
levels it never trains on. (The Block SMB network is only for measuring
seeing on plain blocks; the agent never uses vision in Block SMB.)

```bash
python scripts/vit/train_full_vit.py --epochs 40 --samples-per-epoch 40000
python scripts/vision/evaluate_full_vision.py
python scripts/vit/train_block_vit.py --epochs 40 --samples-per-epoch 40000
python scripts/vision/evaluate_block_vision.py
```

The checkpoints are `data/full_vit/full_vit_scene.pth` and
`data/block_vit/block_vit_scene.pth`, each with a `.json` summary beside it
that records the settings, measurements, code revision and runtime.
[scripts/vit/README.md](../scripts/vit/README.md) describes both.

## 5. Learn The Deciding Network In Block SMB

One command learns the outcome network from nothing, in Block SMB alone,
round after round. Each round:

1. plays fresh random levels and skill scenes (in round 1 with the
   hand-written chooser, after that with the network of the round before);
2. at each decision saves the world, plays every candidate choice from the
   save with the controller, and records how each ended;
3. trains the network on all the rounds' outcomes so far;
4. examines it on Block SMB worlds it never trained on.

```bash
scripts/learn_in_block_smb.sh NAME
```

Its settings come from the environment (`ROUNDS`, `LEVELS`, `SCENES`,
`STEPS`, the network's size, `WORKERS`); the script's opening comment lists
them. Three rounds take about eight hours on the reference machine. The
last round's network is `artifacts/block_smb/outcome_NAME_3/outcome.pt`.
The steps can also be run one at a time:

```bash
python -m retroagi.framework.sampling --domain smb --levels 2000 --workers 14 --output data/datasets/block_smb/outcomes_NEW
python -m retroagi.framework.training --domain smb --data data/datasets/block_smb/outcomes_NEW --fresh data/datasets/block_smb/outcomes_fresh_NEW --output artifacts/block_smb/outcome_NEW
python -m retroagi.framework.practice --domain smb --levels 100 --first-seed 100000 --model artifacts/block_smb/outcome_NEW/outcome.pt
```

## 6. Score It On The Real Game

Nothing learns here. Each obstacle of a real level is played from a saved
state just before it, from the screen (`--source vision`) or, as a
measuring device, with exact positions read from memory (`--source
memory`); whole levels are played from their start. `scripts/dev/scorecard.py`
gathers the held-out levels' scores into one table.

```bash
python -m retroagi.framework.bench --domain smb --bench data/full_smb_obstacles/Level1-1 --source vision --model artifacts/block_smb/outcome_NEW/outcome.pt
python -m retroagi.framework.play --domain smb --level Level1-1 --source vision --model artifacts/block_smb/outcome_NEW/outcome.pt
```

## 7. Preserve The Run

Keep, for each run: the commit hash, the exact commands, the run folders under
`artifacts/`, the sampled outcomes under `data/datasets/block_smb/`, and the
vision checkpoints with their `.json` summaries. Never
commit or upload the game ROM.
