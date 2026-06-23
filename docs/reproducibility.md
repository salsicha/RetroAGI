# Reproducibility Procedure

This procedure starts from a clean checkout and produces a verified local run
with recorded configuration, runtime metadata, metrics, logs, and artifact
paths. It is the minimum repeatable path before comparing experiments or
claiming a checkpoint is known-good.

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

The `full-smb` extra installs stable-retro for real emulator runs and is not
required for unit tests, Block SMB training, or CI smoke training. Install it
only before running Full SMB against a local ROM setup:

```bash
python -m pip install -e '.[full-smb]'
```

Verify the selected runtime:

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.backends.mps.is_available())'
python -c 'import pygame; print(pygame.version.ver)'
```

The `pygame` import must resolve to pygame-ce. If it does not, uninstall the
old `pygame` package and reinstall the project dependencies.

## 3. Run The Baseline Test Suite

```bash
python -m unittest discover -s scripts/tests -v
```

The full suite is the acceptance gate for a clean checkout. For a faster
preflight while debugging environment setup, run:

```bash
python -m unittest -v scripts.tests.test_devices scripts.tests.test_docs scripts.tests.test_synthetic_1d
```

Do not compare training results from a checkout that cannot pass the full
suite.

## 4. Run The Baseline Architecture Promotion Fixture

CI runs the smallest known-good architecture sweep fixture against the baseline
architecture. It verifies the promotion pipeline can instantiate the baseline
model across the Synthetic 1D, Block SMB, and Full SMB `StageSpec` contracts,
run a forward/backward smoke pass, and write a traceable manifest.

```bash
retroagi promote \
  --rung interface-smoke \
  --output artifacts/repro/promotion_baseline_interface.json \
  --artifacts-dir artifacts/repro/promotion_baseline \
  --device cpu \
  --architecture baseline \
  --architecture-config hidden_dim=8 \
  --interface-batch-size 1
```

Verify the promotion fixture:

```bash
test -s artifacts/repro/promotion_baseline_interface.json
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("artifacts/repro/promotion_baseline_interface.json").read_text())
assert manifest["passed"] is True
assert manifest["architecture"]["name"] == "agent_world_model_critic"
assert manifest["architecture"]["config"] == {"hidden_dim": 8}
assert [rung["name"] for rung in manifest["rungs"]] == ["interface-smoke"]
assert manifest["rungs"][0]["status"] == "passed"
assert {stage["stage"] for stage in manifest["rungs"][0]["stages"]} == {
    "synthetic_1d",
    "block_smb",
    "full_smb",
}
print("Promotion fixture verified")
PY
```

This fixture is intentionally cheaper than policy training. It is the minimum
architecture-sweep gate before spending time on the synthetic, Block SMB, and
Full SMB training rungs.

## 5. Run A Traceable CPU Smoke Training

Use the same smoke shape as CI. It is intentionally tiny; the purpose is to
prove the CLI, configuration capture, structured logs, and deterministic
evaluation path are wired correctly.

```bash
retroagi train --stage block-smb \
  --epochs 1 \
  --episodes-per-epoch 1 \
  --rollout-steps 2 \
  --fixed-scenario level_1_flat.json \
  --generated-scenarios 0 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 2 \
  --evaluation-interval-epochs 1 \
  --hidden-dim 8 \
  --device cpu \
  --disable-checkpoint-transfer \
  --output artifacts/repro/block_smb_smoke/run_summary.json \
  --log-path artifacts/repro/block_smb_smoke/events.jsonl
```

Verify the smoke artifacts:

```bash
test -s artifacts/repro/block_smb_smoke/run_summary.json
test -s artifacts/repro/block_smb_smoke/events.jsonl
python -m json.tool artifacts/repro/block_smb_smoke/run_summary.json >/dev/null
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("artifacts/repro/block_smb_smoke/run_summary.json").read_text())
events = [
    json.loads(line)
    for line in Path("artifacts/repro/block_smb_smoke/events.jsonl").read_text().splitlines()
]
assert summary["config"]["seed"] == 0
assert summary["config"]["device"] == "cpu"
assert summary["config"]["ablation"]["checkpoint_transfer_enabled"] is False
assert summary["evaluation"]["fixed_scenarios"]
assert any(event["event"] == "run_started" for event in events)
assert any(event["event"] == "run_finished" for event in events)
print("Smoke artifacts verified")
PY
```

## 6. Reproduce Synthetic 1D

```bash
python -m retroagi.stages.synthetic_1d.train
python -m unittest -v scripts.tests.test_synthetic_1d
```

Expected evidence:

- the learned model beats the seeded random and simple baselines on
  `controller_mse`,
- gradients stay finite,
- deterministic split seeds and train permutations are covered by tests,
- checkpoint save, restore, and resume behavior are covered by tests.

## 7. Reproduce Block SMB Perception

First verify the tracked or supplied Block ViT checkpoint:

```bash
retroagi diagnose-vision --stage block-smb \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --samples 64 \
  --rollout-steps 32 \
  --output artifacts/repro/block_smb_vision_diagnostic.json
```

The diagnostic output must include `perception.bottleneck`. A known-good
perception checkpoint should keep `bottleneck` false under the default
thresholds from `BlockVITPerceptionThresholds`.

To retrain perception from procedural labels:

```bash
python scripts/vit/train_block_vit.py \
  --epochs 20 \
  --samples-per-epoch 2048 \
  --val-samples 512 \
  --device auto \
  --output data/block_vit/block_vit.pth
```

Preserve `data/block_vit/block_vit.pth`,
`data/block_vit/block_vit.json`, and the diagnostic JSON.

## 8. Reproduce Block SMB Policy

Run a traceable policy training job with an explicit seed, checkpoint,
structured log, and run summary:

```bash
retroagi train --stage block-smb \
  --seed 0 \
  --epochs 5 \
  --episodes-per-epoch 2 \
  --rollout-steps 32 \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --checkpoint data/block_smb/policy.pth \
  --output artifacts/block_smb/latest/run_summary.json \
  --log-path artifacts/block_smb/latest/events.jsonl
```

Resume from the same checkpoint to verify continuity:

```bash
retroagi resume --stage block-smb \
  --checkpoint data/block_smb/policy.pth \
  --save-checkpoint data/block_smb/policy.pth \
  --epochs 10 \
  --output artifacts/block_smb/latest/resume_summary.json \
  --log-path artifacts/block_smb/latest/resume_events.jsonl
```

Evaluate against the fixed-scenario threshold protocol:

```bash
retroagi evaluate --stage block-smb \
  --checkpoint data/block_smb/policy.pth \
  --evaluation-episodes 3 \
  --evaluation-max-steps 200 \
  --output artifacts/block_smb/latest/evaluation.json
```

Record deterministic evaluation artifacts:

```bash
retroagi record --stage block-smb \
  --checkpoint data/block_smb/policy.pth \
  --evaluation-episodes 3 \
  --evaluation-max-steps 200 \
  --record-dir artifacts/block_smb/latest/recordings \
  --output artifacts/block_smb/latest/recording_summary.json
```

Expected evidence:

- `data/block_smb/policy.pth` and `data/block_smb/policy.json`,
- resolved configs in all summary JSON files,
- structured `run_started`, `train_epoch`, `deterministic_evaluation`, and
  `run_finished` events,
- `evaluation.success_thresholds_met` and per-scenario `threshold_met`
  diagnostics,
- recording `.npz` files under the record directory.

The scripted known-good baseline at
`artifacts/block_smb/known_good_scripted_seed20260622/` is a regression
baseline for environment and threshold behavior. It is not a substitute for a
learned policy checkpoint.

## 9. Reproduce Full SMB Vision

Generate assets and synthetic data, then train the Full SMB ViT segmenter:

```bash
python scripts/vit/extract_sprites.py
python scripts/vit/generate_dataset.py --train 5000 --val 1000
python scripts/vit/train_vit.py \
  --epochs 30 \
  --batch 64 \
  --dim 192 \
  --depth 6 \
  --device auto
```

Expected artifacts:

- `assets/spritesheets/` and `assets/sprites/`,
- `data/vit/train.npz` and `data/vit/val.npz`,
- `data/vit/full_smb_vit.pth` and `data/vit/full_smb_vit.json`,
- `data/vit/vit_smb.pth`,
- `data/vit/predictions.png`.

The reference target is about 99.94 percent overall accuracy, 99.89 percent
foreground accuracy, and 99.14 percent mean IoU on 1,000 held-out synthetic
scenes.

## 10. Reproduce Full SMB Adapter And Transfer

Run the headless emulator smoke path:

```bash
retroagi evaluate --stage full-smb \
  --steps 500 \
  --seed 0 \
  --encode-observations
```

Transfer the Block SMB policy into the Full SMB contract:

```bash
retroagi transfer --stage full-smb \
  --block-policy-checkpoint data/block_smb/policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint data/full_smb/transferred_policy.pth
```

Compare the transferred policy against a scratch baseline on the same seeded
observation stream:

```bash
retroagi compare --stage full-smb \
  --transfer-checkpoint data/full_smb/transferred_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --steps 128 \
  --seed 0 \
  --scratch-seed 0 \
  --output artifacts/full_smb/transfer_vs_scratch.json
```

Expected evidence:

- `data/full_smb/transferred_policy.pth`,
- `data/full_smb/transferred_policy.json`,
- `artifacts/full_smb/transfer_vs_scratch.json`,
- comparison fields including `action_agreement`, action histograms,
  mean entropies, mean margins, collection reward, resets, terminations, and
  truncations.

## 11. Preserve The Run

Before publishing or comparing results, preserve:

1. Git commit: `git rev-parse HEAD`.
2. Runtime printout from the PyTorch and pygame-ce verification commands.
3. Full test command and result.
4. Training, evaluation, and recording commands.
5. Every `*.pth` checkpoint plus its sidecar `*.json`.
6. Run summaries, structured event logs, diagnostics, comparison JSON, and
   recording directories.
7. Optional TensorBoard or Weights & Biases run location when
   `--tracking-backend` is enabled.

The reproducibility claim is valid only for the exact commit, dependency set,
seed, config, and artifacts listed in the preserved run record.
