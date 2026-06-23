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
assert manifest["game"]["name"] == "smb"
assert manifest["game_promotion"]["phases"][0]["name"] == "architecture-smoke"
assert manifest["game_promotion"]["phases"][0]["rung_gates"]["interface-smoke"]["runtime"]
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

## Progressive-Resolution Promotion Order

Treat the stages as a fidelity ladder with distinct responsibilities, not as
interchangeable training targets:

1. **Synthetic 1D fully validates the architecture.** It should catch tensor
   contract, hierarchy, objective, gradient, baseline, determinism, and
   checkpoint issues before the concept reaches a game-like stage.
2. **Block SMB trains the simplified game models.** Use this rung to train the
   Block ViT and the hierarchical actor/world-model/critic policy on the fast
   synthetic SMB environment, fixed scenarios, generated scenarios, and
   deterministic success thresholds.
3. **Full SMB asset-mock perception bootstraps the ViT.** Before any transferred
   policy uses emulator observations, train or fine-tune the Full SMB ViT on
   synthetic scenarios built from full-game assets and gate that checkpoint on
   held-out semantic and position metrics.
4. **Full SMB verifies inference and continues training.** First validate that
   transferred checkpoints run against the full emulator contract, then continue
   training the transferred models at full fidelity.

## 5. Run A Traceable Architecture Sweep

Use `retroagi experiment` when comparing architecture concepts across runnable
fidelity layers. The command below runs the baseline through Synthetic 1D
architecture validation and a tiny Block SMB simplified-game training smoke,
then writes one combined manifest with the resolved architecture, game profile,
backend/content metadata, per-stage commands, configs, metrics, checkpoints,
logs, gates, promotion decisions, and pass/fail status.

```bash
retroagi experiment \
  --stage synthetic-1d \
  --stage block-smb \
  --game smb \
  --output artifacts/repro/architecture_sweeps/baseline/manifest.json \
  --artifacts-dir artifacts/repro/architecture_sweeps/baseline \
  --seed 0 \
  --device cpu \
  --architecture baseline \
  --architecture-config hidden_dim=8 \
  --synthetic-epochs 1 \
  --synthetic-train-samples 16 \
  --synthetic-validation-samples 8 \
  --synthetic-test-samples 8 \
  --block-epochs 1 \
  --block-episodes-per-epoch 1 \
  --block-rollout-steps 2 \
  --block-evaluation-episodes 1 \
  --block-evaluation-max-steps 2 \
  --block-fixed-scenario level_1_flat.json \
  --gate 'synthetic-1d:controller_mse<100' \
  --gate 'block-smb:eval_success_rate>=0'
```

Verify the combined manifest before comparing it:

```bash
test -s artifacts/repro/architecture_sweeps/baseline/manifest.json
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("artifacts/repro/architecture_sweeps/baseline/manifest.json").read_text())
assert manifest["architecture"] == {
    "name": "agent_world_model_critic",
    "config": {"hidden_dim": 8},
}
assert manifest["seed"] == 0
assert manifest["device"] == "cpu"
assert manifest["game"]["name"] == "smb"
assert manifest["game"]["backend"]["name"] == "stable-retro"
assert any(item["name"] == "smb_rom" for item in manifest["game"]["content_identifiers"])
assert any(item["name"] == "smb_sprites" for item in manifest["game"]["asset_provenance"])
assert manifest["passed"] is True
assert {stage["stage"] for stage in manifest["stages"]} == {
    "synthetic-1d",
    "block-smb",
}
for stage in manifest["stages"]:
    assert stage["command"]
    assert stage["game_stage"]["name"] in {"synthetic", "block"}
    assert Path(stage["summary_path"]).name == "run_summary.json"
    assert Path(stage["checkpoint_path"]).name == "checkpoint.pth"
    assert isinstance(stage["recordings"], list)
    assert isinstance(stage["config"], dict)
    assert isinstance(stage["metrics"], dict)
    assert all(gate["passed"] for gate in stage["gates"])
assert {decision["status"] for decision in manifest["promotion_decisions"]} == {"passed"}
assert manifest["game_promotion"]["phases"][1]["rung_statuses"]["synthetic-concept"] == "passed"
assert manifest["gates"]
print("Architecture sweep manifest verified")
PY
```

Run the same command into a different artifacts directory for each architecture
or ablation variant. For example, add `--ablation critic=false` and use
`artifacts/repro/architecture_sweeps/no_critic/manifest.json` for a comparable
variant.

After two or more manifests exist, build a comparison report:

```bash
retroagi report \
  --input artifacts/repro/architecture_sweeps/baseline/manifest.json \
  --input artifacts/repro/architecture_sweeps/no_critic/manifest.json \
  --baseline-architecture agent_world_model_critic \
  --baseline-config hidden_dim=8 \
  --output artifacts/repro/architecture_sweeps/report.json
```

Expected report evidence:

- one row per stage for every input manifest,
- `game` and `game_key` on every run and report row,
- numeric metric deltas against the selected baseline when matching metrics are
  present, scoped to the same game and comparison row,
- artifact links back to each stage summary, checkpoint, and log,
- pass/fail gates carried through from the combined manifests.

For a multi-game profile smoke, keep each game under its own artifact root.
SMB is the runnable reference for synthetic and block stages; Pong currently
supports the synthetic experiment path as a second-game profile check while
its block and full runners remain planned.

```bash
retroagi experiment \
  --game pong \
  --stage synthetic \
  --output artifacts/repro/multi_game/pong/baseline/manifest.json \
  --artifacts-dir artifacts/repro/multi_game/pong/baseline \
  --seed 0 \
  --device cpu \
  --architecture baseline \
  --architecture-config hidden_dim=8 \
  --synthetic-epochs 1 \
  --synthetic-train-samples 16 \
  --synthetic-validation-samples 8 \
  --synthetic-test-samples 8 \
  --gate 'synthetic:controller_mse<100'
```

Verify the Pong profile manifest:

```bash
test -s artifacts/repro/multi_game/pong/baseline/manifest.json
python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("artifacts/repro/multi_game/pong/baseline/manifest.json").read_text())
assert manifest["passed"] is True
assert manifest["game"]["name"] == "pong"
assert manifest["game"]["backend"]["contract"]["provider_kind"] == "gymnasium"
assert [stage["name"] for stage in manifest["game"]["stage_ladder"]] == [
    "synthetic",
    "block",
    "full",
]
assert manifest["game"]["content_identifiers"] == []
assert manifest["stages"][0]["game_stage"]["name"] == "synthetic"
assert manifest["promotion_decisions"][0]["status"] == "passed"
print("Pong profile manifest verified")
PY
```

Create a cross-game report once both SMB and Pong manifests exist:

```bash
retroagi report \
  --input artifacts/repro/architecture_sweeps/baseline/manifest.json \
  --input artifacts/repro/multi_game/pong/baseline/manifest.json \
  --baseline-architecture agent_world_model_critic \
  --baseline-config hidden_dim=8 \
  --output artifacts/repro/multi_game/report.json
```

Verify the report preserves per-game grouping:

```bash
test -s artifacts/repro/multi_game/report.json
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("artifacts/repro/multi_game/report.json").read_text())
assert report["summary"]["game_count"] >= 2
assert "smb" in report["summary"]["game_row_counts"]
assert "pong" in report["summary"]["game_row_counts"]
assert all("game_key" in row for row in report["rows"])
print("Multi-game report verified")
PY
```

## 6. Run A Traceable CPU Smoke Training

Use the same smoke shape as CI. It is intentionally tiny; the purpose is to
prove the CLI, configuration capture, structured logs, and deterministic
evaluation path are wired correctly.

```bash
retroagi train --game smb --stage block \
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

## 7. Reproduce Synthetic 1D

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

## 8. Reproduce Block SMB Perception

First verify the tracked or supplied Block ViT checkpoint:

```bash
retroagi diagnose-vision --game smb --stage block \
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

## 9. Reproduce Block SMB Policy

Run a traceable policy training job with an explicit seed, checkpoint,
structured log, and run summary:

```bash
retroagi train --game smb --stage block \
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
retroagi resume --game smb --stage block \
  --checkpoint data/block_smb/policy.pth \
  --save-checkpoint data/block_smb/policy.pth \
  --epochs 10 \
  --output artifacts/block_smb/latest/resume_summary.json \
  --log-path artifacts/block_smb/latest/resume_events.jsonl
```

Evaluate against the fixed-scenario threshold protocol:

```bash
retroagi evaluate --game smb --stage block \
  --checkpoint data/block_smb/policy.pth \
  --evaluation-episodes 3 \
  --evaluation-max-steps 200 \
  --output artifacts/block_smb/latest/evaluation.json
```

Record deterministic evaluation artifacts:

```bash
retroagi record --game smb --stage block \
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

## 10. Reproduce Full SMB Vision

Generate full-game assets and synthetic scenarios, then train the Full SMB ViT
segmenter. This is the required perception bootstrap between Block SMB training
and Full SMB policy inference/training:

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
scenes. Do not promote a Block SMB policy to Full SMB inference or continued
training until the Full SMB ViT checkpoint has passed the selected held-out
semantic and position gates.

## 11. Set Up Full SMB Local Content

Full SMB headless training and evaluation require the local stable-retro content
setup described in [full-smb-content.md](full-smb-content.md). The supported
game id is `SuperMarioBros-Nes`; ROM content must be legally obtained,
user-provided, local-only, and excluded from git.

```bash
python -m pip install -e '.[full-smb]'
mkdir -p local/full_smb/roms local/full_smb/checksums
python -m retro.import local/full_smb/roms
shasum -a 256 local/full_smb/roms/<your-rom-file>.nes \
  > local/full_smb/checksums/SuperMarioBros-Nes.sha256
```

Expected local-only evidence:

- `local/full_smb/roms/` exists and is ignored by git,
- stable-retro can resolve `SuperMarioBros-Nes`,
- `local/full_smb/checksums/SuperMarioBros-Nes.sha256` records the SHA-256 hash
  for the imported ROM,
- run notes or `artifacts/full_smb/<run>/content.json` record the game id,
  import command, checksum algorithm, checksum file path, and provenance note
  without copying ROM bytes.

If the backend or imported game is missing, `FullSMBStage()` raises a setup
error that includes the install command, `python -m retro.import
local/full_smb/roms`, checksum path, and legal/provenance reminder.

Run the Full SMB environment capability check:

```bash
retroagi check-env --game smb --stage full \
  --seed 0 \
  --steps 4 \
  --frame-skip 2 \
  --output artifacts/full_smb/env_check.json
```

Verify the report before launching headless training:

```bash
test -s artifacts/full_smb/env_check.json
python - <<'PY'
import json
from pathlib import Path

report = json.loads(Path("artifacts/full_smb/env_check.json").read_text())
assert report["passed"] is True
for check in (
    "backend_import",
    "game_registration",
    "rom_availability",
    "headless_reset",
    "render_reset",
    "save_load_state",
    "action_step",
    "frame_skip",
    "deterministic_seeding",
):
    assert report["checks"][check]["passed"], check
print("Full SMB environment capability check verified")
PY
```

Inspect the Full SMB task catalog before choosing training or evaluation tasks:

```bash
python - <<'PY'
from retroagi.stages.full_smb import full_smb_task_catalog

catalog = full_smb_task_catalog()
assert [task.name for task in catalog.tasks_for_set("smoke")] == ["smoke_1_1_spawn"]
assert all(task.split == "train" for task in catalog.curriculum)
assert all(
    task.split == "heldout"
    for task in catalog.tasks_for_set("heldout_generalization")
)
assert catalog.save_state_artifact_paths
print(catalog.to_manifest()["task_sets"].keys())
PY
```

The catalog is documented in [full-smb-tasks.md](full-smb-tasks.md). Use
`smoke` after `check-env`, train on the ordered `curriculum` tasks, evaluate
on `fixed_benchmark`, and reserve `heldout_generalization` for promotion and
regression reports.

Generate the local save-state recipe manifest and local-only `.state` files:

```bash
python -m retroagi.stages.full_smb.save_states plan \
  --output local/full_smb/states/save_state_plan.json

python -m retroagi.stages.full_smb.save_states create \
  --output-manifest local/full_smb/states/save_state_manifest.json \
  --overwrite
```

Verify that the plan is recipe-only and that generated artifacts stay in the
ignored local content tree:

```bash
python - <<'PY'
import json
from pathlib import Path

plan = json.loads(Path("local/full_smb/states/save_state_plan.json").read_text())
manifest = json.loads(
    Path("local/full_smb/states/save_state_manifest.json").read_text()
)
assert plan["copyrighted_content_committed"] is False
assert all(
    item["path"].startswith("local/full_smb/states/")
    for item in plan["artifacts"]
)
assert all(result["bytes_written"] > 0 for result in manifest["results"])
print("Full SMB local save-state artifacts verified")
PY
```

The save-state workflow is documented in
[full-smb-save-states.md](full-smb-save-states.md). Do not commit the generated
`.state` files or ROM-derived screenshots.

## 12. Reproduce Full SMB Adapter, Inference, And Continued Training

Run the headless emulator smoke path to verify the full observation and action
contract:

```bash
retroagi evaluate --game smb --stage full \
  --steps 500 \
  --seed 0 \
  --encode-observations
```

Transfer the Block SMB policy into the Full SMB contract only after the Full SMB
ViT asset-synthetic checkpoint exists:

```bash
retroagi transfer --game smb --stage full \
  --block-policy-checkpoint data/block_smb/policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint data/full_smb/transferred_policy.pth
```

Compare the transferred policy against a scratch baseline on the same seeded
observation stream, then use the Full SMB training command to continue learning
from the transferred checkpoint when the emulator setup is available:

```bash
retroagi compare --game smb --stage full \
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

## 13. Preserve The Run

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
