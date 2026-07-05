# Operations Reference

This document is the stage-by-stage operations checklist for hardware,
runtime, metrics, and artifacts. It complements the compatibility matrix and
the stage contract docs; use it when deciding whether a run is ready to compare
or preserve.

## Runtime Baseline

Supported environments are defined in
[compatibility.md](compatibility.md): Python 3.12 through 3.14, PyTorch 2.9.1,
Linux x86-64, and macOS Apple Silicon. CPU execution is the required baseline
for tests and smoke runs. `retroagi.core.select_device` chooses CUDA first,
then Apple MPS, then CPU when a stage uses `--device auto`.

Use accelerated hardware for full training runs when available:

| Backend | Use For | Notes |
| --- | --- | --- |
| CPU | Unit tests, smoke training, deterministic debugging | Required to stay functional for all stages. |
| CUDA | Linux GPU policy and vision training | Select with `--device cuda` or `--device auto` on a CUDA host. |
| Apple MPS | macOS Apple Silicon policy and vision training | Select with `--device mps` or `--device auto` on an MPS host. |

Record the exact device, seed, resolved config, code revision, checkpoint path,
and metric payload with every non-smoke experiment. Versioned checkpoints write
a `.json` sidecar beside the `.pth` checkpoint with this trace metadata.

The top-level `retroagi` CLI separates game selection from fidelity-rung
selection. Use `--game smb --stage synthetic|block|full` for the game-neutral
path; legacy SMB-specific stage aliases such as `synthetic-1d`, `block-smb`,
and `full-smb` remain accepted for existing scripts.

The registry also includes a proof-of-concept `pong` profile with documented
`synthetic -> block -> full` rungs, game-owned actions, rewards, tasks, backend
metadata, and non-asset perception sources. Pong is currently a profile and
planning contract; top-level runtime commands still dispatch only the SMB
runners until Pong stage implementations are added.

Each game profile declares a backend provider contract through
`GameBackendSpec`. Experiment manifests write the backend name, installed
version when available, provider kind, entrypoint, reset/step/state API notes,
and capability flags. Use that manifest contract when comparing runs across
`stable-retro`, Gymnasium-compatible wrappers, native Python simulators, or
custom game adapters.

Use `probe_backend_capabilities(...)` from `retroagi.core` when bringing up a
new backend adapter. It exercises deterministic seeded reset, save/load-state
replay, single-step advancement, repeated action replay, rendering, and
headless reset/step behavior against the adapter's normalized API.

Before committing or preserving asset-backed artifacts, check the selected
game's `asset_checklist` in the experiment manifest. For SMB this means sprite
source/license evidence before committing extracted sprites or asset-mock
datasets, local-only ROM handling before Full SMB emulator runs, and generator
config/seed/source-asset provenance before referencing generated datasets from
checkpoints.

When a game does not have a reliable sprite or asset pipeline, declare
non-asset `PerceptionDatasetSourceSpec` entries in its perception pipeline:
`self_supervised` for contrastive or predictive frame objectives,
`emulator_state` for backend object-state labels, and `manual_labels` for human
semantic masks. These source entries are written to experiment manifests so
promotion reports can distinguish asset-backed perception from fallback
supervision.

## Multi-Game Operations

Use the game registry as the source of truth before launching a run:

```bash
python - <<'PY'
from retroagi.core import game_plugin_names, get_game_plugin

for name in game_plugin_names():
    plugin = get_game_plugin(name)
    stages = ", ".join(stage.name for stage in plugin.game.stage_ladder)
    print(f"{name}: {stages}")
PY
```

SMB is the fully runnable reference game for synthetic, block, and full-fidelity
commands. Pong is the proof-of-concept second game profile. It can produce a
traceable synthetic experiment manifest today; its block and full runtime
runners remain planned contracts until the Pong stage adapters are implemented.

Run the SMB reference sweep into a game-scoped artifacts directory:

```bash
retroagi experiment \
  --game smb \
  --stage synthetic \
  --stage block \
  --output artifacts/multi_game/smb/baseline/manifest.json \
  --artifacts-dir artifacts/multi_game/smb/baseline \
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
  --gate 'synthetic:controller_mse<100' \
  --gate 'block:eval_success_rate>=0'
```

Run the Pong profile smoke into its own game-scoped directory:

```bash
retroagi experiment \
  --game pong \
  --stage synthetic \
  --output artifacts/multi_game/pong/baseline/manifest.json \
  --artifacts-dir artifacts/multi_game/pong/baseline \
  --device cpu \
  --architecture baseline \
  --architecture-config hidden_dim=8 \
  --synthetic-epochs 1 \
  --synthetic-train-samples 16 \
  --synthetic-validation-samples 8 \
  --synthetic-test-samples 8 \
  --gate 'synthetic:controller_mse<100'
```

Use this artifact layout for comparable multi-game work:

| Path | Contents |
| --- | --- |
| `artifacts/multi_game/<game>/<architecture>/manifest.json` | Experiment manifest for one architecture/game run. |
| `artifacts/multi_game/<game>/<architecture>/<stage>/run_summary.json` | Stage summary emitted by the stage runner. |
| `artifacts/multi_game/<game>/<architecture>/<stage>/checkpoint.pth` | Stage checkpoint when the runner writes one. |
| `artifacts/multi_game/<game>/<architecture>/<stage>/events.jsonl` | Structured log for stages that emit events. |
| `artifacts/multi_game/reports/<name>.json` | Cross-game or per-game comparison report. |

Build a report from comparable manifests:

```bash
retroagi report \
  --input artifacts/multi_game/smb/baseline/manifest.json \
  --input artifacts/multi_game/pong/baseline/manifest.json \
  --baseline-architecture agent_world_model_critic \
  --baseline-config hidden_dim=8 \
  --output artifacts/multi_game/reports/baseline_cross_game.json
```

Report rows include `game`, `game_key`, architecture, stage/rung, metrics,
gates, artifacts, and regression deltas. Deltas are scoped by both game and
comparison row, so a Pong metric is compared to a Pong baseline rather than an
SMB baseline with the same architecture config.

## Progressive-Resolution Responsibilities

Use the stages in this order when evaluating an architecture concept:

1. **Synthetic 1D validates the architecture.** It is the proof stage for
   tensor contracts, hierarchy behavior, objective terms, gradients,
   deterministic baselines, checkpoint save/restore, and held-out metrics. Do
   not use Block SMB to discover basic architecture-contract failures.
2. **Block SMB trains all simplified game models.** It is the fast synthetic SMB
   world where Block ViT perception and the hierarchical
   actor/world-model/critic policy are trained, ablated, resumed, recorded, and
   measured against fixed and generated scenarios.
3. **Full SMB asset-mock perception bootstraps the ViT.** Before policy
   inference depends on emulator frames, train or fine-tune the Full SMB ViT on
   synthetic scenarios composed from full-game assets and gate the checkpoint on
   held-out semantic and position metrics.
4. **Full SMB verifies inference and continues training.** The emulator stage
   first validates transferred-model inference against the full observation and
   action contracts, then continues training the transferred policy/model stack
   at full fidelity.

## Synthetic 1D

Synthetic 1D is the full architecture-validation stage. It validates the shared
actor/world-model/critic stack without game physics or perception and should
catch architecture-contract, objective, gradient, baseline, and checkpoint
failures before a concept reaches any game-like environment. It also validates
the Block SMB primitive-control contract on perfectly known data: button combo,
hold duration, release, cancel, replan, post-release action, hazard-window
timing, and k-step primitive outcomes for the LSTM world model.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU is sufficient and required. CUDA or MPS may be used through `SyntheticTrainingConfig.device`, but GPU is not required for acceptance. |
| Runtime | The focused Synthetic 1D test suite is expected to finish in under 20 seconds on the current development machine. The baseline held-out training demonstration is expected to run in about 7 seconds. |
| Command | `python -m retroagi.stages.synthetic_1d.train` or `retroagi train --game smb --stage synthetic`. |
| Expected Metrics | `controller_mse`, `controller_mae`, `controller_rmse`, `error_B`, and `accuracy_A`. Training history also records `loss_primitive_labels` and `loss_primitive_outcome`. A passing learned policy beats both the seeded random and train-marginal simple baselines, with the tested margin at least 25 percent below the simple baseline on `controller_mse`. |
| Artifact Locations | No artifact is written by the default CLI path. Programmatic runs that set `SyntheticTrainingConfig(save_checkpoints=True, checkpoint_path=...)` write the versioned checkpoint and sidecar at that checkpoint path. Use `artifacts/synthetic_1d/<run>/` for local run summaries when preserving experiments. |

Validation command:

```bash
python -m unittest -v scripts.tests.test_synthetic_1d
```

## Block SMB Perception

Block SMB perception trains and validates `data/block_vit/block_vit.pth` from
the simplified synthetic game. It is part of training all game-facing models in
the Block SMB rung and is the frozen visual encoder used by Block SMB policy
training unless a run explicitly opts into fine-tuning.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU works for diagnostics and small training runs. CUDA or Apple MPS is recommended for full default training. |
| Runtime | Perception diagnostics are intended as short preflight checks. Full default training collects procedural rollouts every epoch, so runtime scales with `--epochs`, `--samples-per-epoch`, and backend speed. |
| Train Command | `python scripts/vit/train_block_vit.py --epochs 20 --samples-per-epoch 2048 --val-samples 512 --device auto`. |
| Diagnostic Command | `retroagi diagnose-vision --game smb --stage block --vision-checkpoint data/block_vit/block_vit.pth --samples 64 --rollout-steps 32`. |
| Expected Metrics | Training writes `loss`, `semantic_loss`, `position_loss`, `support_loss`, `accuracy`, `foreground_accuracy`, `mean_iou`, and `support_accuracy`. Diagnostics add `position_rmse`, `position_within_tolerance`, `thresholds`, `bottleneck`, and `bottleneck_reasons`. The default diagnostic thresholds are at least 0.95 accuracy, 0.90 foreground accuracy, 0.70 mean IoU, 0.90 support accuracy, at most 0.06 position RMSE, and at least 0.90 position within tolerance. |
| Artifact Locations | Default checkpoint: `data/block_vit/block_vit.pth`. Its sidecar is `data/block_vit/block_vit.json`. Diagnostic summaries should be written with `--output artifacts/block_smb/vision_diagnostic.json` when preserving a run. |

## Block SMB Policy

Block SMB policy training is the main trainable simplified-game stage. It uses
the fixed scenarios plus optional generated scenarios and reports
deterministic evaluation against the fixed-scenario success thresholds.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU is valid for smoke training and deterministic evaluation. CUDA or Apple MPS is recommended for real policy sweeps. |
| Runtime | CI smoke training uses tiny CPU settings. Real runs scale with `--epochs`, `--episodes-per-epoch`, `--rollout-steps`, generated scenarios, evaluation cadence, and whether recording is enabled. |
| Train Command | `retroagi train --game smb --stage block --epochs 5 --vision-checkpoint data/block_vit/block_vit.pth --checkpoint data/block_smb/policy.pth --output artifacts/block_smb/latest/run_summary.json --log-path artifacts/block_smb/latest/events.jsonl`. |
| Resume Command | `retroagi resume --game smb --stage block --checkpoint data/block_smb/policy.pth --epochs 10`. |
| Evaluation Command | `retroagi evaluate --game smb --stage block --checkpoint data/block_smb/policy.pth --evaluation-episodes 3 --evaluation-max-steps 200`. |
| Expected Metrics | Training logs separated objective terms: `loss_representation`, `loss_dynamics`, `loss_reward`, `loss_value`, `loss_policy`, `loss_critic_feedback`, `loss_imagined_rollout`, `loss_total`, `gradient_norm`, and `mean_return`. Evaluation logs `eval_mean_return`, `eval_success_rate`, `eval_threshold_pass_rate`, `eval_tuning_score`, `success_thresholds_met`, and per-scenario `threshold_met` diagnostics. A known-good Block SMB policy must pass every threshold in [block-smb-success-thresholds.md](block-smb-success-thresholds.md). |
| Artifact Locations | Policy checkpoint: `data/block_smb/policy.pth`. Sidecar: `data/block_smb/policy.json`. Run summary: `artifacts/block_smb/latest/run_summary.json`. Structured events: `artifacts/block_smb/latest/events.jsonl`. Optional TensorBoard or W&B local files: `artifacts/block_smb/tracking/` unless `--tracking-log-dir` is set. Evaluation recordings: `artifacts/block_smb/recordings/` or the explicit `--record-dir`. Known-good scripted baseline: `artifacts/block_smb/known_good_scripted_seed20260622/`. |

## Full SMB Vision

Full SMB vision is a required perception-adaptation step between Block SMB and
Full SMB policy inference/training. It bootstraps the default patch-level ViT on
synthetic scenarios composed from full-game assets so emulator observations have
a Full SMB-native semantic vocabulary before transferred policies depend on
them.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU can generate data and run small checks. Apple MPS or CUDA is recommended for the default 20 to 30 epoch training runs. |
| Runtime | The documented reference run for 30 epochs on Apple MPS is about 17 minutes for 5,000 training scenes and 1,000 validation scenes. CPU runtime is expected to be substantially longer. |
| Asset Command | `python scripts/vit/extract_sprites.py`. |
| Dataset Command | `python scripts/vit/generate_dataset.py --train 5000 --val 1000`. |
| Train Command | `python scripts/vit/train_vit.py --epochs 30 --batch 64 --dim 192 --depth 6 --device auto`. |
| Diagnostic Command | `retroagi diagnose-vision --game smb --stage full --vision-checkpoint data/vit/full_smb_vit.pth --samples 64 --rollout-steps 128 --output artifacts/full_smb/perception_diagnostic.json`. |
| Expected Metrics | Synthetic training writes `accuracy`, `foreground_accuracy`, `mean_iou`, `semantic_loss`, `support_loss`, `support_accuracy`, plus per-class `iou/<class>` and `recall/<class>` in the versioned checkpoint. Real-emulator diagnostics add unlabeled `semantic_confidence`, `class_coverage`, `covered_classes`, `temporal_stability`, `position_rmse`, `position_within_tolerance`, `bottleneck`, and `bottleneck_reasons`. The documented reference checkpoint reports about 99.94 percent overall accuracy, 99.89 percent foreground accuracy, and 99.14 percent mean IoU on 1,000 held-out synthetic scenes. |
| Artifact Locations | Sprites: `assets/spritesheets/` and `assets/sprites/`. Synthetic data: `data/vit/train.npz` and `data/vit/val.npz`. Default versioned checkpoint: `data/vit/full_smb_vit.pth`. Legacy raw weights: `data/vit/vit_smb.pth`. Preview image: `data/vit/predictions.png`. Real-emulator diagnostic summaries: `artifacts/full_smb/perception_diagnostic.json`. |

## Full SMB Content Setup

Real Full SMB emulator runs use the local content contract in
[full-smb-content.md](full-smb-content.md). The supported stable-retro game id
is `SuperMarioBros-Nes`, created through `retro.make(game="SuperMarioBros-Nes")`
after the user imports a legally obtained ROM with:

```bash
mkdir -p local/full_smb/roms local/full_smb/checksums
python -m retro.import local/full_smb/roms
shasum -a 256 local/full_smb/roms/<your-rom-file>.nes \
  > local/full_smb/checksums/SuperMarioBros-Nes.sha256
```

`local/full_smb/` is ignored by git. Preserve only metadata and checksums in
run notes or `artifacts/full_smb/<run>/content.json`; never commit or bundle
ROM bytes. Missing backend or missing imported game failures are raised through
the Full SMB content spec and include the install command, import command,
checksum path, and legal/provenance reminder.

Before headless training or evaluation, run the environment capability check:

```bash
retroagi check-env --game smb --stage full \
  --seed 0 \
  --steps 4 \
  --frame-skip 2 \
  --output artifacts/full_smb/env_check.json
```

The report verifies backend import, game registration, ROM availability,
headless reset, render reset, save/load state, action stepping, frame-skip
metadata, and deterministic seeding. Treat a nonzero exit as a setup failure,
not a training failure.

## Full SMB Task Sets

The supported headless train/eval task sets are defined in
[full-smb-tasks.md](full-smb-tasks.md) and exposed by
`retroagi.stages.full_smb.full_smb_task_catalog()`.

| Task Set | Use |
| --- | --- |
| `smoke` | Short reset/step validation after `check-env`. |
| `curriculum` | Ordered training tasks from level starts and local save-state segments. |
| `fixed_benchmark` | Repeatable evaluation tasks for tuning and regression reports. |
| `heldout_generalization` | Withheld tasks for promotion-only generalization checks. |

The catalog uses stable-retro level starts such as `Level1-1` and local
save-state artifact paths under `local/full_smb/states/`. The deterministic
recipes are defined in [full-smb-save-states.md](full-smb-save-states.md) and
exposed by `retroagi.stages.full_smb.full_smb_save_state_plan()`. Generate the
ignored local files only after `check-env` passes:

```bash
python -m retroagi.stages.full_smb.save_states plan \
  --output local/full_smb/states/save_state_plan.json

python -m retroagi.stages.full_smb.save_states create \
  --output-manifest local/full_smb/states/save_state_manifest.json \
  --overwrite
```

The generated `.state` files are local-only ROM-derived artifacts and must not
be committed.

Fixed benchmark success thresholds are documented in
[full-smb-success-thresholds.md](full-smb-success-thresholds.md) and exposed by
`retroagi.stages.full_smb.FIXED_FULL_SMB_SUCCESS_THRESHOLDS`. Full SMB
evaluation reports should preserve per-task progress, completion rate, survival
rate, mean score, mean coins, death count, mean return, episode count, step
budget, and `threshold_met` diagnostics.

## Full SMB Run Artifact Layout

Preserve comparable Full SMB work under one run directory:

`artifacts/full_smb/<run>/`

The helper `retroagi.stages.full_smb.full_smb_artifact_layout("<run>")` defines
the canonical paths and can create the directories before launching commands.
Each preserved run should write an `artifact_layout.json` manifest containing
the resolved paths.

| Path | Contents |
| --- | --- |
| `artifacts/full_smb/<run>/content.json` | Local content metadata, game id, checksum filename/hash, and provenance notes. Do not include ROM bytes. |
| `artifacts/full_smb/<run>/env_check.json` | Full SMB backend/content capability check from `retroagi check-env`. |
| `artifacts/full_smb/<run>/artifact_layout.json` | Serialized `FullSMBArtifactLayout.to_manifest()` output for the preserved run. |
| `artifacts/full_smb/<run>/summaries/throughput_benchmark.json` | Local emulator throughput benchmark with step rate, emulator-frame rate, selected device, and CPU/CUDA/MPS runtime recommendations. |
| `artifacts/full_smb/<run>/summaries/train_summary.json` | Full SMB train or resume summary from `--output-summary`. |
| `artifacts/full_smb/<run>/summaries/resume_summary.json` | Resume summary from `retroagi resume --game smb --stage full --output-summary`. |
| `artifacts/full_smb/<run>/summaries/recording_summary.json` | Record command summary from `--output-summary`. |
| `artifacts/full_smb/<run>/summaries/play_summary.json` | Play command summary, including optional inspection overlay fields. |
| `artifacts/full_smb/<run>/logs/train.jsonl` | Structured training events from `--log-path`. |
| `artifacts/full_smb/<run>/recordings/` | Compressed `.npz` rollout artifacts and recording manifests from evaluate/record/play commands. |
| `artifacts/full_smb/<run>/videos/` | Optional rendered videos such as `evaluation.mp4` and `play.mp4`. |
| `artifacts/full_smb/<run>/evaluations/evaluation.json` | Fixed-task evaluation report and threshold diagnostics. |
| `artifacts/full_smb/<run>/evaluations/fixed_task_thresholds.json` | Optional extracted fixed-task threshold report for comparison dashboards. |
| `artifacts/full_smb/<run>/comparisons/policy_suite_comparison.json` | Named policy-suite comparison report. |
| `artifacts/full_smb/<run>/comparisons/transfer_vs_scratch.json` | Legacy two-policy transfer-vs-scratch report when needed. |
| `artifacts/full_smb/<run>/tracking/` | TensorBoard, W&B offline files, or other tracker output from `--tracking-log-dir`. |
| `artifacts/full_smb/<run>/checkpoints/` | Run-local copies or symlinks for policy and transferred checkpoints. |

## Full SMB Throughput Benchmark And Device Settings

Run a local throughput benchmark after `check-env` passes and before committing
to long Full SMB training or play captures:

```bash
python -m retroagi.stages.full_smb.benchmark \
  --steps 1000 \
  --warmup-steps 100 \
  --frame-skip 2 \
  --device cpu \
  --output artifacts/full_smb/<run>/summaries/throughput_benchmark.json
```

The benchmark reports `steps_per_second`, `emulator_frames_per_second`,
`average_emulator_frames_per_step`, reset counts, terminal counts, final
signals, and `recommended_settings` for `cpu`, `cuda`, and `mps`. Add
`--encode-observations --vision-checkpoint data/vit/full_smb_vit.pth` when the
goal is to measure emulator plus Full SMB ViT preprocessing. Add `--render`
only for play-latency checks; keep training and evaluation benchmarks headless.

Recommended Full SMB device settings:

| Device | Training | Play |
| --- | --- | --- |
| CPU | Use `--device cpu` for CI, `check-env`, smoke training, and machines without an accelerator. Keep `--render-mode none`, freeze perception for policy smoke runs, and start with small `--rollout-steps` such as 32 to 64 until the benchmark report shows enough emulator headroom. | Use CPU for manual debugging and deterministic playback. Use `--render-mode human --fps 30` for interactive play and `--render-mode none` for evaluation, recording, or latency measurement. |
| CUDA | Use `--device cuda` on Linux/NVIDIA systems when policy, world-model, critic, or Full SMB ViT compute dominates the benchmark. The `stable-retro` emulator still steps on CPU, so increase rollout length or policy size only after measuring local `emulator_frames_per_second`. | Use CUDA for heavier policy/ViT inference during rendered playback. Disable recording or video export when measuring interaction latency. |
| MPS | Use `--device mps` on Apple Silicon after comparing the same benchmark with CPU. MPS is most useful for ViT and larger policy compute; tiny one-env rollouts may be limited by device-transfer overhead. | Use MPS for macOS rendered policy playback when model inference is heavier than the baseline smoke configuration; benchmark CPU and MPS with the same `frame_skip` and `--steps`. |

Treat the benchmark as local evidence, not a portable performance guarantee:
ROM setup, stable-retro version, render mode, frame skip, recording, ViT
checkpoint size, and tracker backends all change throughput.

## Full SMB Command Workflow

Use this command order when preserving a trainable and playable Full SMB run.
Create the run layout first so every later command writes to stable paths:

The checked-in documented benchmark contract is
`artifacts/full_smb/documented_benchmark_seed0/benchmark_manifest.json`, with
the matching runbook at `artifacts/full_smb/documented_benchmark_seed0/RUN.md`.
It intentionally commits only commands, paths, and qualification gates because
Full SMB ROM bytes and emulator-trained checkpoints are local-only artifacts.

```bash
python - <<'PY'
import json
from retroagi.stages.full_smb import full_smb_artifact_layout

layout = full_smb_artifact_layout("<run>")
layout.ensure_directories()
layout.files()["layout_manifest"].write_text(
    json.dumps(layout.to_manifest(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
```

Transfer the Block SMB policy into the Full SMB policy contract after the Full
SMB ViT has passed the asset-mock perception gate:

```bash
retroagi transfer --game smb --stage full \
  --block-policy-checkpoint data/block_smb/policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint artifacts/full_smb/<run>/checkpoints/transferred_policy.pth
```

Fine-tune the transferred policy headlessly and preserve the resolved config,
logs, checkpoint, recording manifest, and deterministic evaluation cadence:

```bash
retroagi train --game smb --stage full \
  --mode fine-tune \
  --init-checkpoint artifacts/full_smb/<run>/checkpoints/transferred_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze \
  --task-set curriculum \
  --epochs 1 \
  --updates-per-epoch 1 \
  --rollout-steps 64 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 64 \
  --evaluation-interval-epochs 1 \
  --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --recording-dir artifacts/full_smb/<run>/recordings \
  --recording-path artifacts/full_smb/<run>/recordings/recording_manifest.npz \
  --log-path artifacts/full_smb/<run>/logs/train.jsonl \
  --tracking-log-dir artifacts/full_smb/<run>/tracking \
  --output-summary artifacts/full_smb/<run>/summaries/train_summary.json
```

Use `--mode scratch` and omit `--init-checkpoint` to train a Full SMB policy
from a fresh architecture instance. Use `--perception-mode fine_tune` only when
the run is intentionally updating the Full SMB ViT; otherwise keep
`--perception-mode freeze` so policy changes are easier to compare.

Resume with the same task schedule and tracking destination. Preserve a
separate resumed checkpoint when the interrupted checkpoint should remain
available for audit:

```bash
retroagi resume --game smb --stage full \
  --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --save-checkpoint artifacts/full_smb/<run>/checkpoints/resumed_policy.pth \
  --task-set curriculum \
  --epochs 2 \
  --updates-per-epoch 1 \
  --rollout-steps 64 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 64 \
  --evaluation-interval-epochs 1 \
  --tracking-backend none \
  --tracking-log-dir artifacts/full_smb/<run>/tracking \
  --output-summary artifacts/full_smb/<run>/summaries/resume_summary.json
```

Evaluate the saved policy on the fixed benchmark set and keep the threshold
diagnostics separate from training summaries:

```bash
retroagi evaluate --game smb --stage full \
  --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --evaluation-episodes 3 \
  --evaluation-max-steps 2400 \
  --output-summary artifacts/full_smb/<run>/evaluations/evaluation.json
```

Record deterministic evaluation rollouts for replay and regression debugging:

```bash
retroagi record --game smb --stage full \
  --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --evaluation-episodes 3 \
  --evaluation-max-steps 2400 \
  --record-dir artifacts/full_smb/<run>/recordings \
  --recording-path artifacts/full_smb/<run>/recordings/recording_manifest.npz \
  --output-summary artifacts/full_smb/<run>/summaries/recording_summary.json
```

Play the saved policy locally with rendering, action-repeat control, and the
policy-inspection overlay:

```bash
retroagi play --game smb --stage full \
  --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --level 1-1 \
  --steps 1000 \
  --frame-skip 4 \
  --action-repeat 2 \
  --render-mode human \
  --deterministic-policy \
  --inspection-overlay \
  --fps 30 \
  --record \
  --record-dir artifacts/full_smb/<run>/recordings \
  --record-output artifacts/full_smb/<run>/recordings/play_manifest.npz \
  --output-summary artifacts/full_smb/<run>/summaries/play_summary.json
```

Use `--render-mode none` or `--no-render` for headless playback, and use
`--sampling-policy --temperature <value>` when policy sampling is part of the
experiment. Human-control debugging does not require a checkpoint:

```bash
retroagi play --game smb --stage full \
  --human \
  --task-set smoke \
  --level 1-1 \
  --scenario debug \
  --render-mode human \
  --fps 30
```

Compare transferred, scratch-trained, fine-tuned, and known-good policies on
identical task and seed streams:

```bash
retroagi compare --game smb --stage full \
  --transfer-checkpoint artifacts/full_smb/<run>/checkpoints/transferred_policy.pth \
  --scratch-trained-checkpoint artifacts/full_smb/<run>/checkpoints/scratch_policy.pth \
  --fine-tuned-checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth \
  --known-good-checkpoint artifacts/full_smb/<run>/checkpoints/known_good_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --task-set fixed_benchmark \
  --seed 0 \
  --seed 1 \
  --output artifacts/full_smb/<run>/comparisons/policy_suite_comparison.json
```

The minimum preserved evidence for this workflow is the transfer checkpoint and
sidecar, policy checkpoint and sidecar, train/resume summaries, structured log,
evaluation report, recording summary and manifests, play summary, comparison
report, throughput benchmark, and content metadata. The documented benchmark
qualifies only after the local `evaluation.json` has `success_thresholds_met`
set to true, every fixed task has `threshold_met` set to true, the recording
and play manifests exist, and the comparison report includes `action_agreement`.

## Full SMB Adapter And Transfer

Full SMB connects the shared stage contract to the stable-retro emulator. Its
first responsibility is to verify and validate transferred-model inference
against full emulator observations and actions; after that, training continues
from the transferred model checkpoints at full fidelity.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU is required for headless smoke tests and transfer checks. GPU acceleration is not required for the adapter; CUDA or MPS may be used for policy, world-model, critic, and ViT compute after local throughput benchmarking. |
| Runtime | Headless smoke checks should stay short at the default 200 steps. Comparison runtime scales linearly with `--steps * task_count * seed_count * policy_count` because named policies are evaluated on identical seeded task streams. Full SMB policy training scales with `--epochs`, `--updates-per-epoch`, and `--rollout-steps`; `--vector-env-count` is captured in config/checkpoints but active training remains single-env until vector rollout storage lands. Use `python -m retroagi.stages.full_smb.benchmark` to record local `emulator_frames_per_second` before long runs. |
| Smoke Command | `retroagi evaluate --game smb --stage full --steps 500 --seed 0 --encode-observations`. |
| Transfer Command | `retroagi transfer --game smb --stage full --block-policy-checkpoint data/block_smb/policy.pth --full-smb-vision-checkpoint data/vit/full_smb_vit.pth --output-checkpoint data/full_smb/transferred_policy.pth`. |
| Training Command | `retroagi train --game smb --stage full --mode fine-tune --init-checkpoint data/full_smb/transferred_policy.pth --full-smb-vision-checkpoint data/vit/full_smb_vit.pth --perception-mode freeze --updates-per-epoch 1 --rollout-steps 64 --evaluation-episodes 1 --evaluation-max-steps 64 --evaluation-interval-epochs 1 --recording-dir artifacts/full_smb/<run>/recordings --recording-path artifacts/full_smb/<run>/recordings/recording_manifest.npz --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth --log-path artifacts/full_smb/<run>/logs/train.jsonl --output-summary artifacts/full_smb/<run>/summaries/train_summary.json --tracking-log-dir artifacts/full_smb/<run>/tracking`. Use `--mode scratch` and omit `--init-checkpoint` to start a new Full SMB policy. |
| Resume Command | `retroagi resume --game smb --stage full --checkpoint data/full_smb/policy.pth --save-checkpoint data/full_smb/resumed_policy.pth --epochs 2`; omit `--save-checkpoint` to resume in place. Resume restores saved RNG streams and rejects changes to the saved task schedule, recurrent-state contract, or tracking destination. |
| Evaluation Command | `retroagi evaluate --game smb --stage full --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth --evaluation-episodes 3 --evaluation-max-steps 2400 --output-summary artifacts/full_smb/<run>/evaluations/evaluation.json`. |
| Record Command | `retroagi record --game smb --stage full --checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth --evaluation-episodes 3 --evaluation-max-steps 2400 --record-dir artifacts/full_smb/<run>/recordings --recording-path artifacts/full_smb/<run>/recordings/recording_manifest.npz --output-summary artifacts/full_smb/<run>/summaries/recording_summary.json`. If no recording destination is supplied, the command writes compressed episode artifacts under `artifacts/full_smb/recordings/` and a manifest at `artifacts/full_smb/recording_manifest.npz`. |
| Play Command | `retroagi play --game smb --stage full --checkpoint data/full_smb/policy.pth --task-set fixed_benchmark --level 1-1 --steps 1000 --render-mode human --inspection-overlay --fps 30`. Use `--sampling-policy --temperature 0.75` for stochastic action sampling, `--render-mode none` or `--no-render` for headless playback, `--pause-at-start` for terminal-controlled stepping, `--no-reset-on-done` to stop after the first terminal episode, and `--record --record-dir artifacts/full_smb/recordings --record-output artifacts/full_smb/play_manifest.npz` to preserve playback artifacts. Human debugging uses `retroagi play --game smb --stage full --human --task-set smoke --level 1-1 --scenario debug --render-mode human --fps 30`; line controls are `d/right`, `d+/right_jump`, `a/left`, `a+/left_jump`, `w/space/jump`, and empty input for noop. Terminal controls are `p` for pause/resume, `r` for reset, and `q` for quit when stdin is interactive. |
| Compare Command | `retroagi compare --game smb --stage full --transfer-checkpoint artifacts/full_smb/<run>/checkpoints/transferred_policy.pth --scratch-trained-checkpoint artifacts/full_smb/<run>/checkpoints/scratch_policy.pth --fine-tuned-checkpoint artifacts/full_smb/<run>/checkpoints/policy.pth --known-good-checkpoint artifacts/full_smb/<run>/checkpoints/known_good_policy.pth --task-set fixed_benchmark --seed 0 --seed 1 --output artifacts/full_smb/<run>/comparisons/policy_suite_comparison.json`. |
| Expected Metrics | Smoke output reports `steps`, `resets`, `episodes`, total `reward`, adapter-owned `FullSMBRewardConfig` plus `reward_terms`, the `FullSMBObservationConfig` preprocessing manifest, and `camera_vec` when stage info is retained. Transfer checkpoints preserve source policy metrics and transfer provenance. Trainer checkpoints preserve resolved rollout/update settings, loss weights, reward config, perception mode, deterministic mode, RNG state keys, task/curriculum state, backend/content metadata, source-checkpoint provenance, recording paths, tracking config, periodic deterministic evaluation cadence/results, evaluation recording manifests, and train/evaluation returns. Policy evaluation and recording report `fixed_task_results`, per-task `threshold_met` and `threshold_diagnostics`, `tuning_metrics.threshold_pass_rate`, `tuning_metrics.score`, top-level `success_thresholds_met`, and a `recording` manifest when artifacts are enabled. Play reports steps, resets, completed episodes, total/mean return, selected action IDs/names, control mode, deterministic-vs-sampling mode, render/fps settings, quit status, final signals, last reward terms, last inspection overlay, bounded overlay history, human action bindings when active, and optional recording manifest. Comparisons report one stream per task/seed pair, per-policy action histograms, mean entropies, mean margins, collection reward, reset/termination counts, and aggregate pairwise `action_agreement` across transferred, scratch, fine-tuned, known-good, and additional named policies. |
| Artifact Locations | Use `artifacts/full_smb/<run>/` for preserved Full SMB runs. Transfer checkpoint: `data/full_smb/transferred_policy.pth` and sidecar `data/full_smb/transferred_policy.json`, copied or symlinked to `artifacts/full_smb/<run>/checkpoints/transferred_policy.pth` when preserving a run. Continued policy checkpoint: `artifacts/full_smb/<run>/checkpoints/policy.pth`. Required Full SMB vision checkpoint: `data/vit/full_smb_vit.pth`. Throughput report: `artifacts/full_smb/<run>/summaries/throughput_benchmark.json`. Training log: `artifacts/full_smb/<run>/logs/train.jsonl`. Policy evaluation summary: `artifacts/full_smb/<run>/evaluations/evaluation.json`. Recording summary: `artifacts/full_smb/<run>/summaries/recording_summary.json`. Evaluation/play recordings: compressed per-episode `.npz` files under `artifacts/full_smb/<run>/recordings/<evaluation-prefix>/` plus `artifacts/full_smb/<run>/recordings/recording_manifest.npz`, `artifacts/full_smb/<run>/recordings/recording_manifest_<evaluation-prefix>.npz`, or an explicit play manifest such as `artifacts/full_smb/<run>/recordings/play_manifest.npz`; each episode file stores frames, actions, rewards, signals, task/scenario/state IDs, and termination flags. Optional video export is attempted when `--recording-path` has a video suffix and OpenCV is installed; store videos under `artifacts/full_smb/<run>/videos/`. Optional tracker output: `artifacts/full_smb/<run>/tracking/`. Comparison summaries: `artifacts/full_smb/<run>/comparisons/transfer_vs_scratch.json` for the legacy two-policy report or `artifacts/full_smb/<run>/comparisons/policy_suite_comparison.json` for named policy suites. Legacy flat paths such as `artifacts/full_smb/train.jsonl`, `artifacts/full_smb/evaluation.json`, and `artifacts/full_smb/transfer_vs_scratch.json` remain recognizable but should be migrated into run directories for preserved experiments. |

## Preservation Checklist

Before treating a run as comparable or known-good, preserve:

1. The command line and resolved config.
2. The seed and selected device.
3. The versioned checkpoint plus sidecar JSON.
4. The structured event log or tracking run when available.
5. Evaluation summaries and recordings for policy runs.
6. The exact artifact paths used by downstream transfer or comparison steps.
