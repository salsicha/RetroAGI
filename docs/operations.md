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
failures before a concept reaches any game-like environment.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU is sufficient and required. CUDA or MPS may be used through `SyntheticTrainingConfig.device`, but GPU is not required for acceptance. |
| Runtime | The focused Synthetic 1D test suite is expected to finish in under 20 seconds on the current development machine. The baseline held-out training demonstration is expected to run in about 7 seconds. |
| Command | `python -m retroagi.stages.synthetic_1d.train` or `retroagi train --game smb --stage synthetic`. |
| Expected Metrics | `controller_mse`, `controller_mae`, `controller_rmse`, `error_B`, and `accuracy_A`. A passing learned policy beats both the seeded random and train-marginal simple baselines, with the tested margin at least 25 percent below the simple baseline on `controller_mse`. |
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
| Expected Metrics | Training writes `loss`, `semantic_loss`, `position_loss`, `accuracy`, `foreground_accuracy`, and `mean_iou`. Diagnostics add `position_rmse`, `position_within_tolerance`, `thresholds`, `bottleneck`, and `bottleneck_reasons`. The default diagnostic thresholds are at least 0.95 accuracy, 0.90 foreground accuracy, 0.70 mean IoU, at most 0.06 position RMSE, and at least 0.90 position within tolerance. |
| Artifact Locations | Default checkpoint: `data/block_vit/block_vit.pth`. Its sidecar is `data/block_vit/block_vit.json`. Diagnostic summaries should be written with `--output artifacts/block_smb/vision_diagnostic.json` when preserving a run. |

## Block SMB Policy

Block SMB policy training is the main trainable simplified-game stage. It uses
the four fixed scenarios plus optional generated scenarios and reports
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
| Expected Metrics | `accuracy`, `foreground_accuracy`, `mean_iou`, plus per-class `iou/<class>` and `recall/<class>` in the versioned checkpoint. The documented reference checkpoint reports about 99.94 percent overall accuracy, 99.89 percent foreground accuracy, and 99.14 percent mean IoU on 1,000 held-out synthetic scenes. |
| Artifact Locations | Sprites: `assets/spritesheets/` and `assets/sprites/`. Synthetic data: `data/vit/train.npz` and `data/vit/val.npz`. Default versioned checkpoint: `data/vit/full_smb_vit.pth`. Legacy raw weights: `data/vit/vit_smb.pth`. Preview image: `data/vit/predictions.png`. |

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

## Full SMB Adapter And Transfer

Full SMB connects the shared stage contract to the stable-retro emulator. Its
first responsibility is to verify and validate transferred-model inference
against full emulator observations and actions; after that, training continues
from the transferred model checkpoints at full fidelity.

| Area | Operational Target |
| --- | --- |
| Hardware | CPU is required for headless smoke tests and transfer checks. GPU acceleration is not required for the adapter; CUDA or MPS may be used for model inference comparisons. |
| Runtime | Headless smoke checks should stay short at the default 200 steps. Comparison runtime scales linearly with `--steps` because transferred and scratch policies are evaluated on the same observation stream. |
| Smoke Command | `retroagi evaluate --game smb --stage full --steps 500 --seed 0 --encode-observations`. |
| Transfer Command | `retroagi transfer --game smb --stage full --block-policy-checkpoint data/block_smb/policy.pth --full-smb-vision-checkpoint data/vit/full_smb_vit.pth --output-checkpoint data/full_smb/transferred_policy.pth`. |
| Compare Command | `retroagi compare --game smb --stage full --transfer-checkpoint data/full_smb/transferred_policy.pth --output artifacts/full_smb/transfer_vs_scratch.json`. |
| Expected Metrics | Smoke output reports `steps`, `resets`, `episodes`, total `reward`, adapter-owned `FullSMBRewardConfig` plus `reward_terms`, the `FullSMBObservationConfig` preprocessing manifest, and `camera_vec` when stage info is retained. Transfer checkpoints preserve source policy metrics and transfer provenance. Comparisons report `action_agreement`, transfer and scratch action histograms, mean entropies, mean margins, `collection_reward`, resets, completed episodes, terminated count, and truncated count. |
| Artifact Locations | Transfer checkpoint: `data/full_smb/transferred_policy.pth` and sidecar `data/full_smb/transferred_policy.json`. Required Full SMB vision checkpoint: `data/vit/full_smb_vit.pth`. Comparison summary: `artifacts/full_smb/transfer_vs_scratch.json`. Smoke runs do not write artifacts unless wrapped by a caller; preserve terminal output or add an explicit comparison output for retained experiments. |

## Preservation Checklist

Before treating a run as comparable or known-good, preserve:

1. The command line and resolved config.
2. The seed and selected device.
3. The versioned checkpoint plus sidecar JSON.
4. The structured event log or tracking run when available.
5. Evaluation summaries and recordings for policy runs.
6. The exact artifact paths used by downstream transfer or comparison steps.
