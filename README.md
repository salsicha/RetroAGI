# RetroAGI
General purpose machine learning agent for retro games.

RetroAGI is organized as a progressive-resolution curriculum for training and
promoting an architecture toward full Super Mario Bros:

1. **Synthetic 1D** fully validates the architecture on procedural sequence
   data: tensor contracts, hierarchy behavior, losses, gradients, baselines,
   checkpointing, and deterministic metrics.
2. **Block SMB** trains all trainable game-facing models on a simplified
   synthetic version of SMB: Block ViT perception plus the hierarchical
   actor/world-model/critic policy in fast scenario-driven tasks.
3. **Full SMB asset-mock perception** bootstraps the Full SMB ViT with full game
   assets arranged into synthetic scenarios before any policy relies on full
   emulator observations.
4. **Full SMB** verifies and validates inference in the emulator, then
   continues training the transferred models at full fidelity.

The stage code is separated, but all stages share the same core contract:

```text
observation -> hierarchical actor -> action
observation + action -> world model prediction
prediction -> critic -> actor refinement
```

Shared actor/world-model/critic components live in `retroagi/core`. Stage
adapters live in `retroagi/stages/*` and convert stage-native observations into
the common A/B/C timescale tensors.

## Project Layout

```text
retroagi/
  core/
    interfaces.py      # StageSpec, StageBatch, shared adapter protocol
    models.py          # reusable actor, world model, critic, controller
  stages/
    synthetic_1d/      # procedural one-dimensional validation
    block_smb/         # pygame-ce SMB-like scenarios and adapter
    full_smb/          # stable-retro adapter and emulator runner
scripts/               # compatibility wrappers and older experiments
```

## Architecture

See the [architecture diagram](docs/architecture.html) for the hierarchical
actor/world-model/critic flow.

## Current Status

- **Synthetic 1D** is the architecture validation stage and is reproducible from
  a clean checkout with deterministic datasets, baselines, checkpoint
  save/restore, and held-out metrics.
- **Block SMB** is the simplified synthetic game-training stage. It has a
  pygame-ce environment, fixed success thresholds, Block ViT perception, policy
  training, evaluation, resume, recording, ablations, structured logs, and
  optional TensorBoard or W&B tracking.
- **Full SMB** has the asset-mock ViT bootstrapping rung, a stable-retro stage
  adapter, headless smoke evaluation, Block SMB policy transfer, continued Full
  SMB training, and transfer-vs-scratch comparison tooling.
- **Operations** are covered by CI, native install instructions, stage
  operations guidance, and a clean-checkout reproducibility procedure.

## Supported Platforms

RetroAGI supports Linux x86-64 and macOS Apple Silicon with Python 3.12
through 3.14. It pins PyTorch 2.9.1 with torchvision 0.24.1. CPU-only
execution is the baseline; CUDA and Apple Metal/MPS acceleration are selected
automatically when available. CUDA 12.8 is the primary Linux GPU target, CUDA
13.0 is the secondary Linux GPU target, and MPS is the native macOS GPU target.

See the [compatibility matrix and installation commands](docs/compatibility.md)
before creating an environment.

The [stage semantics](docs/stage-semantics.md) define observations, actions,
rewards, episode endings, and resets across the curriculum.

The [Block SMB success thresholds](docs/block-smb-success-thresholds.md) define
the deterministic fixed-scenario bar for considering a policy known-good.
The current known-good baseline is a scripted policy artifact at
`artifacts/block_smb/known_good_scripted_seed20260622/`.

The [operations reference](docs/operations.md) lists hardware expectations,
runtime targets, metrics, and artifact locations for each stage.

The [reproducibility procedure](docs/reproducibility.md) starts from a clean
checkout and walks through installation, tests, smoke training, and artifact
verification.

The [tensor contracts](docs/tensor-contracts.md) define hierarchy and vision
shapes, dtypes, normalization ranges, and timescales.

The [AI teaching curriculum](docs/ai-teaching-curriculum.md) provides a
10-week path for using the project to teach practical AI engineering.


## Usage
1. Create and activate a supported Python environment using the commands in
   [docs/compatibility.md](docs/compatibility.md). For a fully traceable run
   from a fresh clone, follow
   [docs/reproducibility.md](docs/reproducibility.md).
2. Run a curriculum stage:
   ```bash
   python -m retroagi.stages.synthetic_1d.train
   retroagi train --game smb --stage block --epochs 5 \
     --vision-checkpoint data/block_vit/block_vit.pth \
     --checkpoint data/block_smb/policy.pth \
     --output artifacts/block_smb/latest/run_summary.json
   retroagi resume --game smb --stage block \
     --checkpoint data/block_smb/policy.pth --epochs 10
   retroagi evaluate --game smb --stage block \
     --checkpoint data/block_smb/policy.pth
   retroagi record --game smb --stage block \
     --checkpoint data/block_smb/policy.pth \
     --record-dir artifacts/block_smb/recordings
   retroagi transfer --game smb --stage full \
     --block-policy-checkpoint data/block_smb/policy.pth \
     --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
     --output-checkpoint data/full_smb/transferred_policy.pth
   retroagi train --game smb --stage full \
     --init-checkpoint data/full_smb/transferred_policy.pth \
     --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
     --perception-mode freeze \
     --updates-per-epoch 1 \
     --rollout-steps 64 \
     --checkpoint data/full_smb/policy.pth
   retroagi evaluate --game smb --stage full \
     --checkpoint data/full_smb/policy.pth \
     --evaluation-episodes 3
   retroagi record --game smb --stage full \
     --checkpoint data/full_smb/policy.pth \
     --record-dir artifacts/full_smb/recordings \
     --recording-path artifacts/full_smb/recording_manifest.npz
   retroagi play --game smb --stage full \
     --checkpoint data/full_smb/policy.pth \
     --render \
     --fps 30
   retroagi play --game smb --stage full \
     --human \
     --state Level1-1 \
     --render \
     --fps 30
   retroagi compare --game smb --stage full \
     --transfer-checkpoint data/full_smb/transferred_policy.pth \
     --output artifacts/full_smb/transfer_vs_scratch.json
   retroagi evaluate --game smb --stage full --steps 500 --seed 0
   ```
   The `retroagi` command is the preferred entry point for selecting a game and
   fidelity rung. Use `--game smb --stage synthetic|block|full`; legacy stage
   aliases such as `block-smb` and `full-smb` remain accepted. Stage-specific
   options are forwarded to the selected implementation; the legacy
   `retroagi-block-smb` command remains available for Block SMB-only workflows.
   Full SMB policy training records its perception choice in every checkpoint:
   `--perception-mode freeze` reuses a frozen Full SMB ViT checkpoint,
   `fine_tune` includes trainable ViT parameters in the optimizer, and `replace`
   starts from a fresh trainable Full SMB ViT.
   Full SMB trainer checkpoints also capture the resolved rollout/update shape,
   vector environment request, reward config, loss weights, recording paths,
   tracking config, deterministic mode, training source provenance, and the
   validated Full SMB A/B/C stage-batch contract. Scratch training is selected
   by omitting `--resume` and `--init-checkpoint`, which builds the policy via
   the shared architecture factory before consuming Full SMB stage batches.
   `--init-checkpoint` starts transferred-policy fine-tuning from either a raw
   Block SMB policy checkpoint or an existing Full SMB transfer checkpoint.
   Rollouts carry recurrent world-model state only while the episode continues
   and drop it on manual reset, death, timeout, completion, game over,
   termination, or truncation boundaries.
   Training summaries and checkpoints now include compact rollout replay
   records with actions, rewards, done/truncated flags, episode masks,
   boundary reasons, scenario/task/emulator-state IDs when available, selected
   Full SMB signals, and reward terms.
   Full SMB training also records numerical safety settings and metrics:
   finite loss/logit/gradient checks, gradient clipping, reward-scale bounds,
   reward/value prediction bounds, action entropy, and early-stop reasons for
   NaN or exploding values.
   Real Full SMB emulator runs require the optional stable-retro backend:
   `pip install ".[full-smb]"`. Unit tests, Block SMB training, and CI smoke
   training do not install or build that native emulator dependency.
   Each versioned checkpoint also writes a JSON sidecar beside the `.pth` file
   containing the resolved config, metrics, code revision, runtime environment,
   specs, metadata, and state keys.
   Block SMB ablations can be run with paired switches such as
   `--disable-vision`, `--disable-world-model`, `--disable-critic-feedback`,
   `--disable-hierarchy`, `--disable-recurrent-state`, and
   `--disable-checkpoint-transfer`. The
   resolved ablation config is written beside each run summary and checkpoint.
   Training can write structured JSONL events with `--log-path`; deterministic
   evaluation cadence is controlled with `--evaluation-interval-epochs`.
   Optional TensorBoard and Weights & Biases tracking can be installed with
   `pip install ".[tracking]"` and enabled with
   `--tracking-backend tensorboard|wandb`.
   Low-level controller gain schedules are selectable with
   `--controller-schedule constant|linear`. The Full SMB random-agent runner is
   headless by default; pass `--render` only for local visual inspection. Full
   SMB policy transfer reuses Block SMB actor/world-model/critic weights.
   Before that transferred policy is used for Full SMB inference or continued
   training, the Full SMB ViT must be bootstrapped on synthetic scenarios made
   from full-game assets and loaded as the versioned Full SMB ViT checkpoint.
   Transfer comparisons evaluate the transferred policy and a scratch Full SMB
   baseline on identical seeded observation batches.
   Learned-dynamics imagination is selectable with
   `--imagined-rollout-horizon` and `--imagined-rollout-weight`.
   Target-network stabilization is selectable with `--target-network-mode`,
   `--target-network-tau`, and `--target-network-instability-threshold`.
3. Run the test suite:
   ```bash
   python -m unittest discover -s scripts/tests -v
   ```

Legacy wrappers still work:
   ```bash
   python scripts/simple_transformer.py
   python scripts/run.py
   ```

## Training

Train the Block SMB vision transformer directly from procedural pygame-ce
rollouts. Semantic masks and Mario positions are generated exactly from the
renderer palette:

```bash
python scripts/vit/train_block_vit.py \
  --epochs 20 \
  --samples-per-epoch 2048 \
  --val-samples 512
```

The best checkpoint and its JSON metrics are written to `data/block_vit/`.
Training can be continued with:

```bash
python scripts/vit/train_block_vit.py --epochs 40 --resume data/block_vit/block_vit.pth
```

### Synthetic 1D validation

Stage 1 currently trains the shared hierarchical actor/world-model/critic stack
on synthetic data. The deterministic validation path is covered by:

```bash
timeout 120s python3 -m unittest -v scripts.tests.test_synthetic_1d
```

Expected CPU runtime for this Stage 1 suite is under 20 seconds on the current
development machine; the held-out baseline demonstration itself runs in about
7 seconds. It trains a small `AgentWorldModelCritic` on deterministic Synthetic
1D train data and evaluates on the fixed test split. The expected result is that
the trained policy's held-out `controller_mse` is below both declared baselines:
seeded `random` and train-marginal `simple`, with an asserted margin of at least
25% below the simple baseline. The same suite also verifies finite CPU gradients,
decreasing total loss, deterministic data/permutation seeding, shared-schema
checkpoint save/restore, and these evaluation metrics: `controller_mse`,
`controller_mae`, `controller_rmse`, `error_B`, and `accuracy_A`.

Block SMB policy training uses `AgentWorldModelCritic` through the shared stage
contract and can be run, resumed, evaluated, and recorded from the CLI. Full SMB
keeps emulator interaction isolated behind `retroagi/stages/full_smb` and
supports deterministic smoke evaluation plus transferred-policy comparison.
