# RetroAGI
General purpose machine learning agent for retro games  

RetroAGI is organized as a three-stage curriculum for training an architecture
to play Super Mario Bros:

1. **Synthetic 1D** validates the hierarchy on procedural sequence data.
2. **Block SMB** uses a scriptable pygame-ce-backed platformer with
   low-resolution, scenario-driven tasks.
3. **Full SMB** connects the same architecture to the full emulator.

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

The [tensor contracts](docs/tensor-contracts.md) define hierarchy and vision
shapes, dtypes, normalization ranges, and timescales.

The [AI teaching curriculum](docs/ai-teaching-curriculum.md) provides a
10-week path for using the project to teach practical AI engineering.


## Usage
1. Create and activate a supported Python environment using the commands in
   [docs/compatibility.md](docs/compatibility.md).
2. Run a curriculum stage:
   ```bash
   python -m retroagi.stages.synthetic_1d.train
   retroagi train --stage block-smb --epochs 5 \
     --vision-checkpoint data/block_vit/block_vit.pth \
     --checkpoint data/block_smb/policy.pth \
     --output artifacts/block_smb/latest/run_summary.json
   retroagi resume --stage block-smb --checkpoint data/block_smb/policy.pth --epochs 10
   retroagi evaluate --stage block-smb --checkpoint data/block_smb/policy.pth
   retroagi record --stage block-smb --checkpoint data/block_smb/policy.pth --record-dir artifacts/block_smb/recordings
   retroagi transfer --stage full-smb \
     --block-policy-checkpoint data/block_smb/policy.pth \
     --output-checkpoint data/full_smb/transferred_policy.pth
   retroagi compare --stage full-smb \
     --transfer-checkpoint data/full_smb/transferred_policy.pth \
     --output artifacts/full_smb/transfer_vs_scratch.json
   retroagi evaluate --stage full-smb --steps 500 --seed 0
   ```
   The `retroagi` command is the preferred entry point for stage selection.
   Stage-specific options are forwarded to the selected implementation; the
   legacy `retroagi-block-smb` command remains available for Block SMB-only
   workflows.
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
   SMB policy transfer reuses Block SMB actor/world-model/critic weights and
   pairs them with the versioned Full SMB ViT checkpoint. Transfer comparisons
   evaluate the transferred policy and a scratch Full SMB baseline on identical
   seeded observation batches.
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

Stage 2 has the scriptable environment and adapter in place; training loops can
now reuse `retroagi.core.models.AgentWorldModelCritic` and consume
`BlockSMBStage.encode_observation(...)`. Stage 3 keeps the full emulator runner
isolated behind `retroagi/stages/full_smb`.
