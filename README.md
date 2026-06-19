
<a href="">
  <img src="https://media.githubusercontent.com/media/salsicha/RetroAGI/main/mario.gif"
    height="80" align="right" alt="" />
</a><br>


# RetroAGI
General purpose machine learning agent for retro games  

RetroAGI is organized as a three-stage curriculum for training an architecture
to play Super Mario Bros:

1. **Synthetic 1D** validates the hierarchy on procedural sequence data.
2. **Block SMB** uses a scriptable pygame platformer with low-resolution,
   scenario-driven tasks.
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
    block_smb/         # pygame SMB-like scenarios and adapter
    full_smb/          # stable-retro / emulator runner
scripts/               # compatibility wrappers and older experiments
```

## Diagram
![The Brain](https://github.com/salsicha/RetroAGI/blob/main/docs/architecture.html)


## Build
```bash
./build.sh
```

## Supported Platforms

RetroAGI supports Linux x86-64 with Python 3.12 and pins PyTorch 2.9.1 with
torchvision 0.24.1. CPU-only execution is the baseline. CUDA 12.8 is the
primary GPU target, and CUDA 13.0 is the supported container target. GPU and
container verification remain separate test activities.

See the [compatibility matrix and installation commands](docs/compatibility.md)
before creating an environment.

The [stage semantics](docs/stage-semantics.md) define observations, actions,
rewards, episode endings, and resets across the curriculum.

The [tensor contracts](docs/tensor-contracts.md) define hierarchy and vision
shapes, dtypes, normalization ranges, and timescales.


## Usage
1. Start the container environment:
   ```bash
   ./run.sh
   ```
2. Run a curriculum stage:
   ```bash
   python -m retroagi.stages.synthetic_1d.train
   python -m retroagi.stages.full_smb.run --steps 500
   ```

Legacy wrappers still work:
   ```bash
   python scripts/simple_transformer.py
   python scripts/run.py
   ```

## Training

Train the Block SMB vision transformer directly from procedural pygame
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
