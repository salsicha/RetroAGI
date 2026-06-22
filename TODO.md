# RetroAGI TODO

This roadmap is ordered by dependency and execution priority. Finish each
milestone's exit criteria before expanding the next one.

## Completed Work

**Architecture and stages**

- [x] Separate shared actor/world-model/critic code from stage adapters.
- [x] Define `StageSpec`, `StageBatch`, and the common `StageAdapter` protocol.
- [x] Define shared `VisionSpec`, `VisionOutput`, and `VisionEncoder` contracts.
- [x] Implement synthetic 1D, Block SMB, and Full SMB vision implementations.
- [x] Implement the scriptable Block SMB environment and four fixed scenarios.
- [x] Add deterministic Block SMB scenario generation and environment seeding.
- [x] Implement baseline Block SMB rewards for progress, coins, enemies, death,
      completion, and elapsed time.
- [x] Train Block SMB perception from exact procedural semantic and position
      labels.
- [x] Add Block ViT metrics, checkpoint metadata, and resume support.
- [x] Wrap the existing Full SMB DeepLab checkpoint in the shared vision API.
- [x] Add scenario, vision-interface, checkpoint-load, and trainer smoke tests.
- [x] Specify and test how critic feedback changes the actor's second pass.

**Repository baseline**

- [x] Remove the 550 tracked `.venv/` files; retain `.venv/` in `.gitignore`.
- [x] Move trained models, generated datasets, and other large artifacts to Git
      LFS or external artifact storage.
- [x] Add `pyproject.toml` with pinned runtime, test, formatting, and linting
      dependencies.
- [x] Declare supported Python, PyTorch, CUDA, and CPU-only configurations.
- [x] Remove obsolete Docker setup references and document the supported native
      installation flow.


## P0: Repository Baseline (Invalid)

This milestone is invalid: a CPU-only version is not a supported target for this
project.

- [x] Do not require a clean CPU-only installation to pass the test suite.

**Exit criteria:** none; keep platform support scoped to the declared accelerated
runtime targets.


## P1: Freeze Cross-Stage Contracts

Complete the contracts before building the Block SMB agent loop or Full SMB
adapter against unstable assumptions.

- [x] Document observation, action, reward, termination, truncation, and reset
      semantics for every stage.
- [x] Define one named action vocabulary shared by Block SMB and Full SMB.
- [x] Document all `StageSpec`, `StageBatch`, and `VisionOutput` tensor shapes,
      dtypes, normalization ranges, and timescales.
- [x] Define how vision position, semantic logits, and patch tokens enter the
      A/B/C hierarchy; replace sampling/resizing rules that are only temporary.
- [x] Add typed configuration objects for environment, model, training,
      evaluation, and checkpoints.
- [x] Define a versioned checkpoint schema shared by every stage.
- [x] Validate stage/model/action/checkpoint compatibility at startup.
- [x] Add contract tests that run the same assertions against every adapter and
      vision encoder.

**Exit criteria:** each stage can be swapped behind the same tested interfaces,
and incompatible dimensions, actions, or checkpoints fail with clear errors.

## P2: Validate Stage 1

Use Synthetic 1D to prove the learning architecture before debugging it inside
a game environment.

- [x] Split deterministic train, validation, and test datasets by fixed seed.
- [x] Add reproducible training configuration and complete seeding.
  - Add a SyntheticTrainingConfig dataclass for seed, split sizes/seeds, batch size, epochs, learning rate, tau schedule, device, deterministic mode.
  - Add seed_everything() covering Python random, NumPy, Torch CPU/GPU, and deterministic Torch algorithms.
  - Use config-derived split seeds and a seeded torch.Generator for torch.randperm(...) each epoch.
  - Add tests proving seeded Python/NumPy/Torch streams reproduce and the train permutation is deterministic.

- [x] Track actor pass 1, actor pass 2, world-model, critic, and total losses.
- [x] Define evaluation metrics and random/simple baseline models.
- [x] Add a short CPU smoke test that verifies finite gradients and decreasing
      loss.
- [x] Add checkpoint saving and restoration using the shared schema.
- [x] Demonstrate that the trained policy beats its declared baselines.
- [x] Document expected runtime and results.

**Exit criteria:** Stage 1 trains reproducibly from a clean checkout, resumes
from a checkpoint, and beats a baseline on a held-out test split.

## P3: Train the Block SMB Agent

Perception is trainable; the missing deliverable is an end-to-end agent that
uses it to complete scenarios.

- [x] Add a supported loader for `data/block_vit/block_vit.pth` and decide
      whether perception is frozen or fine-tuned during policy training.
- [x] Add observation normalization, temporal frame stacking, and episode masks.
- [x] Add focused physics tests for collisions, gaps, moving platforms, coins,
      enemies, goals, death, reset, and truncation.
- [x] Document and tune the existing reward terms rather than adding overlapping
      reward logic in the trainer.
- [x] Implement the complete Block SMB actor/world-model/critic training loop.
- [x] Add trajectory or replay storage with correct recurrent-state boundaries.
- [x] Add numerical checks, gradient clipping, and NaN/exploding-loss detection.
- [x] Add curriculum progression across the four fixed scenarios and generated
      scenarios.
- [x] Report deterministic success rate and return for every fixed scenario.
- [x] Record evaluation trajectories and videos.
- [x] Add vectorized or parallel environments after the single-environment loop
      is correct and reproducible.
- [x] Add a real CLI command for Block SMB training, evaluation, checkpoint
      resume, and recording instead of requiring Python API calls.
- [x] Run real Block SMB policy training with the actual Block ViT checkpoint
      and record the resolved config, seed, metrics, checkpoint path, and
      evaluation artifacts.
- [x] Define and document a success threshold for each fixed Block SMB scenario.
- [x] Save a known-good Block SMB policy checkpoint with deterministic metrics
      and evaluation recordings.
- [x] Revisit the Block SMB learning objective and separate representation,
      dynamics, reward, value, and policy terms where measurements show the
      current return-as-controller-target objective is limiting learning.
- [x] Handle recurrent world-model state boundaries inside the model, not only
      through replay metadata.
- [x] Tune Block SMB reward terms, training config, and hyperparameters against
      deterministic return and success-rate metrics.
- [x] Add deterministic Block ViT perception diagnostics for semantic and
      position bottlenecks before blaming policy training failures.
- [x] Improve or retrain Block ViT perception so the diagnostic no longer
      flags semantic or position quality as policy-training bottlenecks.
- [x] Add Block SMB ablations for vision, critic feedback, hierarchy levels,
      recurrent state handling, and checkpoint transfer.

**Exit criteria:** a seeded Block SMB checkpoint resumes correctly and completes
all fixed scenarios at a documented success threshold.

## P4: Complete the Learning System

Do this after Stage 1 and Block SMB expose concrete failure modes.

- [x] Specify exactly how critic output changes the actor's second pass.
- [x] Investigate whether the low-level adaptive controller parameters should
      be gain schedules.
- [x] Separate representation, dynamics, reward, value, and policy objectives.
- [x] Handle recurrent state and episode boundaries in the world model.
- [x] Implement imagined rollouts through learned dynamics.
- [x] Add target networks or other stabilization only where measurements show
      they are needed.
- [x] Add ablations for vision, world model, critic feedback, hierarchy levels,
      and checkpoint transfer.

**Exit criteria:** each architectural component has a defined loss, tested
gradient path, and measured contribution.

## P5: Integrate Full SMB

The selected backend is `stable-retro`. The existing runner is only a random
loop, and the DeepLab wrapper covers perception but not the stage adapter or
environment semantics.

- [x] Implement `stable-retro` behind a Full SMB environment adapter.
- [x] Map emulator buttons to the P1 shared action vocabulary.
- [x] Extract and test position, score, coins, lives, completion, death,
      termination, and truncation signals.
- [x] Add frame skipping, resizing, normalization, stacking, and episode masks.
- [x] Add emulator state save/load for repeatable evaluation.
- [x] Implement `FullSMBStage` with the common stage contract.
- [x] Replace the existing CNN/DeepLab Full SMB semantic segmentation model
      with a Vision Transformer semantic segmentation model.
  - [x] Get game assets by documenting the required SMB sprite sources, download or
    extraction commands, asset licenses/provenance, and local paths under
    `assets/`.
  - [x] Extract/crop sprites and build a class map for Mario, terrain, items,
    enemies, background, and HUD/unknown regions needed by Full SMB.
  - [x] Create a synthetic training-data generator that composes SMB scenes from
    those assets, emits RGB frames plus pixel or patch semantic labels, and
    writes deterministic train/validation splits.
  - [x] Train or convert a Full SMB ViT segmentation checkpoint to the versioned
    schema with `VisionSpec` metadata and held-out synthetic metrics.
  - [x] Replace `FullSMBSegmentationVision`'s DeepLab implementation with the ViT
    loader while preserving the shared `VisionOutput` contract and checkpoint
    compatibility checks.
- [x] Add headless random-agent and deterministic reset smoke tests.
- [x] Transfer Block SMB perception and policy checkpoints into Full SMB.
- [x] Compare transferred checkpoints with training from scratch.

**Exit criteria:** a clean installation can run deterministic headless
evaluations and load a compatible Block SMB checkpoint.

## P6: Operations and Reproducibility

- [x] Remove the Dockerfile from the project and update Docker-dependent
      build/run scripts and documentation to use the supported native setup.
- [x] Add one CLI for training, evaluation, resume, and environment selection.
- [x] Store resolved configuration, code revision, metrics, and environment
      metadata beside every checkpoint.
- [ ] Add structured logging and periodic deterministic evaluation.
- [ ] Add GitHub Actions for formatting, linting, unit tests, and CPU smoke
      training.
- [ ] Integrate TensorBoard or Weights & Biases behind an optional dependency.
- [ ] Document hardware, runtime, expected metrics, and artifact locations for
      every stage.
- [ ] Publish a reproducibility procedure that starts from a clean checkout.

**Exit criteria:** experiments are launchable through one interface, resumable,
traceable to code/configuration, and checked automatically in CI.

## Definition of Done

The full system is working when a clean checkout can:

1. Install using documented commands.
2. Pass its test suite on CPU.
3. Train Stage 1 reproducibly.
4. Train Block SMB to complete all included scenarios.
5. Load the resulting checkpoint into Full SMB.
6. Run deterministic Full SMB evaluations.
7. Produce metrics, logs, checkpoints, and evaluation videos.
8. Resume an interrupted run without losing experiment state.
