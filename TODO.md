# RetroAGI TODO

This roadmap prioritizes making RetroAGI reproducible, testable, and usable as
an end-to-end curriculum from synthetic data through full Super Mario Bros.

## P0: Establish a Reliable Baseline

- [ ] Remove tracked `.venv/` files and keep virtual environments excluded.
- [ ] Move large models and archives to Git LFS or external artifact storage.
- [ ] Add pinned dependencies with supported Python and CUDA versions.
- [ ] Add `pytest`, formatting, and linting dependencies.
- [ ] Package the project with `pyproject.toml`.
- [ ] Make tests run without manually setting `PYTHONPATH`.
- [ ] Add CPU-only installation and execution paths.
- [ ] Verify the Docker build from a clean checkout.
- [ ] Document ROM import requirements without distributing copyrighted ROMs.
- [ ] Archive or delete the obsolete `ai_experiment` branch.

## P1: Define the Full-System Contract

- [ ] Specify observation, action, reward, and termination semantics for every
      stage.
- [ ] Document tensor shapes and timescales in `StageSpec` and `StageBatch`.
- [ ] Define a common environment adapter interface.
- [ ] Keep actor, world model, and critic interfaces independent of
      Mario-specific code.
- [ ] Add configuration objects for model dimensions, training, environments,
      and checkpoints.
- [ ] Validate and reject incompatible stage and model configurations.

## P2: Complete Stage 1

- [ ] Create deterministic training, validation, and test datasets.
- [ ] Add reproducible seeding.
- [ ] Add checkpoint saving and restoration.
- [ ] Track actor, world-model, critic, and total losses independently.
- [ ] Add evaluation metrics and baseline models.
- [ ] Add a short smoke-training test that verifies loss decreases.
- [ ] Demonstrate that learned policies outperform random actions.

## P3: Make Stage 2 Trainable

- [ ] Implement a complete Block SMB training loop.
- [ ] Standardize actions between Block SMB and Full SMB.
- [ ] Define reward shaping for movement, coins, survival, and completion.
- [ ] Add observation normalization and frame stacking.
- [ ] Implement vectorized or parallel environments.
- [ ] Add curriculum progression across scenario files.
- [ ] Add success-rate evaluation for each scenario.
- [ ] Record evaluation videos and trajectories.
- [ ] Test collisions, gaps, goals, death, reset, and truncation behavior.

## P4: Integrate Full SMB

- [ ] Verify `stable-retro` environment creation from a clean installation.
- [ ] Create a Full SMB adapter that implements the common stage contract.
- [ ] Map emulator buttons to the shared action space.
- [ ] Extract reliable position, score, life, and terminal-state signals.
- [ ] Add frame skipping, resizing, normalization, and stacking.
- [ ] Add emulator-state saving for repeatable evaluation.
- [ ] Implement a random-agent smoke test.
- [ ] Transfer Stage 2 checkpoints into Stage 3.
- [ ] Compare transferred policy performance with training from scratch.

## P5: Finish the Learning System

- [ ] Define precisely how actor refinement uses critic and world-model
      outputs.
- [ ] Implement imagined rollouts through the world model.
- [ ] Add target networks or other stabilization where required.
- [ ] Separate representation, dynamics, reward, value, and policy losses.
- [ ] Add replay-buffer or trajectory-storage abstractions.
- [ ] Handle recurrent state and episode boundaries correctly.
- [ ] Add gradient clipping and numerical checks.
- [ ] Detect NaNs, collapsed policies, and exploding losses.
- [ ] Create ablations showing whether each architectural component
      contributes.

## P6: Operations and Reproducibility

- [ ] Add a single CLI, such as
      `python -m retroagi train --stage block_smb`.
- [ ] Store experiment configuration beside every checkpoint.
- [ ] Integrate TensorBoard or Weights & Biases.
- [ ] Add structured logging and periodic evaluation.
- [ ] Support resuming interrupted training.
- [ ] Add checkpoint versioning and compatibility checks.
- [ ] Add GitHub Actions for linting, unit tests, and CPU smoke tests.
- [ ] Document hardware, runtime, and expected results for each stage.

## Definition of Done

The full system is working when a clean checkout can:

1. Install or build using documented commands.
2. Pass its test suite on CPU.
3. Train Stage 1 reproducibly.
4. Train Block SMB to complete all included scenarios.
5. Load the resulting checkpoint into Full SMB.
6. Run deterministic Full SMB evaluations.
7. Produce metrics, logs, checkpoints, and evaluation videos.
8. Resume an interrupted run without losing experiment state.
