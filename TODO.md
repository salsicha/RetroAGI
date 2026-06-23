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
- [x] Add structured logging and periodic deterministic evaluation.
- [x] Add GitHub Actions for formatting, linting, unit tests, and CPU smoke
      training.
- [x] Integrate TensorBoard or Weights & Biases behind an optional dependency.
- [x] Document hardware, runtime, expected metrics, and artifact locations for
      every stage.
- [x] Publish a reproducibility procedure that starts from a clean checkout.

**Exit criteria:** experiments are launchable through one interface, resumable,
traceable to code/configuration, and checked automatically in CI.

## P7: Progressive Architecture Evaluation

The project now has a working staged implementation of one architecture. This
milestone turns it into a rapid architecture-evaluation system where new model
concepts can start on cheap synthetic data and be promoted through increasingly
high-fidelity evaluations.

- [x] Define an `ArchitectureSpec` that names an architecture, declares its
      model factory, supported stage specs, checkpoint compatibility policy,
      configurable hyperparameters, and expected output contract.
- [x] Define a model-factory protocol that can construct actor/world-model/critic
      variants for any compatible `StageSpec` without trainers importing a
      concrete model class directly.
- [x] Register the current `AgentWorldModelCritic` as the baseline architecture
      so existing behavior remains the first comparable entry.
- [x] Refactor Synthetic 1D training to consume an architecture factory instead
      of instantiating `AgentWorldModelCritic` directly.
- [x] Refactor Block SMB policy training to consume an architecture factory and
      save the selected architecture name and config in every checkpoint and
      run summary.
- [x] Refactor Full SMB transfer and comparison to load architecture-specific
      checkpoints through the registry instead of assuming only
      `AgentWorldModelCritic`.
- [x] Add top-level CLI support for architecture selection, for example
      `--architecture baseline` plus architecture-specific config overrides.
- [x] Give Synthetic 1D top-level CLI parity with Block SMB for seed, epochs,
      device, checkpoint, resume, output summary, and architecture selection.
- [x] Implement a stage-agnostic experiment runner that can execute one
      architecture through selected stages and write one combined result
      manifest with commands, configs, seeds, metrics, checkpoints, logs, and
      pass/fail gates.
- [x] Add a progressive-resolution promotion pipeline:
      1. **Interface smoke:** instantiate the architecture for each declared
         `StageSpec`, run one forward/backward pass, and verify finite gradients.
      2. **Synthetic concept check:** train on deterministic Synthetic 1D data
         and require improvement over random and simple baselines.
      3. **Synthetic stress check:** increase sequence length, noise,
         controller schedule difficulty, and held-out split size before
         promotion.
      4. **Block SMB perception-gated smoke:** run tiny CPU policy training with
         checkpoint transfer disabled to verify trainer compatibility.
      5. **Block SMB fixed-scenario training:** run frozen-perception training,
         deterministic evaluation, and threshold diagnostics on all fixed
         scenarios.
      6. **Block SMB generated-scenario generalization:** add generated
         scenarios and report fixed-vs-generated performance separately.
      7. **Full SMB transfer smoke:** transfer the policy into Full SMB and run
         headless seeded observation checks.
      8. **Full SMB transfer-vs-scratch comparison:** compare action agreement,
         entropy, margins, rewards, resets, terminations, and truncations.
      9. **Full SMB fine-tuning/training:** once available, continue training in
         the emulator and compare against transferred and scratch baselines.
      The initial `retroagi promote` implementation runs interface, Synthetic
      concept, and Block SMB smoke checks, writes one promotion manifest, and
      records later high-fidelity rungs as skipped with prerequisite reasons.
- [x] Define small, medium, and full budgets for each promotion layer so model
      ideas can be rejected quickly before spending emulator time.
      `retroagi promote --budget small|medium|full` now records per-rung
      budgets in the manifest, applies them to runnable rungs, and lets
      explicit CLI overrides replace individual budget values.
- [x] Add automatic promotion gates that stop an architecture when it fails
      required metrics, numerical checks, runtime limits, or artifact checks.
      Promotion rungs now record automatic runtime, finite-number,
      required-metric, and artifact-existence gates; later selected rungs are
      marked stopped after the first gate failure.
- [x] Produce a comparison report for architecture sweeps with per-stage metrics,
      artifact links, pass/fail gates, runtime, selected device, and regression
      deltas against the baseline architecture.
      `retroagi report` now reads experiment or promotion manifests, flattens
      per-stage/rung rows, carries metrics/gates/artifacts/runtime/device, and
      computes numeric deltas against a selected baseline architecture/config.
- [x] Add architecture-level ablation support so variants can disable or replace
      vision, hierarchy, world model, critic feedback, recurrent state, target
      networks, controller schedules, and auxiliary objectives consistently
      across stages.
      `--ablation KEY=VALUE` now normalizes architecture variants for
      experiments and promotions, routes controller schedules through
      architecture config, maps trainer-owned switches to stage CLI flags, and
      records the resolved variant in manifests.
- [x] Define architecture-specific checkpoint schema extensions and migration
      rules so multiple model families can coexist without ambiguous state
      loading.
- [x] Add tests proving incompatible architecture checkpoints fail before state
      loading and compatible checkpoints can transfer across Block SMB and Full
      SMB when their declared contracts match.
- [x] Implement direct Full SMB policy fine-tuning, resume, evaluation, and
      checkpointing so the highest-fidelity rung can measure learning instead
      of only transfer and inference comparison.
- [x] Add a known-good architecture sweep fixture that runs the baseline through
      the smallest promotion pipeline in CI or a documented local command.
- [x] Update the reproducibility procedure with architecture-sweep commands and
      the expected combined manifest.
- [x] Add a Block SMB to Full SMB handoff rung where a policy trained on Block
      SMB can load into Full SMB, run deterministic inference, and continue
      direct Full SMB training from the transferred checkpoint while preserving
      architecture-specific checkpoint contracts.
- [x] Add hierarchical transformer controller transfer support so the Mario
      control policy trains on Block SMB, runs inference unchanged on Full SMB,
      and continues learning on Full SMB from the transferred controller
      checkpoint with explicit metrics for transfer quality and adaptation.
- [x] Add a required Full SMB asset-mock perception adaptation rung between
      Block SMB and Full SMB policy inference/training: compose mock scenarios
      from the full game assets, fine-tune or train the Full SMB ViT on those
      synthetic scenes, validate held-out semantic/position metrics, and gate
      policy transfer on the resulting Full SMB ViT checkpoint.

**Exit criteria:** a new architecture concept can be registered once, launched
through a progressive-resolution experiment command, rejected or promoted by
objective gates at each fidelity layer, and compared against the baseline with
one traceable artifact manifest.

## P8: Multi-Game Generalization

The current system is SMB-centered. This milestone makes the staged fidelity
ladder reusable for other games while preserving the ability to test an
architecture cheaply before spending high-fidelity emulator time.

- [ ] Define a `GameSpec` that names a game family, declares its action space,
      observation sources, semantic classes, reward/signal schema, stage ladder,
      emulator backend, asset requirements, and licensing/provenance rules.
- [ ] Split game-neutral stage contracts from SMB-specific assumptions so
      `StageSpec` no longer depends on `SMBAction` naming or SMB vocabulary
      size checks.
- [ ] Replace the global SMB action vocabulary with per-game `ActionSpec`
      definitions that support discrete buttons, multi-button combos,
      continuous controls, no-op/release behavior, and stable integer IDs.
- [ ] Add action mapping tests proving each game profile maps abstract policy
      actions to backend-native controls without depending on button order.
- [ ] Define a `GameSignalExtractor` protocol for score, progress, health,
      lives, inventory, collectibles, completion, death, timeout, and any
      game-specific objectives.
- [ ] Add a reward-term configuration schema that lets each game own its reward
      terms without leaking SMB reward logic into shared trainers.
- [ ] Define a per-game scenario/task schema for fixed tasks, procedural task
      generation, curriculum progression, success thresholds, and deterministic
      reset seeds.
- [ ] Generalize synthetic low-fidelity data generation so each game can define
      cheap concept data before any pixels or emulator frames are involved.
- [ ] Generalize the Block SMB-style mid-fidelity simulator pattern into
      "block game" adapters: simplified physics, symbolic state, exact semantic
      labels, fast reset, fixed scenarios, and procedural scenarios.
- [ ] Add a game plugin registry that loads game profiles, stage adapters,
      vision encoders, reward configs, success thresholds, and asset pipelines
      by name.
- [ ] Extend the top-level CLI to accept `--game` independently from `--stage`,
      for example `retroagi train --game smb --stage block` and later
      `retroagi train --game <new-game> --stage synthetic|block|full`.
- [ ] Add a stage-resolution naming convention that is game-neutral:
      `synthetic`, `block`, `full`, with optional intermediate rungs such as
      `symbolic`, `tile`, `sprite`, or `emulator` when a game needs them.
- [ ] Define how progressive-resolution architecture promotion composes with
      game promotion:
      1. **Architecture smoke:** validate the model against game-neutral tensor
         contracts.
      2. **Game synthetic:** train on the game's cheap synthetic control task.
      3. **Game block:** train in the game's simplified simulator with exact
         labels and fast deterministic scenarios.
      4. **Game full smoke:** run headless emulator resets, observations, and
         action mappings.
      5. **Game transfer:** transfer the policy from block to full fidelity.
      6. **Game full comparison:** compare transferred, scratch, and prior
         known-good policies on seeded observation streams.
      7. **Game full training:** once supported, fine-tune or train directly in
         the emulator and report threshold metrics.
- [ ] Add game-level promotion gates so each game can define required metrics,
      success thresholds, runtime budgets, artifact checks, and failure reasons
      at each fidelity rung.
- [ ] Add game-level experiment manifests that include architecture, game,
      stage, seed, device, config, backend version, ROM/content identifiers,
      asset provenance, metrics, checkpoints, logs, recordings, and promotion
      decisions.
- [ ] Separate visual perception pipelines by game profile, including semantic
      vocabularies, sprite/asset extraction, synthetic frame composition,
      checkpoint naming, and diagnostic thresholds.
- [ ] Add an asset and licensing checklist for each game profile before assets
      or generated datasets are committed or referenced.
- [ ] Add support for games with no reliable asset pipeline by allowing
      self-supervised, emulator-state, or manually labeled perception datasets
      as alternative vision sources.
- [ ] Add backend abstraction for emulator providers such as `stable-retro`,
      native Python simulators, Gymnasium-compatible envs, or custom adapters.
- [ ] Add deterministic backend capability tests for reset seeding, save/load
      state, frame stepping, action repeat, rendering mode, and headless mode.
- [ ] Add a first non-SMB proof-of-concept game profile with all three planned
      rungs documented, even if only the synthetic and block rungs are initially
      implemented.
- [ ] Add a second-game smoke test that proves the architecture registry,
      action mapping, signal extraction, and experiment manifest are not
      hard-coded to SMB.
- [ ] Update operations and reproducibility docs with multi-game commands,
      artifact layouts, and per-game promotion reports.

**Exit criteria:** adding a new game requires a game profile, action spec,
signal extractor, stage adapters, perception pipeline, and success thresholds,
not trainer rewrites. At least one non-SMB game can run through the synthetic
and block rungs, produce a traceable experiment manifest, and exercise the same
architecture promotion machinery as SMB.

## P9: Full SMB Training And Play

Full SMB currently supports adapter smoke tests, policy transfer, and
transfer-vs-scratch comparison. This milestone turns Full SMB into a trainable
and playable end target.

- [ ] Define the supported Full SMB content setup: required ROM/game identifier,
      stable-retro integration path, local file locations, checksum handling,
      legal/provenance notes, and failure messages when content is missing.
- [ ] Add a Full SMB environment capability check command that verifies backend
      import, game registration, ROM availability, headless reset, render reset,
      save/load state, action stepping, frame skip, and deterministic seeding.
- [ ] Define Full SMB train/eval task sets using emulator states or level starts,
      including short smoke tasks, fixed benchmark tasks, curriculum tasks, and
      held-out generalization tasks.
- [ ] Create and document deterministic Full SMB save-state artifacts for
      starting positions, level sections, death/retry states, and benchmark
      tasks without committing copyrighted content.
- [ ] Define Full SMB success thresholds for each fixed task: progress,
      completion, survival, score/coins, time budget, death count, and minimum
      return.
- [ ] Expand Full SMB signal extraction with tested memory variables or backend
      info fields for x/y position, screen/level, score, coins, lives, power
      state, death, timeout, flag/level completion, and game-over state.
- [ ] Define a Full SMB reward config owned by the Full SMB adapter, separating
      emulator progress, completion, survival, score/coin, enemy, damage, death,
      and frame-penalty terms from Block SMB rewards.
- [ ] Add reward-term breakdowns to Full SMB step info and tests proving the
      reported terms sum to the scalar reward.
- [ ] Verify and tune Full SMB frame preprocessing: crop/resize policy,
      frame-skip, frame stacking, grayscale/RGB choice, normalization, HUD
      handling, episode masks, and camera/scroll position encoding.
- [ ] Add Full SMB perception diagnostics on real emulator frames using the
      Full SMB ViT checkpoint: semantic confidence, class coverage, temporal
      stability, position consistency, and bottleneck flags.
- [ ] Decide whether Full SMB policy training freezes, fine-tunes, or replaces
      Full SMB ViT perception; expose that choice in config and checkpoints.
- [ ] Implement `FullSMBTrainingConfig` with seed, device, rollout length,
      epochs/updates, vector env count, learning rate, loss weights, reward
      config, checkpoint paths, resume paths, recording paths, tracking config,
      and deterministic mode.
- [ ] Implement direct Full SMB policy training from scratch using the shared
      architecture factory and Full SMB stage batches.
- [ ] Implement transferred-policy fine-tuning from a Block SMB or Full SMB
      transfer checkpoint.
- [ ] Add recurrent-state and episode-boundary handling in Full SMB rollouts,
      including death, timeout, level completion, game over, and manual reset.
- [ ] Add rollout/replay storage for Full SMB with saved actions, rewards,
      dones, truncations, episode masks, scenario/task IDs, emulator state IDs,
      and selected signal fields.
- [ ] Add numerical safety checks for Full SMB training: finite losses,
      gradient clipping, reward scale checks, value/reward prediction bounds,
      action entropy tracking, and early stop on NaN or exploding loss.
- [ ] Add periodic deterministic Full SMB evaluation during training, separate
      from stochastic exploration rollouts.
- [ ] Save versioned Full SMB trainer checkpoints with model, optimizer,
      optional perception, RNG state, task/curriculum state, config, metrics,
      backend metadata, and source checkpoint provenance.
- [ ] Implement robust Full SMB resume so interrupted training can continue
      with the same task schedule, RNG streams, optimizer, recurrent-state
      expectations, and tracking destination.
- [ ] Add Full SMB recording support for evaluation episodes, including frame
      arrays, actions, rewards, signals, task IDs, and optional video export.
- [ ] Add `retroagi train --stage full-smb` with scratch and fine-tune modes.
- [ ] Add `retroagi resume --stage full-smb` with checkpoint and optional output
      checkpoint paths.
- [ ] Add `retroagi evaluate --stage full-smb --checkpoint ...` that loads a
      saved policy and reports fixed-task threshold diagnostics.
- [ ] Add `retroagi record --stage full-smb --checkpoint ...` that records
      deterministic policy rollouts and writes artifacts.
- [ ] Add `retroagi play --stage full-smb --checkpoint ...` for interactive
      playback of a trained policy with render, speed, pause, reset, deterministic
      or sampling mode, and optional recording.
- [ ] Add a human-control mode under `retroagi play --stage full-smb --human`
      for manual debugging of action mapping, reward signals, rendering, and
      save-state starts.
- [ ] Add CLI options for selecting Full SMB task set, level/state, render mode,
      max steps, frame skip, action repeat, deterministic policy, and recording
      output.
- [ ] Add a policy-inspection overlay for play mode showing action name,
      action probabilities, reward terms, score/progress signals, termination
      reason, and current task threshold status.
- [ ] Add comparison commands for transferred, scratch-trained, fine-tuned, and
      known-good policies on identical seeded Full SMB task streams.
- [ ] Define Full SMB artifact layout under `artifacts/full_smb/<run>/` for
      summaries, structured logs, recordings, videos, evaluation reports,
      comparison reports, and tracking outputs.
- [ ] Add Full SMB CPU smoke tests using minimal steps and mocked or tiny backend
      paths where real emulator content is unavailable in CI.
- [ ] Add local integration tests, skipped unless content is available, that run
      real stable-retro reset/step/evaluate/play smoke paths.
- [ ] Add tests for Full SMB checkpoint compatibility, resume behavior,
      transfer-to-train loading, fixed-task threshold diagnostics, and play-mode
      policy loading.
- [ ] Benchmark emulator throughput and document recommended CPU, CUDA, and MPS
      settings for Full SMB training and play.
- [ ] Update operations and reproducibility docs with Full SMB train, resume,
      evaluate, record, play, and comparison commands.
- [ ] Produce one known-good Full SMB policy artifact or documented benchmark
      run that can be loaded, evaluated, recorded, and played locally.

**Exit criteria:** a supported local setup can train or fine-tune a Full SMB
policy, resume it, evaluate it against documented fixed-task thresholds, record
deterministic rollouts, and play the saved policy with live rendering and
diagnostic overlays.

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
9. Train or fine-tune a Full SMB policy.
10. Play a saved Full SMB policy with rendering and diagnostics.
