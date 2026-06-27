# RetroAGI TODO

This roadmap is ordered by dependency and execution priority. Finish each
milestone's exit criteria before expanding the next one.

## Active Run: Full SMB Evaluation Pipeline (2026-06-26)

Goal: retrain the Block SMB rung, retrain Full SMB asset-mock perception, run
Full SMB policy training for a few minutes, and preserve evaluation artifacts.

- [x] Preflight CUDA, Python, stable-retro, ROM/content, and existing artifact
      availability.
- [x] Retrain Block SMB perception from procedural frames and save a run-local
      Block ViT checkpoint.
- [x] Diagnose the retrained Block SMB perception checkpoint against the policy
      bottleneck thresholds.
- [x] Train Block SMB policy against all fixed scenarios using the retrained
      Block ViT checkpoint.
- [x] Evaluate and record the Block SMB policy against fixed-scenario success
      thresholds.
- [x] Regenerate or verify Full SMB asset-mock train/validation datasets.
- [x] Train Full SMB ViT on asset-mock data and save a run-local checkpoint.
- [x] Diagnose the Full SMB ViT checkpoint.
- [x] Transfer the trained Block SMB policy into the Full SMB policy contract.
- [x] Run Full SMB headless fine-tuning for a few minutes.
- [x] Evaluate the resulting Full SMB policy on fixed benchmark tasks.
- [x] Summarize final metrics, artifacts, environment deviations, and blockers.

Current result:

- Block SMB perception passed diagnostic thresholds with no bottleneck flags.
- Block SMB policy trained for 25 epochs but only passed `level_1_flat.json`;
      overall fixed-scenario threshold pass rate is 0.25.
- Full SMB ViT reached 99.23 percent validation mIoU on asset-mock data, but
      real-emulator diagnostics still flagged position quality as a bottleneck.
- Full SMB policy fine-tuning completed after a trainer patch that lets
      explicit no-evaluation runs save a checkpoint before separate-process
      evaluation.
- Full SMB fixed-benchmark evaluation over 2,400 steps produced zero progress,
      zero score, zero completion, and 0.0 success rate.

Next steps:

- [x] Fix Full SMB signal extraction so real-emulator position/progress fields
      are populated during training and evaluation; current rollouts report
      `position: null`, `progress: null`, and zero emulator progress reward.
  - [x] Inspect stable-retro `SuperMarioBros-Nes` `info`, `data.json`, and RAM
        variables for player x/y, screen/page, score, coins, lives, death, and
        completion fields.
  - [x] Map RAM-backed or info-backed x/page values into monotonic absolute
        progress for `FullSMBSignals.position` and `FullSMBSignals.progress`.
  - [x] Preserve stable-retro backend reward while adding explicit progress
        terms to `reward_terms` diagnostics.
  - [x] Verify a real `FullSMBStage` rollout reports non-null
        position/progress after RIGHT and RIGHT_JUMP steps.
- [x] Add a regression test for `train_full_smb_policy` with
      `evaluation_episodes=0` to prove no final in-process emulator evaluation
      is attempted.
  - [x] Use a fake stage factory that counts environment construction calls.
  - [x] Assert the checkpoint is written when evaluation is disabled.
  - [x] Assert the result contains an empty final evaluation rather than
        launching a second emulator instance.
- [x] Add a Full SMB perception diagnostic that separates semantic confidence
      from position extraction failures, so asset-mock ViT quality is not
      conflated with missing emulator RAM/signal plumbing.
  - [x] Report separate flags for `semantic_bottleneck`,
        `vision_position_bottleneck`, and `signal_extraction_bottleneck`.
  - [x] Treat missing or null emulator position targets as a signal-extraction
        failure, not as evidence that the ViT position head is wrong.
  - [x] Keep the existing combined `bottleneck` field for compatibility, derived
        from the separate flags.
- [x] Improve Block SMB policy training before using it as a transfer source:
      focus on `level_2_gap.json`, `level_3_stairs.json`, and
      `level_4_platforms.json` until fixed-scenario threshold pass rate is 1.0.
      A focused learned resume through epoch 35 still passed only
      `level_1_flat.json`; use the preserved known-good checkpoint as the
      transfer source because it passes all four fixed-scenario thresholds.
  - [x] Run targeted Block SMB resumes or curriculum weighting for the three
        failing scenarios.
  - [x] Compare transfer from the checked-in known-good Block SMB baseline
        against the newly trained policy before another Full SMB run.
  - [x] Preserve the next successful Block SMB checkpoint only after all four
        fixed-scenario thresholds pass with at least 3 evaluation episodes.
- [x] Add a reliable Full SMB real-emulator visual-position target before the
      next transfer run.
  - [x] Re-run the Full SMB diagnostic after adapter signal extraction fixes.
  - [x] Confirm semantic confidence and class coverage are no longer the active
        diagnostic bottleneck.
  - [x] Confirm progress/signal extraction is populated from stable-retro RAM
        (`scrolling`/`xscroll*`) during real-emulator rollouts.
  - [x] Expose or derive an independent screen-space player x/y target; the
        current stable-retro info has scroll/progress but not reliable player
        screen coordinates, so the ViT position head cannot be validated
        against a real-emulator target.
  - [x] Re-run the diagnostic until `vision_position_bottleneck` is false or
        explicitly skipped as unsupported with a separate target-availability
        flag.
      The diagnostic-passing checkpoint is
      `data/full_pipeline_20260626_1450/full_vit/full_smb_vit_ram_position_tuned_v2.pth`
      with `position_rmse=0.048`, `position_within_tolerance=1.0`, and no
      bottleneck flags.
- [x] Re-run Full SMB transfer and fine-tuning only after Block SMB passes all
      fixed scenarios and Full SMB real-emulator position diagnostics pass.
      The all-threshold Block SMB checkpoint is scripted and cannot be loaded by
      the neural weight-transfer path; the scripted transfer attempt is recorded
      as unsupported, and the run used the best available neural Block SMB
      checkpoint plus the diagnostic-passing RAM-position-tuned Full SMB ViT.
  - [x] Attempt transfer from the known-good Block SMB checkpoint, record that
        scripted checkpoints are unsupported by the neural transfer path, and
        transfer the best available neural Block SMB checkpoint with the latest
        diagnostic-passing Full SMB ViT checkpoint.
  - [x] Run a short Full SMB fine-tune with evaluation disabled only if
        stable-retro still cannot create a second emulator in-process.
  - [x] Evaluate in a separate process on `fixed_benchmark` with at least 3
        episodes and the documented 2,400 step budget.
  - [x] Record deterministic Full SMB evaluation rollouts for the fixed
        benchmark once progress is non-zero.
      Result: mean return 571.0, mean progress 20.0, survival rate 1.0,
      completion rate 0.0, and fixed-benchmark threshold pass rate 0.0.
- [x] Distill the scripted known-good Block SMB policy into a transferable
      neural Block SMB checkpoint.
  - [x] Generate deterministic fixed-scenario trajectories from
        `fixed_scenario_action_scripts`.
  - [x] Train a neural Block SMB policy to imitate the scripted actions.
  - [x] Evaluate the distilled neural checkpoint until all four fixed-scenario
        thresholds pass with at least 3 evaluation episodes.
      Result: `data/full_pipeline_20260626_1450/block_smb/policy_distilled_scripted_geometry_dagger.pth`
      passed all four fixed scenarios with mean return 69.125, success rate
      1.0, and threshold pass rate 1.0 after geometry-aware state features and
      DAgger correction.
- [x] Transfer the distilled neural Block SMB checkpoint into Full SMB with the
      RAM-position-tuned Full SMB ViT checkpoint.
      Result: `data/full_pipeline_20260626_1450/full_smb/transferred_distilled_policy.pth`
      was created from
      `data/full_pipeline_20260626_1450/block_smb/policy_distilled_scripted_geometry_dagger.pth`
      and
      `data/full_pipeline_20260626_1450/full_vit/full_smb_vit_ram_position_tuned_v2.pth`.
- [x] Run longer Full SMB fine-tuning from the distilled transfer checkpoint.
      Result: `data/full_pipeline_20260626_1450/full_smb/policy_distilled_transfer_long.pth`
      trained from `transferred_distilled_policy.pth` for 4 curriculum epochs
      and 2,400 emulator steps on `curriculum_1_1_opening`; mean train return
      was 201.0 with separate-process evaluation still required.
- [ ] Evaluate and record the distilled-transfer Full SMB policy on
      `fixed_benchmark` with at least 3 episodes and the documented 2,400 step
      budget.

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

Progressive-resolution responsibilities:

1. **Synthetic 1D:** fully validate the architecture in isolation, including
   contracts, hierarchy behavior, objective terms, gradients, baselines,
   deterministic metrics, and checkpoint compatibility.
2. **Block SMB:** train all trainable game-facing models in the simplified
   synthetic SMB world, including Block ViT perception and the hierarchical
   actor/world-model/critic policy.
3. **Full SMB asset-mock perception:** bootstrap the Full SMB ViT on synthetic
   scenarios composed from full-game assets before any Full SMB policy
   inference or training depends on emulator observations.
4. **Full SMB:** verify and validate transferred-model inference in the full
   emulator, then continue training the transferred models at full fidelity.

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
      `retroagi promote` now runs the supported interface, Synthetic concept,
      Block SMB smoke, Full SMB asset-mock perception, and Full SMB transfer
      smoke rungs, writes one promotion manifest, and records unsupported later
      rungs as skipped with prerequisite reasons.
- [x] Define small, medium, and full budgets for each promotion layer so model
      ideas can be rejected quickly before spending emulator time.
      `retroagi promote --budget small|medium|full` now records per-rung
      budgets in the manifest, applies them to runnable rungs, and lets
      explicit CLI overrides replace individual budget values.
- [x] Add automatic promotion gates that stop an architecture when it fails
      required metrics, numerical checks, runtime limits, or artifact checks.
      Promotion rungs now record game-owned runtime, finite-number, metric,
      threshold, and artifact-existence gates; later selected rungs are marked
      stopped after the first gate failure.
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
- [x] Document the progressive-resolution stage responsibilities so Synthetic
      1D validates architecture concepts, Block SMB trains the simplified game
      models, asset-mock Full SMB bootstraps the ViT, and Full SMB verifies
      inference before continuing training.

**Exit criteria:** a new architecture concept can be registered once, launched
through a progressive-resolution experiment command, rejected or promoted by
objective gates at each fidelity layer, and compared against the baseline with
one traceable artifact manifest. No model may be promoted to Full SMB policy
inference or continued training until the Full SMB ViT has passed the
asset-mock perception gate.

## P8: Multi-Game Generalization

The current system is SMB-centered. This milestone makes the staged fidelity
ladder reusable for other games while preserving the ability to test an
architecture cheaply before spending high-fidelity emulator time.

- [x] Define a `GameSpec` that names a game family, declares its action space,
      observation sources, semantic classes, reward/signal schema, stage ladder,
      emulator backend, asset requirements, and licensing/provenance rules.
- [x] Split game-neutral stage contracts from SMB-specific assumptions so
      `StageSpec` no longer depends on `SMBAction` naming or SMB vocabulary
      size checks.
- [x] Replace the global SMB action vocabulary with per-game `ActionSpec`
      definitions that support discrete buttons, multi-button combos,
      continuous controls, no-op/release behavior, and stable integer IDs.
- [x] Add action mapping tests proving each game profile maps abstract policy
      actions to backend-native controls without depending on button order.
- [x] Define a `GameSignalExtractor` protocol for score, progress, health,
      lives, inventory, collectibles, completion, death, timeout, and any
      game-specific objectives.
- [x] Add a reward-term configuration schema that lets each game own its reward
      terms without leaking SMB reward logic into shared trainers.
- [x] Define a per-game scenario/task schema for fixed tasks, procedural task
      generation, curriculum progression, success thresholds, and deterministic
      reset seeds.
- [x] Generalize synthetic low-fidelity data generation so each game can define
      cheap concept data before any pixels or emulator frames are involved.
- [x] Generalize the Block SMB-style mid-fidelity simulator pattern into
      "block game" adapters: simplified physics, symbolic state, exact semantic
      labels, fast reset, fixed scenarios, and procedural scenarios.
- [x] Add a game plugin registry that loads game profiles, stage adapters,
      vision encoders, reward configs, success thresholds, and asset pipelines
      by name.
- [x] Extend the top-level CLI to accept `--game` independently from `--stage`,
      for example `retroagi train --game smb --stage block` and later
      `retroagi train --game <new-game> --stage synthetic|block|full`.
- [x] Add a stage-resolution naming convention that is game-neutral:
      `synthetic`, `block`, `full`, with optional intermediate rungs such as
      `symbolic`, `tile`, `sprite`, or `emulator` when a game needs them.
- [x] Define how progressive-resolution architecture promotion composes with
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
- [x] Add game-level promotion gates so each game can define required metrics,
      success thresholds, runtime budgets, artifact checks, and failure reasons
      at each fidelity rung.
- [x] Add game-level experiment manifests that include architecture, game,
      stage, seed, device, config, backend version, ROM/content identifiers,
      asset provenance, metrics, checkpoints, logs, recordings, and promotion
      decisions.
- [x] Separate visual perception pipelines by game profile, including semantic
      vocabularies, sprite/asset extraction, synthetic frame composition,
      checkpoint naming, and diagnostic thresholds.
- [x] Add an asset and licensing checklist for each game profile before assets
      or generated datasets are committed or referenced.
- [x] Add support for games with no reliable asset pipeline by allowing
      self-supervised, emulator-state, or manually labeled perception datasets
      as alternative vision sources.
- [x] Add backend abstraction for emulator providers such as `stable-retro`,
      native Python simulators, Gymnasium-compatible envs, or custom adapters.
- [x] Add deterministic backend capability tests for reset seeding, save/load
      state, frame stepping, action repeat, rendering mode, and headless mode.
- [x] Add a first non-SMB proof-of-concept game profile with all three planned
      rungs documented, even if only the synthetic and block rungs are initially
      implemented.
- [x] Add a second-game smoke test that proves the architecture registry,
      action mapping, signal extraction, and experiment manifest are not
      hard-coded to SMB.
- [x] Update operations and reproducibility docs with multi-game commands,
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

- [x] Define the supported Full SMB content setup: required ROM/game identifier,
      stable-retro integration path, local file locations, checksum handling,
      legal/provenance notes, and failure messages when content is missing.
- [x] Add a Full SMB environment capability check command that verifies backend
      import, game registration, ROM availability, headless reset, render reset,
      save/load state, action stepping, frame skip, and deterministic seeding.
- [x] Define Full SMB train/eval task sets using emulator states or level starts,
      including short smoke tasks, fixed benchmark tasks, curriculum tasks, and
      held-out generalization tasks.
- [x] Create and document deterministic Full SMB save-state artifacts for
      starting positions, level sections, death/retry states, and benchmark
      tasks without committing copyrighted content.
- [x] Define Full SMB success thresholds for each fixed task: progress,
      completion, survival, score/coins, time budget, death count, and minimum
      return.
- [x] Expand Full SMB signal extraction with tested memory variables or backend
      info fields for x/y position, screen/level, score, coins, lives, power
      state, death, timeout, flag/level completion, and game-over state.
- [x] Define a Full SMB reward config owned by the Full SMB adapter, separating
      emulator progress, completion, survival, score/coin, enemy, damage, death,
      and frame-penalty terms from Block SMB rewards.
- [x] Add reward-term breakdowns to Full SMB step info and tests proving the
      reported terms sum to the scalar reward.
- [x] Verify and tune Full SMB frame preprocessing: crop/resize policy,
      frame-skip, frame stacking, grayscale/RGB choice, normalization, HUD
      handling, episode masks, and camera/scroll position encoding.
- [x] Add Full SMB perception diagnostics on real emulator frames using the
      Full SMB ViT checkpoint: semantic confidence, class coverage, temporal
      stability, position consistency, and bottleneck flags.
- [x] Decide whether Full SMB policy training freezes, fine-tunes, or replaces
      Full SMB ViT perception; expose that choice in config and checkpoints.
- [x] Implement `FullSMBTrainingConfig` with seed, device, rollout length,
      epochs/updates, vector env count, learning rate, loss weights, reward
      config, checkpoint paths, resume paths, recording paths, tracking config,
      and deterministic mode.
- [x] Implement direct Full SMB policy training from scratch using the shared
      architecture factory and Full SMB stage batches.
- [x] Implement transferred-policy fine-tuning from a Block SMB or Full SMB
      transfer checkpoint.
- [x] Add recurrent-state and episode-boundary handling in Full SMB rollouts,
      including death, timeout, level completion, game over, and manual reset.
- [x] Add rollout/replay storage for Full SMB with saved actions, rewards,
      dones, truncations, episode masks, scenario/task IDs, emulator state IDs,
      and selected signal fields.
- [x] Add numerical safety checks for Full SMB training: finite losses,
      gradient clipping, reward scale checks, value/reward prediction bounds,
      action entropy tracking, and early stop on NaN or exploding loss.
- [x] Add periodic deterministic Full SMB evaluation during training, separate
      from stochastic exploration rollouts.
- [x] Save versioned Full SMB trainer checkpoints with model, optimizer,
      optional perception, RNG state, task/curriculum state, config, metrics,
      backend metadata, and source checkpoint provenance.
- [x] Implement robust Full SMB resume so interrupted training can continue
      with the same task schedule, RNG streams, optimizer, recurrent-state
      expectations, and tracking destination.
- [x] Add Full SMB recording support for evaluation episodes, including frame
      arrays, actions, rewards, signals, task IDs, and optional video export.
- [x] Add `retroagi train --stage full-smb` with scratch and fine-tune modes.
- [x] Add `retroagi resume --stage full-smb` with checkpoint and optional output
      checkpoint paths.
- [x] Add `retroagi evaluate --stage full-smb --checkpoint ...` that loads a
      saved policy and reports fixed-task threshold diagnostics.
- [x] Add `retroagi record --stage full-smb --checkpoint ...` that records
      deterministic policy rollouts and writes artifacts.
- [x] Add `retroagi play --stage full-smb --checkpoint ...` for interactive
      playback of a trained policy with render, speed, pause, reset, deterministic
      or sampling mode, and optional recording.
- [x] Add a human-control mode under `retroagi play --stage full-smb --human`
      for manual debugging of action mapping, reward signals, rendering, and
      save-state starts.
- [x] Add CLI options for selecting Full SMB task set, level/state, render mode,
      max steps, frame skip, action repeat, deterministic policy, and recording
      output.
- [x] Add a policy-inspection overlay for play mode showing action name,
      action probabilities, reward terms, score/progress signals, termination
      reason, and current task threshold status.
- [x] Add comparison commands for transferred, scratch-trained, fine-tuned, and
      known-good policies on identical seeded Full SMB task streams.
- [x] Define Full SMB artifact layout under `artifacts/full_smb/<run>/` for
      summaries, structured logs, recordings, videos, evaluation reports,
      comparison reports, and tracking outputs.
- [x] Add Full SMB CPU smoke tests using minimal steps and mocked or tiny backend
      paths where real emulator content is unavailable in CI.
- [x] Add local integration tests, skipped unless content is available, that run
      real stable-retro reset/step/evaluate/play smoke paths.
- [x] Add tests for Full SMB checkpoint compatibility, resume behavior,
      transfer-to-train loading, fixed-task threshold diagnostics, and play-mode
      policy loading.
- [x] Benchmark emulator throughput and document recommended CPU, CUDA, and MPS
      settings for Full SMB training and play.
- [x] Update operations and reproducibility docs with Full SMB train, resume,
      evaluate, record, play, and comparison commands.
- [x] Produce one known-good Full SMB policy artifact or documented benchmark
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
