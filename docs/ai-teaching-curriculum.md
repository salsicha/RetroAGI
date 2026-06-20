# Teaching AI With RetroAGI

This curriculum uses RetroAGI as a practical path from supervised sequence
learning to environment-backed reinforcement learning. It is designed for a
10-week course, reading group, or self-study sequence where students learn by
modifying a real codebase rather than isolated notebooks.

## Audience

Students should already be comfortable with:

- Python and basic command-line workflows.
- PyTorch tensors, modules, optimizers, and losses.
- Basic supervised learning concepts.
- Reading tests as executable specifications.

Helpful but not required:

- Reinforcement learning terminology.
- Vision model basics.
- Game environment APIs such as Gymnasium.

## Course Outcomes

By the end, students should be able to:

- Explain RetroAGI's actor, world-model, critic, and A/B/C timescale structure.
- Trace how stage-native observations become shared `StageBatch` tensors.
- Train and evaluate the Synthetic 1D stage reproducibly.
- Explain why perception, reward, dynamics, value, and policy objectives should
  be separated.
- Modify Block SMB reward, observation, and episode contracts without adding
  hidden trainer-side behavior.
- Design a measurable Block SMB policy-training experiment.
- Evaluate whether a change improved learning using deterministic tests,
  metrics, and saved configuration.

## Ground Rules

- Treat `docs/stage-semantics.md` and `docs/tensor-contracts.md` as the
  interface contract before changing code.
- Keep reward shaping in the environment. Trainers should consume the scalar
  reward returned by `step` unless a new multi-objective interface is explicitly
  designed.
- Prefer small experiments with fixed seeds before large training runs.
- Every project change should add or update a test that would catch a regression.
- Do not tune by anecdote. Record the config, seed, metric, and checkpoint path.

## Setup Checkpoint

Students should complete this before week 1:

```bash
python3 -m unittest discover -s scripts/tests -v
python -m retroagi.stages.synthetic_1d.train
```

Expected result: the test suite passes, and the Synthetic 1D trainer produces a
finite training history and checkpoint-compatible metrics.

## Weekly Plan

| Week | Topic | Project Area | Lab Deliverable |
| --- | --- | --- | --- |
| 1 | Repository orientation and contracts | `retroagi/core`, `docs/*` | Draw the data path from observation to `StageBatch` to model outputs |
| 2 | Synthetic 1D data and hierarchy | `retroagi/stages/synthetic_1d` | Add one dataset-shape or seed-reproducibility test |
| 3 | Reproducible training | Synthetic 1D trainer | Run a seeded training comparison and report metrics |
| 4 | Vision as representation | `BlockVisionTransformer`, `VisionOutput` | Inspect one Block SMB frame and its semantic/token outputs |
| 5 | Environment semantics | `MarioScenarioEnv`, `BlockSMBStage` | Explain one transition's observation, action, reward, termination, and info |
| 6 | Rewards and episode boundaries | Block SMB reward config and tests | Tune one reward term and show the `reward_terms` breakdown still sums correctly |
| 7 | World models and critic feedback | `AgentWorldModelCritic` | Propose a loss decomposition for dynamics, value, reward, and policy |
| 8 | Policy-training loop design | P3 TODOs | Write a minimal single-environment training-loop design doc |
| 9 | Evaluation and curriculum progression | Fixed and generated scenarios | Define success-rate and return metrics for each fixed scenario |
| 10 | Transfer to Full SMB | `retroagi/stages/full_smb` | Identify which Block SMB contracts must be preserved at the emulator boundary |

## Module Details

### 1. Contracts Before Models

Start with the shared `StageAdapter`, `StageSpec`, `StageBatch`, `VisionSpec`,
and `VisionOutput` contracts. Students should understand that the stage adapter
is the boundary between a game-specific environment and the shared learning
system.

Exercise:

- Pick one field from `StageBatch`.
- Find where it is produced.
- Find where it is consumed.
- Write down its shape, dtype, range, and failure mode if the contract changes.

Assessment:

- Student can explain why contract validation catches model/stage mismatch
  before state-dict loading.

### 2. Synthetic 1D As A Control Experiment

Synthetic 1D is the simplest place to study the hierarchy without game physics,
vision noise, sparse rewards, or episode boundaries.

Exercise:

- Run the Synthetic 1D tests.
- Change only the dataset seed.
- Compare controller MSE, A-token accuracy, and B-parameter error.

Assessment:

- Student can distinguish model improvement from a changed data split.
- Student can explain why deterministic data generation matters.

### 3. Reproducibility And Checkpoints

Focus on seeded dataset splits, seeded train permutations, checkpoint schemas,
and resume equivalence.

Exercise:

- Train for one epoch with checkpoint saving enabled.
- Resume for another epoch.
- Compare the resumed path against uninterrupted training.

Assessment:

- Student can identify which state must be saved: model, optimizer, epoch,
  global step, metrics, config, and compatibility specs.

### 4. Perception As A Stage Boundary

Block SMB perception converts RGB frames into position, semantic logits,
semantic IDs, and tokens. The hierarchy should not need to know whether those
features came from synthetic labels, a ViT, or a future emulator adapter.

Exercise:

```bash
python scripts/vit/train_block_vit.py --epochs 1 --samples-per-epoch 128 --val-samples 32
```

Assessment:

- Student can explain how semantic logits feed A/B streams and how position,
  semantics, state, and patch tokens fill C-stream slots.

### 5. Environment Semantics And Rewards

Use Block SMB to teach the difference between game state, observation, reward,
termination, truncation, and diagnostics.

Exercise:

- Create a one-coin scenario.
- Step until coin collection.
- Inspect `reward`, `info["reward_terms"]`, `info["reward_total"]`, and
  `info["reward_config"]`.

Assessment:

- Student can explain why reward terms are logged for diagnosis but the scalar
  `reward` is the trainer's optimization signal.

### 6. Episode Boundaries And Temporal Context

Frame stacks and episode masks make temporal context explicit. This module
prepares students for recurrent state and replay-buffer boundaries.

Exercise:

- Reset a Block SMB stage.
- Encode observations before and after a truncation.
- Inspect `metadata["observation"]["frame_mask"]` and
  `metadata["episode"]["mask"]`.

Assessment:

- Student can explain why timeout should set `truncated`, not silently rewrite
  the transition as `terminated`.

### 7. Separating Learning Objectives

The project should not hide representation, dynamics, reward, value, and policy
learning inside one opaque loss. This module is design-heavy and should produce
a proposal before code.

Exercise:

- Write a table with each objective, its prediction target, loss function,
  source tensors, and gradient path.
- Include where critic feedback changes the actor's second pass.

Assessment:

- Student can explain why adding more scalar reward terms in the trainer is not
  the same thing as learning a value or reward model.

### 8. Designing The Block SMB Training Loop

This module converts the previous contracts into a minimal single-environment
training design.

Required design points:

- Reset and step lifecycle.
- Action sampling policy.
- Trajectory or replay storage.
- Episode-mask handling.
- Reward source.
- Gradient clipping and NaN checks.
- Checkpoint/resume state.
- Deterministic evaluation cadence.

Assessment:

- Student can defend what is in the first training loop and what is explicitly
  deferred.

### 9. Measuring Scenario Progress

Students should define evaluation before optimizing. The fixed scenarios should
be used as regression tests for behavior, not just demos.

Exercise:

- Define success, return, timeout rate, death rate, coin count, and max progress
  for each fixed scenario.
- Decide which metrics are logged every epoch and which are reserved for final
  evaluation.

Assessment:

- Student can explain why deterministic evaluation uses fixed seeds and
  separate training/evaluation scenarios.

### 10. Transfer To Full SMB

Full SMB should be introduced as a contract-transfer problem. The first goal is
not maximum score; it is preserving action, observation, reward, termination,
truncation, reset, and checkpoint compatibility at the emulator boundary.

Exercise:

- Compare Block SMB and Full SMB stage semantics.
- Identify which game variables are needed to reproduce Block SMB diagnostics.
- Write the adapter acceptance tests before implementing the adapter.

Assessment:

- Student can explain what should be normalized at the Full SMB adapter
  boundary and what must not leak into shared training code.

## Capstone Options

Choose one:

- Implement the initial single-environment Block SMB policy-training loop.
- Add deterministic evaluation reports for the four fixed Block SMB scenarios.
- Add a replay buffer with correct episode-boundary handling.
- Add a reward-model or value-model objective without changing environment
  reward semantics.
- Prototype a Full SMB adapter acceptance-test suite.

## Grading Rubric

| Area | Criteria |
| --- | --- |
| Contract discipline | Changes preserve documented shapes, dtypes, action vocabulary, and reset/step semantics |
| Reproducibility | Experiments record seeds, configs, metrics, code revision, and checkpoint paths |
| Measurement | Claims are supported by deterministic tests or evaluation metrics |
| Code quality | Changes are scoped, tested, and compatible with existing stage abstractions |
| Technical explanation | Student can explain why the chosen objective or interface is correct |

## Instructor Notes

- Keep early assignments small. The project already has enough moving parts; do
  not start with emulator integration.
- Require students to read failing tests before changing code.
- Reward experiments should modify `BlockSMBRewardConfig`, not trainer-local
  reward arithmetic.
- Treat TODO items as design prompts. A student proposal should name the
  interface it changes, the tests it needs, and the metrics that would prove it
  helped.
