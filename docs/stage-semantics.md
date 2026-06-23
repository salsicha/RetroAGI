# Stage Environment Semantics

This document defines the observation, action, reward, episode-ending, and reset
semantics for every RetroAGI curriculum stage. It distinguishes behavior that is
implemented today from contracts that a future adapter must satisfy.

## Shared Stage Contract

Environment-backed stages implement the `StageAdapter` protocol:

```python
observation = stage.reset(seed=seed)
observation, reward, terminated, truncated, info = stage.step(action)
```

- `observation` is the stage-native state consumed by its vision or hierarchy
  adapter.
- `action` advances the environment by one stage-native control step.
- `reward` is a Python `float` for that transition, not cumulative episode
  return.
- `terminated` means the episode reached an environment-defined terminal state.
- `truncated` means an external limit ended the episode without a terminal
  environment state.
- `info` contains non-observation diagnostics and structured state.
- Callers must reset when either `terminated` or `truncated` is true.
- Environments do not auto-reset. Calling `step` after an episode ends is
  outside the supported contract.
- `terminated` and `truncated` are independent. Both may be true on the same
  transition if a terminal event occurs exactly at a limit.

`StageAdapter.reset` returns only the observation and stores reset metadata for
`encode_observation`. A native environment may expose the Gymnasium
`(observation, info)` reset form underneath the adapter.

## Stage 1: Synthetic 1D

**Status:** implemented as a finite supervised data generator and trainer, not
as an episodic `StageAdapter`.

### Observation

Each generated sample contains three synchronized sequences:

- `X_A`: eight slow discrete tokens in `[0, 19]`.
- `X_B`: sixteen medium discrete tokens in `[0, 19]`.
- `X_C`: sixty-four fast scalar inputs sampled from a standard normal
  distribution.

The paired targets are `Y_A`, `Y_B`, and `Y_C`. A and B advance deterministic
modulo-20 concepts from randomly selected starts. Each B-level concept defines
the sine/cosine parameters used to produce the corresponding C-level continuous
targets.

### Action

There is no external environment action. The model's action is its length-64
continuous controller prediction for `Y_C`. This is an optimization output, not
an input to `generate_hierarchical_data`.

### Reward

There is no scalar environment reward. Training uses four losses:

- first-pass controller mean squared error,
- second-pass controller mean squared error,
- world-model mean squared error,
- critic feedback magnitude regularization.

Evaluation additionally reports A-token accuracy and B-parameter error.

### Termination And Truncation

Neither concept applies within a sample. Every sample has fixed sequence
lengths from `SYNTHETIC_1D_SPEC`; dataset iteration ends when the selected batch
or epoch is exhausted.

### Reset

There is no `reset` method. Generating a new dataset is the reset-equivalent
operation. Current generation uses NumPy's global random state, so callers must
seed NumPy before generation when reproducibility is required.

## Stage 2: Block SMB

**Status:** fully implemented by `MarioScenarioEnv` and exposed through
`BlockSMBStage`.

### Observation

The native observation is an `(240, 256, 3)` `uint8` RGB array in HWC layout.
It shows the current camera viewport. Coordinates in scenario and `info` state
are world coordinates; rendered x-coordinates are shifted by `camera_x`.

The native environment also returns `info` with:

- Mario position, velocity, grounded state, facing, skid state, coyote frames,
  and jump buffer;
- camera position and maximum rightward position reached;
- normalized nearest-coin and nearest-enemy features;
- normalized distance to a platform below Mario;
- a 14-element `float32` `state_vec`.

`BlockSMBStage.reset` returns only the RGB observation and retains `info` as
`last_info`. `BlockSMBStage.encode_observation` combines the RGB vision output
with `state_vec`.

### Action

Both SMB game stages use the action space declared by `SMB_GAME_SPEC`, currently
the shared `SMBAction` vocabulary. One action advances one physics and rendering
frame:

| Name | Stable ID | Block SMB | Full SMB NES buttons |
| --- | ---: | ---: | --- |
| `NOOP` | `0` | `0` | none |
| `RIGHT` | `1` | `1` | `RIGHT` |
| `RIGHT_JUMP` | `2` | `2` | `RIGHT` + `A` |
| `LEFT` | `3` | `3` | `LEFT` |
| `LEFT_JUMP` | `4` | `4` | `LEFT` + `A` |
| `JUMP` | `5` | `5` | `A` |

Jump actions are held controls, not one-shot events. A transition from
non-jump to jump registers a buffered press; releasing jump while rising cuts
upward velocity. Horizontal input uses acceleration, momentum, and skidding.
Raw integer IDs remain accepted by `BlockSMBStage` for compatibility, but new
code should use the enum names.

This transfer vocabulary intentionally leaves NES-only buttons such as `B`,
`START`, and `SELECT` released. Adding run/fire or menu actions requires an
explicit vocabulary version rather than changing the six stable IDs above.

### Reward

Transition reward is owned by `MarioScenarioEnv` and is the sum of named terms
from `BlockSMBRewardConfig`:

| Term | Default | Notes |
| --- | ---: | --- |
| `progress` | `0.05 * delta_x` | Added only when Mario exceeds the episode's previous maximum x-position |
| `coin` | `+10.0` | Added once for each newly collected coin |
| `enemy_stomp` | `+5.0` | Added when a stomp kills a live enemy |
| `goal` | `+50.0` | Added when Mario intersects the goal rectangle |
| `fall_death` | `-10.0` | Added when Mario falls below the viewport |
| `enemy_hit` | `-10.0` | Added when Mario touches a live enemy without stomping |
| `frame_penalty` | `-0.01` | Added once per `step` transition |

Progress is initialized to Mario's reset x-position, so spawning at `x > 0`
does not grant a progress bonus. Event rewards are additive; the `score` field
tracks coin and enemy points but is not the episode reward.

Every `step` info dict includes `reward_terms`, with each term above plus
`total`. The returned scalar reward must equal `info["reward_terms"]["total"]`.
Block SMB trainers must consume that environment reward directly. If a training
experiment needs different shaping, it should tune `BlockSMBRewardConfig` at
environment construction and log the resolved config, not add overlapping
progress, coin, enemy, goal, death, or time rewards inside the trainer.

Every `step` info dict includes reward diagnostics:

- `reward_terms`: additive components for `progress`, `coin`, `enemy_stomp`,
  `goal`, `fall_death`, `enemy_hit`, `frame_penalty`, and `total`.
- `reward_total`: the scalar transition reward after summing all terms.
- `reward_config`: the resolved reward config for the transition.

These diagnostics are for logging, tests, and ablations. They are not separate
trainer rewards unless a future task explicitly defines a multi-objective
training interface.

### Training And Evaluation

`train_and_evaluate_block_smb` is the supported P3 trainer entry point. It uses
the shared actor/world-model/critic module, stores rollout transitions in
`BlockSMBReplayBuffer`, passes each batch's episode mask into the shared world
model, and carries `WorldModelState` between rollout steps while the episode
continues. Terminal or truncated transitions drop that state so the next
episode starts from zero recurrent memory. The collector detaches carried state
between environment steps, keeping temporal context explicit without growing an
unbounded rollout graph.

The Block SMB trainer reports separate objective terms instead of treating the
discounted return as a C-level controller target:

| Term | Metric | Target |
| --- | --- | --- |
| Representation | `loss_representation` | predicted next-state representation matches the next observation representation |
| Dynamics | `loss_dynamics` | world model predicts the next C-stream state |
| Reward | `loss_reward` | reward head predicts the immediate environment reward |
| Value | `loss_value` | value head predicts the discounted return target |
| Policy | `loss_policy` | actor log probability is weighted by return advantage |
| Critic feedback | `loss_critic_feedback` | critic output magnitude stays bounded |
| Imagined rollout | `loss_imagined_rollout` | learned dynamics are recursively unrolled from replay states and matched to observed future C states and rewards |
| Entropy | `loss_entropy` | action entropy is logged and weighted as an exploration bonus |

Legacy aliases (`loss_actor_pass1`, `loss_actor_pass2`, `loss_world_model`, and
`loss_critic`) remain in metrics for older run-summary consumers, but new
analysis should use the separated terms above.

Reward and objective tuning is part of the saved `BlockSMBTrainingConfig`.
`reward_config` stores the exact `BlockSMBRewardConfig` used to build train and
evaluation environments, and the Block SMB CLI exposes each reward term plus the
separated objective weights. Evaluation adds `tuning_metrics` beside the fixed
scenario results:

- `threshold_pass_rate`: fraction of fixed scenarios passing the documented
  deterministic success thresholds;
- `mean_success_rate`: mean deterministic goal-completion rate;
- `mean_return`: mean deterministic scenario return;
- `score`: scalar ordering for tuning sweeps, with threshold coverage weighted
  ahead of success rate and raw return.

Use these fields to compare reward and hyperparameter changes. A higher return
without a higher threshold pass rate should not be treated as solving P3.

Imagined rollouts are controlled by
`BlockSMBTrainingConfig.imagined_rollout_horizon` and
`imagined_rollout_weight`, exposed by the CLI as
`--imagined-rollout-horizon` and `--imagined-rollout-weight`. A horizon of zero
preserves the one-step objective. Positive horizons start from real replay
states, recursively feed predicted C states through the actor/world-model loop,
and compare each imagined future against the matching observed transition inside
the same trajectory. The trainer logs `loss_imagined_dynamics`,
`loss_imagined_reward`, `loss_imagined_rollout`, and
`imagined_rollout_steps`.

Target-network stabilization is available through
`BlockSMBTrainingConfig.target_network_mode`, `target_network_tau`, and
`target_network_instability_threshold`, exposed by the CLI as
`--target-network-mode`, `--target-network-tau`, and
`--target-network-instability-threshold`. The default mode is `off`. Mode `on`
always uses an exponential-moving-average copy of the actor/world-model/critic
for representation targets. Mode `auto` enables that target only when the
measured one-step dynamics MSE from the current replay exceeds the configured
threshold. This keeps stabilization tied to measured instability instead of
silently changing the default objective. Checkpoints store the target-network
state when it exists. The trainer logs `target_network_active`,
`target_network_instability`, `target_network_drift`, and `target_network_tau`.

The low-level adaptive controller supports `controller_schedule="constant"` and
`controller_schedule="linear"`, exposed by the CLI as
`--controller-schedule`. The default `constant` schedule preserves existing
checkpoint behavior by repeating each B-level gain across its C slots. The
`linear` schedule treats B-level gains as control points and interpolates C-level
gains inside each B chunk. This is the implemented answer to the P4 gain-schedule
question: the project can now measure piecewise-constant control against a
smooth gain schedule with the resolved choice saved in run summaries and
checkpoints.

Block SMB ablation runs are part of `BlockSMBTrainingConfig.ablation` and are
recorded in checkpoint and run-summary configs. The CLI exposes each switch in
both enable and disable form so saved checkpoint settings can be inherited or
overridden:

| Pathway | Disable flag | Effect |
| --- | --- | --- |
| Vision | `--disable-vision` | Replaces A/B semantic streams with background tokens and zeros C position, semantic-probability, and patch-token slots while preserving symbolic environment state. |
| World model | `--disable-world-model` | Bypasses learned dynamics, uses the current C-stream state as the next-state prediction, and disables dynamics and imagined-rollout loss contributions while retaining diagnostics. |
| Critic feedback | `--disable-critic-feedback` | Still computes critic output for metrics, but does not inject it into the actor's second pass. |
| Hierarchy levels | `--disable-hierarchy` | Replaces A/B semantic streams with background tokens while preserving C-stream inputs. |
| Recurrent state | `--disable-recurrent-state` | Starts the world model from zero recurrent memory at each rollout step instead of carrying state between steps. |
| Checkpoint transfer | `--disable-checkpoint-transfer` | Uses a fresh frozen Block ViT for policy observations instead of loading `data/block_vit/block_vit.pth` or a supplied vision checkpoint. |

If deterministic policy training stalls, run the Block ViT perception
diagnostic before changing policy losses:

```bash
retroagi-block-smb diagnose-vision \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --samples 64 \
  --rollout-steps 32
```

The diagnostic compares semantic patch IDs and normalized Mario position
against exact palette-derived labels from procedural Block SMB frames. It
reports accuracy, foreground accuracy, mean IoU, per-class IoU, position RMSE,
and `bottleneck_reasons`. If `perception.bottleneck` is true, improve or
retrain the Block ViT checkpoint before interpreting low policy success as an
actor/world-model/critic failure.

The tracked `data/block_vit/block_vit.pth` checkpoint was retrained through
epoch 20 with `position_weight=16.0`. On the standard 64-frame diagnostic sample
it now reports `perception.bottleneck=false`, `mean_iou=0.9802`,
`foreground_accuracy=0.9955`, `position_rmse=0.0185`, and
`position_within_tolerance=0.9844`.

The trainer applies finite-loss and finite-gradient checks before each optimizer
step and clips gradients with `BlockSMBTrainingConfig.gradient_clip_norm`.
Curriculum order starts with the four fixed scenario files and can append seeded
procedural scenarios through `generated_scenarios`. Evaluation reports deterministic
return and success rate for each fixed scenario. It also attaches the documented
[Block SMB success thresholds](block-smb-success-thresholds.md), per-scenario
threshold diagnostics, and a top-level `success_thresholds_met` boolean. Optional
recording writes compressed `.npz` archives containing RGB frames, selected
actions, and rewards for each evaluation episode.

`SequentialBlockSMBVectorEnv` is the supported vector-environment scaffold for
P3. It steps multiple independent `MarioScenarioEnv` instances sequentially; true
parallel execution remains an optimization layer above this tested contract.

### Termination

`terminated` becomes true when any of these occurs:

- Mario's y-position exceeds the viewport height;
- Mario collides with a live enemy without satisfying the stomp condition;
- Mario intersects the goal rectangle.

The environment does not encode a separate terminal reason in `info`; callers
can infer it from state and scenario geometry only when needed.

### Truncation

`truncated` becomes true when `steps >= max_steps`. `max_steps` defaults to
1,000 frames and is reset neither from a scenario field nor by procedural
generation. Timeout carries no additional reward beyond the normal per-frame
penalty.

### Reset

The native call is:

```python
observation, info = env.reset(scenario=scenario, seed=seed)
```

Reset:

- sets steps, score, and camera position to zero;
- sets maximum progress to Mario's initial x-position;
- recreates Mario, platform, coin, enemy, and goal state from the scenario;
- clears velocity, collection, enemy-death, and jump state;
- renders and returns the initial RGB frame and structured `info`.

If no scenario is supplied, a small built-in scenario is used. The
`BlockSMBStage` constructor may pin a scenario for every reset. A reset seed
seeds the environment's internal RNG, although fixed scenarios currently have
deterministic reset state and procedural scenarios are generated separately by
passing a seed to `MarioScenarioEnv.generate_scenario`.

## Stage 3: Full SMB

**Status:** perception and a `FullSMBStage` lifecycle adapter exist. The adapter
wraps `stable-retro` lazily through the shared `GymnasiumBackendAdapter`, maps
the shared SMB action vocabulary to backend-specific button vectors, and
normalizes both Gym-style four-value and Gymnasium-style five-value `step`
results into the shared stage contract. Backend game-variable extraction is
normalized into `full_smb_signals` and `state_vec`. Frame skipping, resizing,
normalization, stacking, and episode masks are implemented. Emulator state
snapshots are implemented for repeatable evaluation through the backend
adapter's `env` or `env.em` save/load state API. `FullSMBSegmentationVision`
now defaults to the versioned 13-class patch-level Vision Transformer
checkpoint; the previous six-class DeepLab wrapper remains available as
`FullSMBDeepLabSegmentationVision` for legacy checkpoint inspection.

Backend-specific values must be normalized at this boundary rather than leaking
into shared training code.

### Observation

The stage-native observation is the RGB frame returned by `stable-retro` for
`SuperMarioBros-Nes`. The adapter exposes a contiguous `uint8` HWC RGB array
and places backend metadata in `info`. RGBA inputs are truncated to RGB, and
floating point RGB inputs in `[0,1]` are converted to `uint8`.

Policy observations are controlled by `FullSMBObservationConfig`:

| Field | Default | Meaning |
| --- | ---: | --- |
| `frame_skip` | `1` | Number of backend frames to advance for one shared action. Rewards are summed across executed frames. |
| `frame_stack` | `4` | Number of normalized frames retained in observation metadata. Reset padding is marked invalid in `frame_mask`. |
| `resize_shape` | `(224, 256)` | Height/width used for normalized frame tensors and vision input. `None` preserves the backend frame size. |

`encode_observation` sends the resized `[0,1]` HWC frame to the Full SMB vision
encoder. It also records `frame_stack` as `[1, frame_stack, 3, H, W]`,
`frame_mask`, `frame_skip`, `resize_shape`, and `normalized_range` in
`batch.metadata["observation"]`.

### Action

`FullSMBStage` uses the action space declared by `SMB_GAME_SPEC`, currently the
same `SMBAction` vocabulary used by Block SMB. `full_smb_action` translates
each action to the `stable-retro` NES button vector by reading `env.buttons`;
it does not assume fixed button indices. The adapter records the shared action
ID, name, backend button names, and button vector in `info["action"]`. When
`frame_skip > 1`, the same button vector is applied repeatedly until the
configured count is reached or the backend reports termination or truncation.
`info["action"]` records `frames_executed` and per-frame rewards.

### Info Signals

`FullSMBStage` extracts common stable-retro game variables into
`info["full_smb_signals"]`:

| Field | Meaning |
| --- | --- |
| `position` | Raw `(x, y)` player/world position when backend variables expose it. X may be derived from `xscrollHi`, `xscrollLo`, and `screen_x`. |
| `score` | Backend score counter, if present. |
| `coins` | Coin counter, if present. |
| `lives` | Lives counter, if present. |
| `completion` | True when a completion flag or terminal reason indicates level clear, goal, or flag completion. |
| `death` | True when a death flag or terminal reason indicates death or game over. |
| `terminated` | The adapter-level termination boolean for the transition. |
| `truncated` | The adapter-level truncation boolean for the transition. |

The adapter also writes `info["state_vec"]`, a nine-value `float32` vector with
normalized x, y, score, coins, lives, completion, death, terminated, and
truncated values. Missing raw values are encoded as zero.

### Reward

Until a project-level reward contract is implemented, the adapter passes
through the scalar reward emitted by the `stable-retro` game integration for
each executed backend frame and sums those rewards for the adapter transition.
It does not silently combine that reward with Block SMB reward terms. Any
shaping must be explicit, configured, and reported separately in `info`.

### Termination

The adapter forwards the backend's `terminated` value when the backend returns a
five-value Gymnasium step result. For legacy four-value Gym results, `done`
maps to `terminated` unless `info["truncated"]` or
`info["TimeLimit.truncated"]` is true. Terminal game conditions such as death,
game over, or level completion are also reflected in
`info["full_smb_signals"]` when the backend exposes a matching flag or terminal
reason.

### Truncation

The adapter forwards the backend's `truncated` value independently from
termination when available. Legacy four-value Gym backends can still express
timeouts through `info["truncated"]` or `info["TimeLimit.truncated"]`.
Project-imposed step or time limits must set `truncated`, never rewrite a
timeout as `terminated`. Frame skipping stops immediately when either flag is
reported.

### Reset

The shared backend adapter calls reset, returns the normalized initial
observation/info pair, and passes a supplied seed through when the backend
supports it. If backend reset does not accept `seed`, the adapter calls
`env.seed(seed)` when available before resetting. `FullSMBStage` then begins a
new emulator episode, clears adapter-owned episode mask state, and repopulates
the frame stack with invalid reset padding plus the first valid reset frame.

### Smoke Checks

`run_headless_random_agent_smoke(stage, FullSMBSmokeConfig(...))` runs seeded
random actions without calling `render()` unless rendering is explicitly
enabled. It records executed steps, resets, episode endings, reward totals,
action IDs, observation checksums, final signals, and optional
`encode_observation` coverage.

`run_deterministic_reset_smoke(make_stage, seed=..., steps=...)` creates two
fresh stages, resets both with the same seed, replays the same action sequence,
and compares observation checksums, rewards, ending flags, signals, and executed
action IDs. A mismatch names the first trace field that diverged.

The default command-line smoke runner is headless:

```bash
python -m retroagi.stages.full_smb.run --steps 500 --seed 0
```

Pass `--render` only for local visual inspection.

### Checkpoint Transfer

`transfer_block_smb_checkpoint_to_full_smb(...)` loads a schema-v1 Block SMB
trainer checkpoint, validates that its A/B/C hierarchy dimensions and
declared action metadata match `FULL_SMB_SPEC`, reuses the
actor/world-model/critic weights, and writes a Full SMB transfer checkpoint
with source provenance.

Block ViT perception weights are validated and recorded as source provenance,
but they are not loaded into Full SMB directly because the semantic vocabularies
differ. Before a transferred policy is used for Full SMB inference or continued
training, the Full SMB ViT must be bootstrapped on full-game assets composed
into synthetic scenarios and validated on held-out semantic and position
metrics. Full SMB policy execution then uses that versioned Full SMB ViT
checkpoint through `FullSMBSegmentationVision`.

```bash
python -m retroagi.stages.full_smb.transfer \
  --block-policy-checkpoint data/block_smb/policy.pth \
  --output-checkpoint data/full_smb/transferred_policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth
```

`load_transferred_full_smb_policy(...)` restores a saved transfer checkpoint,
and `select_transferred_full_smb_action(...)` runs deterministic or sampled
action selection from a `FullSMBStage.encode_observation(...)` batch.

The Full SMB stage is the full-fidelity validation and continuation rung. Its
first job is to verify that transferred Block SMB policy checkpoints perform
valid inference against emulator observations, shared actions, and Full SMB
signals. After that contract is validated, direct Full SMB training resumes from
the transferred checkpoint rather than treating transfer as the final result.

`compare_transferred_checkpoint_with_scratch(...)` evaluates a transferred Full
SMB checkpoint against either a supplied scratch-trained Full SMB checkpoint or,
when no scratch checkpoint exists yet, a same-architecture scratch-initialized
baseline. The comparison trajectory is collected with seeded random actions so
both policies see identical `FullSMBStage` batches. The report includes action
agreement, action histograms, mean entropy, mean top-logit margin, reset counts,
episode endings, and collection reward.

```bash
python -m retroagi.stages.full_smb.compare \
  --transfer-checkpoint data/full_smb/transferred_policy.pth \
  --scratch-checkpoint data/full_smb/scratch_policy.pth \
  --output artifacts/full_smb/transfer_vs_scratch.json
```

Omit `--scratch-checkpoint` to compare against a deterministic scratch
initialization controlled by `--scratch-seed`.

### Emulator State

`FullSMBStage.save_emulator_state()` captures a `FullSMBEmulatorState` snapshot
containing:

- the backend emulator state from `env.get_state()` / `env.set_state()` or
  stable-retro-style `env.em.get_state()` / `env.em.set_state()`;
- the last RGB observation;
- `last_info`, episode mask, termination, and truncation flags;
- adapter-owned frame stack tensors and frame-mask values.

`FullSMBStage.load_emulator_state(snapshot)` restores the backend and adapter
state and returns the restored RGB observation. Replaying the same action after
load must produce the same next backend transition when the underlying emulator
state API is deterministic.

Deterministic local save-state recipes live in
`retroagi.stages.full_smb.save_states` and are documented in
[full-smb-save-states.md](full-smb-save-states.md). They define reset seeds,
stable-retro source states, scripted actions, local output paths, and linked
task names. The generated `.state` files are local-only ROM-derived artifacts
under `local/full_smb/states/` and must not be committed.
