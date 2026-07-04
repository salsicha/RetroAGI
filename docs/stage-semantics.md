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
- normalized short-horizon step progress, goal direction, support edge,
  next-platform delta, and ground-ahead probes;
- a 27-element `float32` `state_vec`, where the final three values encode
  `death`, `terminated`, and `truncated` for the transition that produced the
  observation.

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

Policy-driven Block SMB and Full SMB training, evaluation, and playback pass
model-selected jump actions through `SMBParameterizedPrimitiveExecutor`. The
executor reads B-level primitive duration logits, holds `A` for the selected
duration bin, releases to `RIGHT`, `LEFT`, or `NOOP`, and uses the current ViT
output from `StageBatch.metadata["vision"]` to terminate on ground/platform
landing or enemy contact. It also suppresses repeated jump starts until the
policy emits a non-jump action. `SMBJumpActionTerminator` remains available as a
lower-level landing-release helper. The same policy paths pass pure walk actions
through `SMBWalkActionLimiter`, which caps continuous `RIGHT` or `LEFT` holds at
one second before inserting a release frame.

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

### Parameterized Scenario Distribution

Fixed Block SMB scenarios are regression probes and known-good sentinels. They
should not be the only policy-training distribution. The simplified environment
has exact semantic masks, exact symbolic state, deterministic physics, and fast
reset, so it should provide the high-volume Monte Carlo training data before a
policy is promoted to Full SMB.

The target contract is a versioned parameterized distribution of scenario
families: flat runs, gaps, stairs, platform chains, moving bridges, enemy hops,
enemy patrols, enemy-gap combinations, enemy stomps, retreat/recovery states,
wait-timing tasks, and mixed sections. Each sampled scenario should record its
distribution ID, family, split, seed, sampled parameters, constraints, and
oracle/reachability metadata.

Training should sample from train splits, tune on held-out validation splits,
and report final test-split results separately from fixed-scenario thresholds.
Promotion to Full SMB should require both fixed-scenario success and
distribution-level coverage metrics. The detailed implementation plan lives in
[block-smb-monte-carlo-curriculum.md](block-smb-monte-carlo-curriculum.md).

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

The diagnostic compares semantic patch IDs, air/ground/platform support state,
and normalized Mario position against exact palette-derived labels from
procedural Block SMB frames. It reports accuracy, foreground accuracy, mean IoU,
per-class IoU, support accuracy, position RMSE, and `bottleneck_reasons`. If
`perception.bottleneck` is true, improve or retrain the Block ViT checkpoint
before interpreting low policy success as an actor/world-model/critic failure.

The tracked `data/block_vit/block_vit.pth` checkpoint was retrained through
epoch 20 with `position_weight=16.0`. On the standard 64-frame diagnostic sample
it now reports `perception.bottleneck=false`, `mean_iou=0.9802`,
`foreground_accuracy=0.9955`, `position_rmse=0.0185`, and
`position_within_tolerance=0.9844`.

The trainer applies finite-loss and finite-gradient checks before each optimizer
step and clips gradients with `BlockSMBTrainingConfig.gradient_clip_norm`.
Curriculum order starts with the fixed scenario files and can append seeded
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
| `crop_margins` | `(0, 0, 0, 0)` | Top/right/bottom/left pixels cropped before resizing. |
| `hud_policy` | `preserve` | `preserve` keeps the SMB HUD; `crop` adds `hud_crop_top` to the top crop margin. |
| `hud_crop_top` | `24` | Extra top pixels removed only when `hud_policy="crop"`. |
| `color_mode` | `rgb` | `rgb` preserves color; `grayscale` converts luminance and repeats it across three channels for ViT compatibility. |
| `normalization_mean` | `(0.0, 0.0, 0.0)` | Per-channel mean subtracted after conversion to `[0,1]`, crop, optional grayscale, and resize. |
| `normalization_std` | `(1.0, 1.0, 1.0)` | Per-channel positive scale applied after mean subtraction. |
| `include_camera_state` | `False` | When true, appends the four-value `camera_vec` to the C-stream state after the stable nine-value `state_vec`. |

`encode_observation` sends the preprocessed HWC frame to the Full SMB vision
encoder. It records `frame_stack` as `[1, frame_stack, 3, H, W]`,
`frame_mask`, `frame_skip`, `resize_shape`, `effective_crop_margins`,
`hud_policy`, `color_mode`, normalization settings, `camera_vec`, and
`camera_state_enabled` in `batch.metadata["observation"]`.

`retroagi diagnose-vision --game smb --stage full` runs the Full SMB ViT on
real emulator frames and reports unlabeled perception diagnostics:
`semantic_confidence`, `class_coverage`, `covered_classes`,
`temporal_stability`, position consistency against `camera_vec`/`state_vec`,
and `bottleneck_reasons`. These metrics do not replace synthetic asset-mock
IoU; they catch real-frame confidence, coverage, temporal, and localization
failures before policy-training issues are blamed on the controller.

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
`info["full_smb_signals"]`. Values can come from top-level backend `info`
fields or nested `memory`, `ram`, `variables`, `game_variables`, or
`backend_info` mappings:

| Field | Meaning |
| --- | --- |
| `position` | Raw `(x, y)` player/world position when backend variables expose it. X may be derived from `xscrollHi`, `xscrollLo`, and `screen_x`. |
| `screen` | Backend screen coordinate or screen tuple when exposed through `screen`, `screen_x`, or `screen_y`. |
| `level`, `world`, `stage` | Level identity from explicit level fields or a world/stage pair such as `2-1`. |
| `score` | Backend score counter, if present. |
| `coins` | Coin counter, if present. |
| `lives` | Lives counter, if present. |
| `power_state` | Mario/player power state from status or power fields, normalized for common small/big/fire values. |
| `completion` | True when a completion flag or terminal reason indicates level clear, goal, or flag completion. |
| `death` | True when a death flag or terminal reason indicates death or game over. |
| `game_over` | True when a backend game-over flag or terminal reason indicates game over. |
| `timeout` | True when the transition is truncated or a backend timeout/time-up field is set. |
| `terminated` | The adapter-level termination boolean for the transition. |
| `truncated` | The adapter-level truncation boolean for the transition. |

The adapter also writes `info["state_vec"]`, a nine-value `float32` vector with
normalized x, y, score, coins, lives, completion, death, terminated, and
truncated values. Missing raw values are encoded as zero. This compact state
vector remains stable for checkpoint compatibility. `info["camera_vec"]`
separately encodes normalized raw scroll x, screen x, screen y, and player
x-offset within the camera viewport. New experiments can opt into appending
that vector to the C-stream with `include_camera_state=True`; richer extracted
fields remain available through `full_smb_signals`.

### Reward

Full SMB reward tuning is owned by `FullSMBRewardConfig`, not
`BlockSMBRewardConfig`. The default config preserves the current backend reward
behavior by using the `stable-retro` emulator progress reward with weight
`1.0` and setting all additional shaping terms to `0.0`.

| Term | Default | Direction | Source |
| --- | ---: | --- | --- |
| `emulator_progress` | `1.0` | positive | `stable-retro` backend reward |
| `completion` | `0.0` | positive | `full_smb_signals.completion` |
| `survival` | `0.0` | positive | alive/episode state |
| `score` | `0.0` | positive | `full_smb_signals.score` deltas |
| `coin` | `0.0` | positive | `full_smb_signals.coins` deltas |
| `enemy` | `0.0` | positive | enemy-defeat objective signals |
| `damage` | `0.0` | negative | damage or unsafe-collision signals |
| `death` | `0.0` | negative | `full_smb_signals.death` |
| `frame_penalty` | `0.0` | negative | executed backend frame count |

`FullSMBStage` returns the sum of the Full SMB reward terms as the scalar
transition reward. It reports `info["reward_terms"]` with each term above plus
`total`, `info["reward_total"]` with the returned scalar reward, and the
resolved `reward_config` manifest. With the default config, `emulator_progress`
is the summed backend frame reward and every other shaping term is zero, so the
adapter remains behavior-compatible with stable-retro smoke tests. Custom Full
SMB reward configs must shape rewards through these adapter terms rather than
reusing Block SMB progress, coin, enemy, goal, death, or time terms.

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
That continuation step must declare its perception mode: `freeze` keeps the
asset-mock Full SMB ViT fixed, `fine_tune` loads that checkpoint and includes
trainable ViT parameters in the optimizer, and `replace` starts from a fresh
trainable Full SMB ViT. The selected mode, checkpoint path, trainable/frozen
status, optimizer participation, and saved perception state are written into the
Full SMB policy checkpoint config and metadata.

`FullSMBTrainingConfig` now owns the Full SMB trainer contract: seed, selected
device, deterministic Torch mode, rollout length, updates per epoch, requested
vector environment count, learning rate, loss weights, adapter-owned
`FullSMBRewardConfig`, checkpoint/resume/init paths, recording paths, structured
log path, and optional tracking backend settings. The current online trainer is
single-env, so checkpoints record both the requested `vector_env_count` and
`active_vector_env_count=1`; the later vector rollout task is responsible for
turning that request into parallel emulator execution.

Direct scratch training is selected with `--mode scratch` or by omitting both
`resume_path` and `init_checkpoint` in auto mode. In that mode the trainer
constructs the policy through the shared architecture factory, consumes
`FullSMBStage.encode_observation(...)` A/B/C `StageBatch` tensors directly,
validates their Full SMB sequence lengths and dtypes before policy inference,
and records `training_source.mode="scratch"` plus the stage-batch contract in
checkpoint config and metadata. Transferred-policy fine-tuning is selected with
`--mode fine-tune --init-checkpoint ...`; the init checkpoint may point at either
a raw Block SMB policy checkpoint, which the trainer converts into the Full SMB
contract in memory, or an existing Full SMB transfer checkpoint. Both paths start
Full SMB optimizer updates at epoch zero and record
`training_source.mode="init_checkpoint"`, the source checkpoint identity, and
whether the source was a Block SMB policy or a Full SMB transfer checkpoint.
Resume runs record their source checkpoint provenance in the same fields.

Full SMB training now carries the shared world-model recurrent state between
rollout steps while the same episode is continuing. The trainer drops that state
at explicit Full SMB boundaries: manual reset, termination, truncation, death,
timeout, level completion, or game over. Boundary counts and recurrent reset
counts are logged as rollout metrics, and checkpoint rollout metadata records
`recurrent_state_policy="carry_until_full_smb_boundary"` plus the reset reasons.
Training results and policy checkpoints also include compact rollout replay
metadata under `rollouts` and `config.rollout_storage` /
`metadata.training.rollout_storage`. Each recorded step stores the selected SMB
action, scalar reward, done/terminated/truncated flags, episode mask, boundary
reasons, scenario/task/emulator-state identifiers when exposed by the backend,
and selected `full_smb_signals` plus reward terms. This storage is intended for
resume diagnostics, promotion evidence, and deterministic replay inputs.
Evaluation recording is a separate artifact stream: when `recording_dir` or
`recording_path` is configured, deterministic Full SMB evaluations write
compressed per-episode `.npz` files containing initial-plus-post-step RGB frame
arrays, actions, action names, rewards, termination flags, serialized
`full_smb_signals`, task/scenario/emulator-state IDs, and episode metadata.
Evaluation manifests are stored in the result/checkpoint metadata, and video
export is attempted only for video-suffixed `recording_path` values when OpenCV
is available.

Full SMB training stops before an optimizer step when numerical checks fail.
The trainer validates finite action logits, log-probs, entropy, policy losses,
total losses, gradients, scaled rewards, and reward/value predictions. It clips
gradients with `FullSMBTrainingConfig.gradient_clip_norm`, records entropy,
gradient norm, clip-event, scaled-reward, and prediction-bound metrics, and
writes the active thresholds into `config.safety` and
`metadata.training.safety`. The configurable guards are
`max_abs_loss`, `max_abs_scaled_reward`, and `max_abs_prediction`; CLI aliases
are `--max-abs-loss`, `--max-abs-scaled-reward`, and
`--max-abs-prediction`.

`compare_transferred_checkpoint_with_scratch(...)` evaluates a transferred Full
SMB checkpoint against either a supplied scratch-trained Full SMB checkpoint or,
when no scratch checkpoint exists yet, a same-architecture scratch-initialized
baseline. The comparison trajectory is collected with seeded random actions so
both policies see identical `FullSMBStage` batches. The report includes action
agreement, action histograms, mean entropy, mean top-logit margin, reset counts,
episode endings, and collection reward.

`compare_full_smb_policy_suite(...)` extends that protocol to named policy
roles. It can compare the transferred policy, a scratch-trained checkpoint or
scratch initialization, a fine-tuned checkpoint, a known-good checkpoint, and
additional `NAME=CHECKPOINT` policies across every selected task/seed stream.
Each stream is driven by the same seeded random actions, so pairwise action
agreement is computed from identical Full SMB observations.

```bash
python -m retroagi.stages.full_smb.compare \
  --transfer-checkpoint data/full_smb/transferred_policy.pth \
  --scratch-trained-checkpoint data/full_smb/scratch_policy.pth \
  --fine-tuned-checkpoint data/full_smb/fine_tuned_policy.pth \
  --known-good-checkpoint data/full_smb/known_good_policy.pth \
  --task-set fixed_benchmark \
  --seed 0 \
  --seed 1 \
  --output artifacts/full_smb/policy_suite_comparison.json
```

Omit `--scratch-checkpoint` or `--scratch-trained-checkpoint` to compare against
a deterministic scratch initialization controlled by `--scratch-seed`.

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

Fixed benchmark success thresholds live in
`retroagi.stages.full_smb.success` and are documented in
[full-smb-success-thresholds.md](full-smb-success-thresholds.md). Full SMB
evaluators should report progress, completion, survival, score/coins, deaths,
return, episode count, step budget, and `threshold_met` diagnostics against
that module rather than defining separate trainer-owned success logic.
