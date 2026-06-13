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

There is no scalar environment reward. Training uses three losses:

- first-pass controller mean squared error,
- second-pass controller mean squared error,
- world-model mean squared error.

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

Both game stages use the shared `SMBAction` vocabulary. One action advances one
physics and rendering frame:

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

Transition reward is the sum of all events on the frame:

| Event | Reward |
| --- | ---: |
| New maximum x-position | `0.05 * delta_x` |
| Collect a coin | `+10.0` |
| Stomp an enemy | `+5.0` |
| Touch the goal | `+50.0` |
| Fall below the viewport | `-10.0` |
| Touch a live enemy without stomping | `-10.0` |
| Every frame | `-0.01` |

Progress is rewarded only when Mario exceeds the episode's previous maximum
x-position. Event rewards are additive; the `score` field tracks coin and enemy
points but is not the episode reward.

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

- sets steps, score, camera position, and maximum progress to zero;
- recreates Mario, platform, coin, enemy, and goal state from the scenario;
- clears velocity, collection, enemy-death, and jump state;
- renders and returns the initial RGB frame and structured `info`.

If no scenario is supplied, a small built-in scenario is used. The
`BlockSMBStage` constructor may pin a scenario for every reset. A reset seed
seeds the environment's internal RNG, although fixed scenarios currently have
deterministic reset state and procedural scenarios are generated separately by
passing a seed to `MarioScenarioEnv.generate_scenario`.

## Stage 3: Full SMB

**Status:** perception and a direct `stable-retro` random-action runner exist;
`FullSMBStage` and its shared lifecycle adapter are not yet implemented.

The rules below are the required contract for that adapter. Backend-specific
values must be normalized at this boundary rather than leaking into shared
training code.

### Observation

The stage-native observation is the RGB frame returned by `stable-retro` for
`SuperMarioBros-Nes`. The adapter must expose a `uint8` HWC RGB array and place
backend metadata, including game variables, in `info`. Frame resizing,
normalization, skipping, and stacking are not yet part of the implemented
contract.

### Action

The runner uses the shared `SMBAction` vocabulary defined in the Block SMB
section. `full_smb_action` translates each action to the `stable-retro` NES
button vector by reading `env.buttons`; it does not assume fixed button indices.
The future `FullSMBStage` must use this mapping at its adapter boundary.

### Reward

Until a project-level reward contract is implemented, the adapter must pass
through the scalar reward emitted by the `stable-retro` game integration for
each transition. It must not silently combine that reward with Block SMB reward
terms. Any shaping must be explicit, configured, and reported separately in
`info`.

### Termination

The adapter must forward the backend's `terminated` value. Terminal game
conditions such as death, game over, or level completion must be represented by
the game integration and exposed with a reason in `info` when available.

### Truncation

The adapter must forward the backend's `truncated` value independently from
termination. Project-imposed step or time limits must set `truncated`, never
rewrite a timeout as `terminated`.

The current random runner does not yet implement this correctly: it uses
ambiguous `done`/`term` names and resets on only one boolean. It is not the
shared stage contract.

### Reset

The future adapter must call the backend reset, return only its initial RGB
observation, retain reset `info`, and pass a supplied seed through when the
backend supports it. Reset must begin a new emulator episode and clear
adapter-owned frame stacks, counters, recurrent episode state, and reward
shaping state. Deterministic emulator save-state reset is not implemented yet.
