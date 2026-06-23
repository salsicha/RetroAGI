# Tensor Contracts

This document defines shapes, dtypes, value ranges, and timescales for
`StageSpec`, `StageBatch`, and `VisionOutput`. Missing or temporary contracts
are labeled explicitly.

## Notation

- `B`: batch size.
- `L_A`, `L_B`, `L_C`: A-, B-, and C-stream sequence lengths.
- `R_AB`: B steps per A step.
- `R_BC`: C steps per B step.
- `K`: semantic class count.
- `D`: vision token width.
- `H`, `W`: image height and width.
- `G_H`, `G_W`: semantic grid height and width.
- `N = G_H * G_W`: patch-token count.

Shapes are batch-first. Unless stated otherwise, floating tensors use
`torch.float32` and index tensors use `torch.int64` (`torch.long`). Model
outputs are on the model device; an adapter keeps its batch tensors on the
vision model device.

## GameSpec

`GameSpec` is immutable game-profile metadata. It sits above `StageSpec` and
declares the game family, per-game action space, observation sources, semantic
classes, signal and reward schemas, progressive stage ladder, emulator backend,
asset requirements, and licensing/provenance rules. The current `smb` profile
uses this ladder:

```text
synthetic -> block -> full_asset_mock -> full
```

Synthetic validates the architecture, Block SMB trains the simplified game
models, `full_asset_mock` bootstraps the Full SMB ViT from full-game assets in
synthetic scenarios, and Full SMB validates inference before continuing
training in the emulator.

The action space is a tuple of `ActionSpec` entries owned by the game profile.
Each entry declares a stable integer ID plus optional backend metadata:
multi-button combos, release-all/no-op behavior, per-button release behavior,
continuous controls, and a stage-native discrete backend ID when it differs
from the stable policy ID. Legacy SMB helpers still accept `SMBAction`, but the
source of truth for SMB backend mapping is `SMB_GAME_SPEC.action_space`.
`GameSpec.action_backend_id(...)` maps an abstract policy action to a
stage-native discrete action, and `GameSpec.action_button_vector(...)` maps the
same policy action to a backend button vector using button names rather than
backend-specific button positions.

`GameSignalExtractor` is the game-neutral backend-info extraction protocol. It
returns `GameSignals` with common fields for position/progress, score, health,
lives, inventory, collectibles, completion, death, timeout, termination,
truncation, game-specific objectives, and termination reason. Stage adapters may
extend that dataclass with legacy convenience fields, as Full SMB does with
`coins`, but shared trainers should read the common `GameSignals` fields.

`RewardConfigSchema` is the game-owned reward configuration contract. It
declares each tunable reward term, default value, expected sign direction,
source signal, and description. Shared trainers should ask the selected
`GameSpec` to validate reward overrides instead of carrying game-specific
reward names such as `coin`, `enemy_hit`, or `fall_death` in trainer code.

`GameTaskSchema` is the game-owned task and curriculum contract. It declares
fixed benchmark tasks, procedural task templates, deterministic reset seeds,
curriculum order, task sources, and `TaskSuccessThreshold` gates. Block SMB's
fixed-scenario success thresholds now derive from `SMB_GAME_SPEC.task_schema`
so future games can expose the same shape without trainer rewrites.

`SyntheticDataSpec` is the game-owned low-fidelity data contract for concept
checks that run before pixels, physics, or emulator frames. It declares the
stage, observation and target kinds, generator entrypoint, deterministic split
sizes/seeds, shape contract, and metadata such as `pixel_free` and
`emulator_free`. SMB declares its existing Synthetic 1D generator as
`synthetic_1d_concept` under `SMB_GAME_SPEC.synthetic_data`.

`BlockGameSpec` is the game-owned mid-fidelity simulator contract. It declares
the block-stage adapter, environment implementation, simplified physics model,
native observation kind, symbolic state fields, exact semantic/position label
sources, fixed scenario files, procedural scenario generator, reset modes, and
capability metadata such as `fast_reset` and `exact_semantic_labels`. SMB
declares its current pygame-ce simulator as `SMB_GAME_SPEC.block_game`, which
gives future games the same place to describe their simplified training rung
without adding game assumptions to shared trainers.

## StageSpec

`StageSpec` is immutable metadata, not a tensor.

| Field | Type | Contract |
| --- | --- | --- |
| `name` | `str` | Stable stage identifier. |
| `observation_kind` | `str` | Native observation description. |
| `action_kind` | `str` | Stage action contract. |
| `seq_len_a` | positive `int` | `L_A`, the slow stream length. |
| `ratio_ab` | positive `int` | `R_AB`, B steps represented per A step. |
| `ratio_bc` | positive `int` | `R_BC`, C steps represented per B step. |
| `vocab_size` | positive `int` | Exclusive upper bound for A/B token IDs. |
| `action_space_name` | `str` | Optional game-neutral action-space identifier. |
| `action_count` | positive `int` or `None` | Optional discrete policy-action count. |
| `action_names` | tuple of `str` | Optional stable action names matching `action_count`. |

Derived lengths are:

```text
L_B = L_A * R_AB
L_C = L_B * R_BC
```

Synthetic 1D, Block SMB, and Full SMB currently use `L_A=8`, `R_AB=2`,
`R_BC=4`, and `vocab_size=20`, producing `L_B=16` and `L_C=64`. These are
representation lengths, not wall-clock durations.

`StageSpec` does not know about any concrete game's enum type. A discrete game
stage declares `action_count` and optional `action_names`; compatibility checks
only verify that the vocabulary can represent the declared action count. The
game profile owns button combos, no-op/release behavior, stable IDs, and
backend-native mappings.

Timescale ordering is slow A, medium B, fast C. One A slot spans two B slots
and eight C slots; one B slot spans four C slots. C-stream recurrent
processing currently uses groups of `R_BC=4` values.

## StageBatch

`StageBatch` is the hierarchy input and optional supervision contract.

| Field | Shape | Dtype | Range and meaning |
| --- | --- | --- | --- |
| `src_a` | `[B, L_A]` | `long` | IDs in `[0, vocab_size-1]`; slow context. |
| `target_a` | `[B, L_A]` or `None` | `long` | A supervision in the same ID range. |
| `src_b` | `[B, L_B]` | `long` | IDs in `[0, vocab_size-1]`; medium context. |
| `target_b` | `[B, L_B]` or `None` | `long` | B supervision in the same ID range. |
| `src_c` | `[B, L_C]` | floating | Fast continuous input; stage-specific range. |
| `target_c` | `[B, L_C]` or `None` | floating | Fast target; stage-specific range. |
| `metadata` | not a tensor | mapping or `None` | Diagnostics, not a shared model input. |

A and B must be integer IDs because both pass through `nn.Embedding`. C is
currently scalar-valued: the controller and world model consume `[B, L_C]`,
not `[B, L_C, features]`.

### Vision-To-Hierarchy Fusion

Environment stages use `VisionHierarchyProjector`:

- A is the dominant semantic class after class probabilities are averaged over
  the full image height and `L_A` ordered horizontal regions.
- B uses the same operation with `L_B` horizontal regions.
- C concatenates normalized position, global mean semantic probabilities,
  optional stage state, and pooled patch-token content in that order.
- Patch-token content is bounded with `tanh`, flattened in token/channel order,
  and adaptive-average-pooled into all remaining C slots.

This mapping uses semantic logits rather than sampled `semantic_ids`, preserves
left-to-right scene order, and never interpolates across heterogeneous feature
groups. Exact C offsets are recorded in `metadata["vision_fusion"]`, so stages
with different position, class, or state widths retain an inspectable layout.

### Synthetic 1D

The generator returns six tensors corresponding to a `StageBatch`, but does
not yet instantiate the dataclass.

| Tensor | Shape | Dtype | Range |
| --- | --- | --- | --- |
| `X_A`, `Y_A` | `[B, 8]` | `long` | IDs `[0,19]`. |
| `X_B`, `Y_B` | `[B, 16]` | `long` | IDs `[0,19]`. |
| `X_C` | `[B, 64]` | `float32` | Standard-normal samples; unbounded. |
| `Y_C` | `[B, 64]` | `float32` | `sin(concept) * X_C + cos(concept)`; unbounded. |

`Y_A` is the next A token. `Y_B` is the combined A/B concept used to derive
the four C targets beneath each B step. The trainer passes these tensors
directly to the model rather than through a stage adapter.

### Actor/Critic Refinement

`AgentWorldModelCritic` runs the shared actor twice for every batch:

1. The first actor pass receives `src_a`, `src_b`, and `src_c` with
   `criticism=None`.
2. The first pass C actions and B-level controller parameters are expanded to
   C resolution by the selected low-level controller schedule and passed to the
   world model.
3. The critic maps the predicted C state to a feedback tensor with shape
   `[B, L_A, d_model]`.
4. The second actor pass receives the original `src_a`, `src_b`, and `src_c`
   plus that exact critic tensor.

The critic tensor is added as an unscaled residual to the A stream after token
embedding and positional encoding, and before the A-level transformer:

```text
encoded_A = positional_encoding(embedding(src_a) * sqrt(d_model))
refined_A = encoded_A + criticism
```

No gating, normalization, clipping, detach, or loss weighting is applied inside
the actor. Loss weights and critic regularization remain trainer-owned. The
critic output must match the encoded A shape exactly and live on the same
device.

### Low-Level Controller Schedules

`AdaptiveController` treats the B-level `w_b` and `b_b` outputs as control
points for C-level gains. The supported schedules are:

| Schedule | C-level gain construction | Use |
| --- | --- | --- |
| `constant` | Repeat each B gain across its `R_BC` C slots. | Backward-compatible default for existing checkpoints and Synthetic 1D metrics. |
| `linear` | Linearly interpolate from each B gain to the next B gain across the current C chunk; the last B slot holds its own value. | Block SMB gain-schedule comparison when piecewise-constant control may be too abrupt. |

The actor still returns `w_b` and `b_b` with shape `[B,L_B]` in both modes.
`AgentWorldModelCritic` expands them through the actor's controller and passes
that same C-level context to the world model, so dynamics prediction observes
the low-level gains that produced the C actions.

The world model accepts an optional `WorldModelState` carrying LSTM hidden and
cell tensors between calls. Without a state it starts from zeros. Callers can
pass `episode_mask` to reset recurrent memory inside the model:

- shape `[B]` or `[B,1]`: reset or keep state before the first C chunk;
- shape `[B,ceil(L_C / R_BC)]`: reset or keep state before each recurrent
  C chunk.

Mask values use `1.0` to keep memory and `0.0` to reset it. When
`return_world_model_state=True`, `AgentWorldModelCritic` appends the next
`WorldModelState` to its usual output tuple. Default calls keep the original
seven-output contract.

### Block SMB

`BlockSMBStage.encode_observation` returns:

| Field | Shape | Dtype | Range and construction |
| --- | --- | --- | --- |
| `src_a` | `[B,8]` | `long` | Dominant semantic class in eight ordered horizontal scene regions. |
| `src_b` | `[B,16]` | `long` | Dominant semantic class in sixteen ordered horizontal scene regions. |
| `src_c` | `[B,64]` | `float32` | Fixed-layout position, semantics, state, and patch-token content. |
| all targets | `None` | n/a | Policy targets are not implemented. |

The current adapter accepts one environment observation and one `info` mapping
at a time, so its implemented batch size is `B=1`. The vision encoders support
larger batches, but batched environment metadata is not yet defined.

`VisionHierarchyProjector` applies this contract:

1. Apply semantic softmax over classes.
2. Average probabilities over the full image height and eight horizontal
   regions; the dominant class IDs form A.
3. Repeat at sixteen horizontal regions; the dominant class IDs form B.
4. Build C with fixed, non-overlapping slots:
   - `[0:2]`: normalized `(x,y)` position;
   - `[2:9]`: seven global mean semantic probabilities;
   - `[9:23]`: the 14-element environment `state_vec`;
   - `[23:64]`: 41 adaptive-average bins over flattened
     `tanh(patch_tokens)`.

This preserves horizontal scene order in A/B and keeps heterogeneous C values
in declared slots. No semantic-grid point sampling or interpolation across
position, class probabilities, state, and token features remains.

Position and probabilities are in `[0,1]`; bounded token content is in
`[-1,1]`. State features are converted to finite `float32` values and clipped
to the adapter-configured range, which defaults to `[-1,1]`.

Block SMB observation preprocessing is recorded in `metadata["observation"]`:

- `frame_stack`: normalized RGB frames with shape `[1,T,3,240,256]`, where
  `T` defaults to `4` and values are clipped to `[0,1]`.
- `frame_mask`: boolean shape `[1,T]`; reset padding is `False`, real frames
  are `True`.
- `frame_stack_size`, `normalized_range`, and `state_range`: the resolved
  preprocessing contract used by the adapter.

Episode boundaries are recorded in `metadata["episode"]`. `mask` has shape
`[1]` and is `1.0` for continuing transitions, `0.0` after termination or
truncation; `terminated` and `truncated` retain the environment booleans.

### Full SMB

`FullSMBStage` uses `FULL_SMB_SPEC` with `seq_len_a=8`, `ratio_ab=2`,
`ratio_bc=4`, `seq_len_b=16`, `seq_len_c=64`, and `vocab_size=20`. Its
`StageBatch` is produced by `VisionHierarchyProjector`. With the current
Full SMB ViT output, the C prefix contains two normalized position values
followed by thirteen semantic probabilities, then the nine-value Full SMB
signal vector: normalized x, y, score, coins, lives, completion, death,
terminated, and truncated. The remaining C slots contain pooled patch-token
features.

Full SMB observation metadata includes normalized resized frame tensors:
`frame_stack` has shape `[B, S, 3, H, W]`, where `S` is
`FullSMBObservationConfig.frame_stack` and `(H, W)` is `resize_shape` when set.
`frame_mask` has shape `[B, S]` and marks reset padding as invalid. The episode
metadata mask has shape `[1]` and is `0.0` after termination or truncation.

## VisionOutput

| Field | Shape | Dtype | Range and meaning |
| --- | --- | --- | --- |
| `position` | `[B, P]` | floating | Normalized position; `P = position_dim`. |
| `semantic_logits` | `[B,K,G_H,G_W]` or `[B,K,H,W]` | floating | Unnormalized class scores; unbounded. |
| `semantic_ids` | `[B,G_H,G_W]` or `[B,H,W]` | `long` | Argmax IDs `[0,K-1]`. |
| `tokens` | `[B,N,D]` | floating | Latent or pooled semantic tokens; generally unbounded. |
| `metadata` | not a tensor | mapping or `None` | Grid, image, checkpoint, or sequence diagnostics. |

`semantic_logits` are not probabilities. Consumers apply `softmax` over
dimension 1. Image encoders accept HWC/BHWC or CHW/BCHW input, convert to
`float32` BCHW, discard alpha, and divide by 255 only when values exceed 1.

### Synthetic Linear Vision

For input `[B,L]` with IDs in `[0,K-1]`:

| Field | Shape | Range |
| --- | --- | --- |
| `position` | `[B,1]` | `[0,1]`; normalized index of the maximum token value, not a spatial coordinate. |
| `semantic_logits` | `[B,K,1,L]` | Exact one-hot values `{0,1}`. |
| `semantic_ids` | `[B,1,L]` | Original IDs `[0,K-1]`. |
| `tokens` | `[B,L,D]` | Learned embeddings; unbounded. |

Defaults are `K=20`, `D=64`, and `P=1`.

### Block SMB ViT

Input is resized to `240x256` and divided into 16x16 patches, giving
`G_H=15`, `G_W=16`, and `N=240`.

| Field | Shape | Range |
| --- | --- | --- |
| `position` | `[B,2]` | `[0,1]`, `(x,y)`, probability-weighted Mario patch center. |
| `semantic_logits` | `[B,7,15,16]` | Unbounded seven-class logits. |
| `semantic_ids` | `[B,15,16]` | IDs `[0,6]`. |
| `tokens` | `[B,240,D]` | Transformer tokens; unbounded. Default `D=64`. |

The compatible sprite ViT uses the same grid with `K=13` and default `D=192`.
Its position is normalized `(x,y)` for the Mario class.

### Full SMB ViT

`FullSMBSegmentationVision` is backed by `FullSMBVisionTransformer` and accepts
HWC/BHWC or CHW/BCHW RGB input. Input is resized to `240x256` and divided into
16x16 patches, giving `G_H=15`, `G_W=16`, and `N=240`.

| Field | Shape | Range |
| --- | --- | --- |
| `position` | `[B,2]` | `[0,1]`, `(x,y)`, probability-weighted Mario patch center. |
| `semantic_logits` | `[B,13,15,16]` | Unbounded thirteen-class logits. |
| `semantic_ids` | `[B,15,16]` | IDs `[0,12]`. |
| `tokens` | `[B,240,D]` | Transformer tokens; unbounded. Default `D=192`. |

The thirteen semantic classes are exactly
`sky, ground, brick, question_block, pipe, coin, goomba, koopa, mario,
mushroom, hill, cloud, bush`.

### Full SMB Legacy DeepLab

`FullSMBDeepLabSegmentationVision` preserves the previous checkpoint wrapper for
inspection and migration. For normalized model input `[B,3,H,W]`:

| Field | Shape | Range |
| --- | --- | --- |
| `position` | `[B,2]` | `[0,1]`, `(x,y)`, probability-weighted Mario pixel center. |
| `semantic_logits` | `[B,6,H,W]` | Unbounded six-class logits. |
| `semantic_ids` | `[B,H,W]` | IDs `[0,5]`. |
| `tokens` | `[B,240,6]` | Logits pooled to 15x16 and flattened row-major; unbounded. |

DeepLab preserves input spatial size. Its token width is six class logits, not
a learned embedding width comparable to ViT tokens. New Full SMB policy code
should use the ViT-backed `FullSMBSegmentationVision`.

## Consumer Requirements

- Batch size agrees across every tensor in one object.
- A/B lengths follow `R_AB`; C length follows `R_BC` and is divisible by B
  length for the selected controller schedule.
- A/B IDs remain below `vocab_size` before embedding.
- Position order is `(x,y)` whenever `P=2`.
- Semantic class ordering is exactly `VisionSpec.semantic_classes`.
- Raw logits from different architectures are not calibrated probabilities.
- Metadata is observational and is not required by shared model code.
