# Fresh Parameterized-Primitive SMB Run - 2026-07-03

## Summary

Run directory: `artifacts/fresh_full_smb_20260703_215926`

Code revision: `79a8e92`

This run was executed after adding explicit Level-B primitive heads and the
stateful `SMBParameterizedPrimitiveExecutor`. The pipeline completed:

1. Full SMB environment/backend validation.
2. Fresh Block SMB training with fixed scenarios plus small Monte Carlo training
   samples.
3. Ungated transfer to Full SMB.
4. Zero-shot Full SMB Level 1-1 evaluation.
5. Short Full SMB fine-tuning with parameterized primitive execution.
6. Final deterministic Full SMB Level 1-1 evaluation and recording.

Result: the model still does not complete Full SMB Level 1-1. The fresh Block
SMB deterministic policy collapsed to `NOOP`. Zero-shot Full SMB therefore
stalled at spawn. Full SMB fine-tuning recovered forward movement, but the final
policy used only a two-step `RIGHT_JUMP` at the start, then mostly held `RIGHT`
and died at progress `316`.

## Commands

Environment check:

```bash
python -m retroagi.cli check-env --game smb --stage full \
  --seed 0 --steps 8 --frame-skip 2 \
  --output artifacts/fresh_full_smb_20260703_215926/full_smb/env_check.json
```

Fresh Block SMB training:

```bash
python -m retroagi.stages.block_smb.cli train \
  --seed 20260703 --device cuda --nondeterministic \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/block_policy.pth \
  --output artifacts/fresh_full_smb_20260703_215926/block_smb/train_summary.json \
  --log-path artifacts/fresh_full_smb_20260703_215926/logs/block_train.jsonl \
  --epochs 2 --episodes-per-epoch 16 --rollout-steps 160 \
  --evaluation-episodes 1 --evaluation-max-steps 160 \
  --monte-carlo-train-samples-per-epoch 8 \
  --monte-carlo-validation-samples 4 --monte-carlo-test-samples 0 \
  --world-model-slot-weight terminal_outcome=6.0 \
  --world-model-slot-weight state=2.0 \
  --world-model-slot-weight support_state=2.0 \
  --reward-loss-weight 0.05 --value-loss-weight 0.35 \
  --action-aux-weight 0.10 --critic-loss-weight 0.002 \
  --target-network-mode auto --enable-recurrent-state \
  --enable-checkpoint-transfer
```

Transfer and zero-shot Full SMB evaluation:

```bash
python -m retroagi.stages.full_smb.transfer \
  --block-policy-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/block_policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/transferred_policy.pth \
  --device cuda --allow-ungated-block-source

python -m retroagi.stages.full_smb.train evaluate \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/transferred_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --output-summary artifacts/fresh_full_smb_20260703_215926/evaluations/full_zero_shot_level_1_1.json
```

Full SMB fine-tune, final evaluation, and recording:

```bash
python -m retroagi.stages.full_smb.train train \
  --mode fine-tune \
  --init-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/transferred_policy.pth \
  --checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set curriculum \
  --seed 20260703 --epochs 1 --updates-per-epoch 4 \
  --rollout-steps 160 --evaluation-episodes 1 \
  --evaluation-max-steps 512 --frame-skip 4 --device cuda \
  --nondeterministic --action-aux-weight 0.10 \
  --critic-loss-weight 0.002 --world-model-weight 0.05 \
  --save-checkpoints \
  --recording-dir artifacts/fresh_full_smb_20260703_215926/recordings \
  --recording-path artifacts/fresh_full_smb_20260703_215926/recordings/full_train_recording_manifest.npz \
  --log-path artifacts/fresh_full_smb_20260703_215926/logs/full_train.jsonl \
  --output-summary artifacts/fresh_full_smb_20260703_215926/full_smb/train_summary.json

python -m retroagi.stages.full_smb.train evaluate \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --output-summary artifacts/fresh_full_smb_20260703_215926/evaluations/full_finetuned_level_1_1.json

python -m retroagi.stages.full_smb.train record \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_215926/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --record-dir artifacts/fresh_full_smb_20260703_215926/recordings/final_eval \
  --recording-path artifacts/fresh_full_smb_20260703_215926/recordings/final_eval_manifest.npz \
  --output-summary artifacts/fresh_full_smb_20260703_215926/evaluations/full_finetuned_level_1_1_record.json
```

## Metrics

### Environment

Artifact: `artifacts/fresh_full_smb_20260703_215926/full_smb/env_check.json`

The Full SMB backend check passed. Backend probe fields for action repeat,
frame step, headless reset, render, deterministic reset seed, and save/load
state were all true.

### Block SMB Source

Artifact: `artifacts/fresh_full_smb_20260703_215926/block_smb/train_summary.json`

| Metric | Value |
| --- | ---: |
| train mean return | 6.8231 |
| loss total | 0.9204 |
| loss policy | 0.2894 |
| loss action aux | 1.8668 |
| loss dynamics | 0.1096 |
| dynamics semantic prediction accuracy | 0.0 |
| dynamics terminal outcome MAE | 0.1724 |
| fixed eval mean return | -1.6 |
| fixed eval success rate | 0.0 |
| fixed threshold pass rate | 0.0 |
| MC validation success rate | 0.0 |

Block deterministic evaluation failed every fixed scenario. Monte Carlo
validation action counts were:

| Action ID | Count |
| --- | ---: |
| `0` / `NOOP` | 640 |
| `1` / `RIGHT` | 0 |
| `2` / `RIGHT_JUMP` | 0 |
| `3` / `LEFT` | 0 |
| `4` / `LEFT_JUMP` | 0 |
| `5` / `JUMP` | 0 |

The fresh source checkpoint did not meet transfer-source gates. Transfer was
forced only to measure downstream Full SMB behavior.

The Block Monte Carlo training set included 8 samples from these families:
`flat_run`, `single_gap`, `stair_climb`, `platform_chain`, `moving_bridge`,
`enemy_hop`, `enemy_patrol`, and `enemy_gap`. It did not include the chained,
mixed-section, retreat-recovery, wait-timing, enemy-stomp, or
full-SMB-opening-proxy families in this bounded run.

### Full SMB Zero-Shot Transfer

Artifact: `artifacts/fresh_full_smb_20260703_215926/evaluations/full_zero_shot_level_1_1.json`

| Metric | Value |
| --- | ---: |
| steps | 2400 |
| max progress | 40.0 |
| mean return | 0.0 |
| completion rate | 0.0 |
| survival rate | 1.0 |
| death count | 0.0 |
| threshold pass rate | 0.0 |

The transferred model did not leave the spawn area. It survived because it did
not engage the level.

### Full SMB Fine-Tuned Policy

Artifacts:

- `artifacts/fresh_full_smb_20260703_215926/full_smb/train_summary.json`
- `artifacts/fresh_full_smb_20260703_215926/evaluations/full_finetuned_level_1_1.json`
- `artifacts/fresh_full_smb_20260703_215926/evaluations/full_finetuned_level_1_1_record.json`
- `artifacts/fresh_full_smb_20260703_215926/recordings/final_eval/evaluation/evaluation_episode0000.npz`

| Metric | Value |
| --- | ---: |
| final eval steps | 88 |
| max progress | 316.0 |
| mean return | 242.0 |
| completion rate | 0.0 |
| survival rate | 0.0 |
| death count | 1.0 |
| threshold pass rate | 0.0 |

Final recorded action distribution:

| Action | Count |
| --- | ---: |
| `RIGHT` | 81 |
| `NOOP` | 5 |
| `RIGHT_JUMP` | 2 |

The first 30 actions were:

```text
RIGHT_JUMP, RIGHT_JUMP, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT,
RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, NOOP, RIGHT, RIGHT, RIGHT,
RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT
```

The maximum recorded progress was at step 35: position `[316.0, 176.0]`.
The last nonterminal frames stayed at x progress `316.0` while Mario fell from
y `120.0` to `239.0`; the terminal frame recorded `death=true` and
`termination_reason=life_lost`.

## Analysis

The parameterized primitive architecture is now active in the training and
evaluation path, but the fresh policy did not learn a valid Block SMB source
skill. The clearest failure is deterministic action collapse: despite positive
sampled training return, deterministic Block evaluation selected only `NOOP`.
That means the policy logits and primitive heads were not trained strongly
enough to make forward movement the top deterministic action.

This creates the same transfer issue as the previous run, but with a different
failure shape. Previously the policy often overused `RIGHT_JUMP`. In this run,
Block SMB deterministic evaluation used no jump or movement at all. Full SMB
zero-shot reflected that source failure exactly: it stalled at progress `40.0`
for the full 2,400-step budget.

Direct Full SMB fine-tuning did recover movement. The final policy reached
progress `316.0`, compared with the zero-shot `40.0`, but it still died before
the first benchmark gate. The final action trace shows the post-change temporal
control problem: the executor no longer produces an overlong jump burst. Instead
the learned duration/control is too short or too rare. The policy opens with
only two `RIGHT_JUMP` actions, then mostly runs right. That is insufficient for
the first real Level 1-1 hazard.

The primitive auxiliary loss is being computed and optimized, but it remains
high (`1.8668` in Block SMB). The bounded run did not provide enough direct
duration supervision to make the duration bins meaningful. The current auxiliary
loss is mostly self-consistency around selected actions; it is not yet a strong
teacher for "hold A this many frames before this obstacle." Full SMB imitation
now has primitive-target support, but this run did not include an imitation
warm-start pass.

The small Monte Carlo sample set was useful as a smoke test, but it was not
representative enough for transfer. It omitted chained obstacle, wait timing,
enemy stomp, mixed-section, retreat-recovery, and full-SMB-opening-proxy
families. Deterministic validation then used only `NOOP`, so the Monte Carlo
gate correctly failed with 0% success.

The Full SMB pipeline itself remains functional without the removed primitive
planner: checkpoint transfer works, Full SMB fine-tuning updates the policy,
evaluation records deterministic rollouts, and the new primitive executor is
used in the emulator path. The problem is policy quality and supervision, not a
broken Full SMB execution loop.

## Conclusion

The current fresh model does not play Full SMB successfully and cannot complete
Level 1-1. Parameterized primitives changed the observed failure from overlong
jumping toward under-jumping or no-jump-at-hazard behavior. That is useful
evidence that the executor is affecting behavior, but the policy still needs
stronger training targets.

Required next work:

1. Add a fresh Full SMB imitation warm start to this pipeline so scripted jump
   runs directly supervise duration bins and release timing.
2. Expand Block Monte Carlo training to include chained and
   full-SMB-opening-proxy families in routine fresh runs.
3. Add deterministic action-collapse gates before transfer: reject policies
   whose fixed or MC validation action counts are all `NOOP`.
4. Add explicit obstacle-window duration labels from save-state sweeps near the
   first enemy/pipe region.
