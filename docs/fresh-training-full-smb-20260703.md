# Fresh Block SMB To Full SMB Run - 2026-07-03

## Summary

Run directory: `artifacts/fresh_full_smb_20260703_181025`

This was a fresh policy run after removing the Level 1-1 primitive planner. The
pipeline completed a new Block SMB training run, transferred that checkpoint to
Full SMB, evaluated the transferred policy directly on Full SMB Level 1-1, ran a
short Full SMB fine-tune, and evaluated/recorded the fine-tuned policy.

Result: the model does not complete Full SMB Level 1-1. The zero-shot
transferred policy stalled at spawn-level progress. The fine-tuned policy moved
right, reached progress 307, then died after 93 Full SMB policy steps.

## Commands

Environment check:

```bash
python -m retroagi.cli check-env --game smb --stage full \
  --seed 0 --steps 8 --frame-skip 2 \
  --output artifacts/fresh_full_smb_20260703_181025/full_smb/env_check.json
```

The Full SMB backend check passed.

An initial Block SMB run attempted full in-training Monte Carlo sweep validation
with `--monte-carlo-parameter-sweep`. It reached epoch 1, then spent too long in
`monte_carlo_validation` and was interrupted. A second bounded sweep attempt hit
the same in-loop validation path. Logs are preserved as:

- `artifacts/fresh_full_smb_20260703_181025/logs/block_train.jsonl`
- `artifacts/fresh_full_smb_20260703_181025/logs/block_train_complete.jsonl`

The completed fresh Block SMB run used the current fixed curriculum:

```bash
python -m retroagi.stages.block_smb.cli train \
  --seed 20260703 --device cuda --nondeterministic \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/block_policy.pth \
  --output artifacts/fresh_full_smb_20260703_181025/block_smb/train_summary.json \
  --log-path artifacts/fresh_full_smb_20260703_181025/logs/block_train_final.jsonl \
  --epochs 1 --episodes-per-epoch 8 --rollout-steps 120 \
  --evaluation-episodes 1 --evaluation-max-steps 120 \
  --world-model-slot-weight terminal_outcome=6.0 \
  --world-model-slot-weight state=2.0 \
  --world-model-slot-weight support_state=2.0 \
  --reward-loss-weight 0.05 --value-loss-weight 0.35 \
  --action-aux-weight 0.05 --critic-loss-weight 0.002 \
  --target-network-mode auto --enable-recurrent-state \
  --enable-checkpoint-transfer
```

Transfer used `--allow-ungated-block-source` because the fresh Block SMB policy
failed its fixed-scenario gates:

```bash
python -m retroagi.stages.full_smb.transfer \
  --block-policy-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/block_policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/transferred_policy.pth \
  --device cuda --allow-ungated-block-source
```

Zero-shot Full SMB evaluation:

```bash
python -m retroagi.stages.full_smb.train evaluate \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/transferred_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --output-summary artifacts/fresh_full_smb_20260703_181025/evaluations/full_zero_shot_level_1_1.json
```

Full SMB fine-tune:

```bash
python -m retroagi.stages.full_smb.train train \
  --mode fine-tune \
  --init-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/transferred_policy.pth \
  --checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set curriculum \
  --seed 20260703 --epochs 1 --updates-per-epoch 2 --rollout-steps 128 \
  --evaluation-episodes 1 --evaluation-max-steps 512 \
  --frame-skip 4 --device cuda --nondeterministic --save-checkpoints \
  --recording-dir artifacts/fresh_full_smb_20260703_181025/recordings \
  --recording-path artifacts/fresh_full_smb_20260703_181025/recordings/full_train_recording_manifest.npz \
  --log-path artifacts/fresh_full_smb_20260703_181025/logs/full_train.jsonl \
  --output-summary artifacts/fresh_full_smb_20260703_181025/full_smb/train_summary.json
```

Final Full SMB evaluation and recording:

```bash
python -m retroagi.stages.full_smb.train evaluate \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --output-summary artifacts/fresh_full_smb_20260703_181025/evaluations/full_finetuned_level_1_1.json

python -m retroagi.stages.full_smb.train record \
  --policy-checkpoint artifacts/fresh_full_smb_20260703_181025/checkpoints/full_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze --task-set fixed_benchmark --level 1-1 \
  --evaluation-episodes 1 --evaluation-max-steps 2400 \
  --frame-skip 4 --device cuda \
  --record-dir artifacts/fresh_full_smb_20260703_181025/recordings/final_eval \
  --recording-path artifacts/fresh_full_smb_20260703_181025/recordings/final_eval_manifest.npz \
  --output-summary artifacts/fresh_full_smb_20260703_181025/evaluations/full_finetuned_level_1_1_record.json
```

## Metrics

### Block SMB Source

Artifact: `artifacts/fresh_full_smb_20260703_181025/block_smb/train_summary.json`

The completed Block SMB run trained for one epoch over the 16 fixed scenarios,
including variable pits, under-enemy-platform, and delay scenarios. It did not
pass any fixed-scenario threshold.

| Metric | Value |
| --- | ---: |
| train mean return | -0.7831 |
| eval mean return | -1.2 |
| eval success rate | 0.0 |
| eval threshold pass rate | 0.0 |
| semantic prediction accuracy | 0.0 |
| dynamics terminal outcome MAE | 0.0436 |
| total loss | 0.5516 |

The source checkpoint was therefore not a valid transfer candidate by the normal
promotion gates. Transfer was forced only to measure Full SMB behavior.

### Full SMB Zero-Shot Transfer

Artifact: `artifacts/fresh_full_smb_20260703_181025/evaluations/full_zero_shot_level_1_1.json`

| Metric | Value |
| --- | ---: |
| steps | 2400 |
| max progress | 40.0 |
| mean return | 0.0 |
| completion rate | 0.0 |
| survival rate | 1.0 |
| death count | 0.0 |
| threshold pass rate | 0.0 |

The transferred model survived only because it effectively failed to leave the
spawn area. It did not approach the first enemy, pipe, or any Level 1-1 gate.

### Full SMB Fine-Tuned Policy

Artifacts:

- `artifacts/fresh_full_smb_20260703_181025/full_smb/train_summary.json`
- `artifacts/fresh_full_smb_20260703_181025/evaluations/full_finetuned_level_1_1.json`
- `artifacts/fresh_full_smb_20260703_181025/evaluations/full_finetuned_level_1_1_record.json`

| Metric | Value |
| --- | ---: |
| final eval steps | 93 |
| max progress | 307.0 |
| mean return | 231.0 |
| completion rate | 0.0 |
| survival rate | 0.0 |
| death count | 1.0 |
| threshold pass rate | 0.0 |

The fine-tuned model learned forward motion and reached the early Level 1-1
enemy region, but died before any benchmark success criterion. The final
recorded rollout ended with `termination_reason=life_lost`, `death=true`,
`progress=307.0`, and `lives=1`.

Recorded rollout:

`artifacts/fresh_full_smb_20260703_181025/recordings/final_eval/evaluation/evaluation_episode0000.npz`

Action distribution in the final recorded rollout:

| Action | Count |
| --- | ---: |
| RIGHT | 64 |
| RIGHT_JUMP | 19 |
| LEFT_JUMP | 5 |
| NOOP | 4 |
| LEFT | 1 |

The first 18 executed actions were `RIGHT_JUMP`. The final 20 actions were
mostly `RIGHT`, with two `NOOP`s. This is still not a stable learned motor
program: it begins with an overlong jump burst, then mostly runs right into the
early hazard region.

## Analysis

The failure starts before Full SMB transfer. The fresh Block SMB source did not
solve even the fixed Block SMB curriculum. All fixed scenarios returned -1.2 and
success rate 0.0, so the transferred Full SMB model began from a weak policy.
The low terminal-outcome MAE only says the model can fit mostly non-terminal
terminal bits in a short run; it does not imply the action policy knows how to
avoid death.

Zero-shot Full SMB behavior confirms the transfer gap. The transferred model
made no useful progress in 2,400 policy steps. That is worse than the
fine-tuned result, but it is also safer: it did not die because it did not
engage the level.

The short Full SMB fine-tune changed the policy enough to move right. That
proves the Full SMB training path, checkpoint transfer, recording, and fixed
benchmark evaluation are functional without the removed primitive planner.
However, the resulting behavior died at progress 307, which is consistent with
reaching the first early hazard region without a reliable jump/release/cooldown
sequence.

The action trace shows the same temporal-control problem seen in earlier runs:
the policy can choose `RIGHT` and `RIGHT_JUMP`, but it does not yet execute a
well-timed option. The opening uses a long burst of `RIGHT_JUMP`, then mostly
`RIGHT`. The generic jump terminator and one-second walk limiter prevent some
pathological holds, but they are not enough to turn per-frame logits into a
solved Level 1-1 route.

The Monte Carlo sweep path also needs runtime work. Enabling
`--monte-carlo-parameter-sweep` currently causes Block SMB evaluation to attach
a full validation sweep whenever the config is evaluated. That made the
training runs spend excessive time inside `monte_carlo_validation` before a
checkpoint was saved. For routine fresh-run analysis, the sweep should either
save a checkpoint before validation or expose a way to use sweep samples for
training without automatically running the whole validation sweep in every
evaluation.

## Conclusion

The current architecture and training code can complete the mechanics of the
pipeline:

1. train a fresh Block SMB policy;
2. save and transfer the checkpoint to Full SMB;
3. fine-tune in Full SMB;
4. evaluate and record Full SMB rollouts without the Level 1-1 primitive
   planner.

The resulting model does not play Full SMB successfully. It does not complete
Level 1-1, and after short Full SMB fine-tuning it dies around progress 307.

The next required work is not another planner. The training source needs to pass
Block SMB fixed and Monte Carlo gates before transfer, and the learned motor
controller needs explicit sequence-level supervision or loss terms for jump
hold, release, cooldown, and replan timing. Without that, Full SMB training can
induce forward motion, but it still fails at the first real timing hazard.
