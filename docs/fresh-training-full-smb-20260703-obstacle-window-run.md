# Fresh Block-to-Full SMB Training Run - 2026-07-03

Run directory: `artifacts/fresh_full_smb_20260703_230924`

## Result

The fresh run did not produce a policy that can play Full SMB level 1-1.

Block SMB training completed, but deterministic validation collapsed to all
`NOOP` actions. The normal Full SMB transfer gate correctly rejected the
checkpoint. A forced diagnostic transfer was still evaluated in Full SMB, then
fine-tuned with the Full SMB imitation warm start and obstacle-window label
pipeline. The fine-tuned policy also failed level 1-1.

## Environment

The Full SMB environment probe passed before training:

- Output: `artifacts/fresh_full_smb_20260703_230924/full_smb/env_check.json`
- Backend: stable-retro Full SMB
- Device used for training/evaluation: CUDA, Tesla V100-SXM2-32GB

Local save states for the first enemy/pipe obstacle-window sweep were generated:

- `local/full_smb/states/curriculum/1_1_first_enemy_approach.state`
- `local/full_smb/states/curriculum/1_1_midpipe.state`
- Manifest: `artifacts/fresh_full_smb_20260703_230924/full_smb/save_state_manifest.json`

## Block SMB Training

Command family: `python -m retroagi.stages.block_smb.cli train`

Key settings:

- Epochs: `2`
- Episodes per epoch: `16`
- Rollout steps: `160`
- Fixed scenarios: `16`
- Monte Carlo parameter sweep: enabled
- Monte Carlo train samples: `45`
- MC families included chained obstacles, chained enemy gauntlet, full-SMB-opening proxy, and mixed section families

Artifacts:

- Summary: `artifacts/fresh_full_smb_20260703_230924/block_smb/train_summary.json`
- Log: `artifacts/fresh_full_smb_20260703_230924/logs/block_train.jsonl`
- Checkpoint: `artifacts/fresh_full_smb_20260703_230924/checkpoints/block_policy.pth`

Final Block SMB metrics:

- Final train mean return: `1.5431250000000003`
- Fixed validation success rate: `0.0`
- Fixed validation mean return: `-1.6`
- Fixed validation action counts: `NOOP=2560`, all other actions `0`
- MC validation success rate: `0.0`
- MC validation mean return: `-1.5999999999999994`
- MC validation action counts: `NOOP=7200`, all other actions `0`
- Fixed all-`NOOP` collapse gate: `true`
- MC all-`NOOP` collapse gate: `true`

The gated transfer failed as expected:

```text
Block SMB checkpoint is not eligible for Full SMB transfer: fixed scenario threshold pass rate is missing or below 1.0, held-out Monte Carlo validation gate is missing or failed, fixed deterministic policy collapsed to all NOOP actions, Monte Carlo validation deterministic policy collapsed to all NOOP actions
```

## Forced Transfer Diagnostic

Because the normal transfer was blocked, a forced diagnostic transfer was
created only to run the failed policy through Full SMB:

- Forced checkpoint: `artifacts/fresh_full_smb_20260703_230924/checkpoints/transferred_policy_forced.pth`
- Zero-shot Full SMB eval: `artifacts/fresh_full_smb_20260703_230924/evaluations/full_zero_shot_forced_level_1_1.json`

Forced zero-shot Full SMB level 1-1 result:

- Success rate: `0.0`
- Completion rate: `0.0`
- Max progress: `40.0`
- Mean return: `0.0`
- Death count: `0.0`
- Steps: `2400`

This confirms the transfer gate was preventing a non-moving policy from entering
the normal Full SMB pipeline.

## Full SMB Fine-Tune

Command family: `python -m retroagi.stages.full_smb.train train`

Key settings:

- Mode: `fine-tune`
- Init checkpoint: forced transferred checkpoint
- Imitation warm start: enabled
- Warm-start steps: `600`
- Warm-start epochs: `3`
- Obstacle-window duration labels: enabled
- Epochs: `1`
- Updates per epoch: `4`
- Rollout steps: `160`

Artifacts:

- Summary: `artifacts/fresh_full_smb_20260703_230924/full_smb/train_summary.json`
- Log: `artifacts/fresh_full_smb_20260703_230924/logs/full_train.jsonl`
- Checkpoint: `artifacts/fresh_full_smb_20260703_230924/checkpoints/full_policy.pth`

Warm-start observations:

- Scripted opening samples collected: `85`
- Scripted opening max progress: `312.0`
- Release supervision count: `10`
- Duration supervision count: `2`
- Final warm-start action accuracy: `0.0`
- Mean warm-start action accuracy: `0.003472222222222222`

Obstacle-window duration labels did not add training data:

- Windows attempted: `2`
- Trial count: `12`
- Windows labeled: `0`
- Skipped windows:
  - `first_enemy_approach`: `no_candidate_progressed`
  - `first_pipe_midpipe`: `no_candidate_progressed`

Full SMB rollout/evaluation result:

- Four fine-tune rollouts all returned `0.0`
- Epoch evaluation max progress: `40.0`
- Epoch evaluation success rate: `0.0`
- Epoch evaluation mean return: `0.0`

## Final Full SMB Evaluation

Artifacts:

- Final eval: `artifacts/fresh_full_smb_20260703_230924/evaluations/full_finetuned_level_1_1.json`
- Recorded final eval: `artifacts/fresh_full_smb_20260703_230924/evaluations/full_finetuned_level_1_1_record.json`
- Episode recording: `artifacts/fresh_full_smb_20260703_230924/recordings/final_eval/evaluation/evaluation_episode0000.npz`

Final Full SMB level 1-1 result:

- Success rate: `0.0`
- Completion rate: `0.0`
- Max progress: `40.0`
- Mean return: `0.0`
- Death count: `0.0`
- Steps: `2400`

Recorded action counts from the final Full SMB rollout:

- `LEFT`: `2239`
- `LEFT_JUMP`: `12`
- `NOOP`: `149`
- `RIGHT`: `0`
- `RIGHT_JUMP`: `0`
- `JUMP`: `0`

The fine-tuned policy was not an all-`NOOP` policy, but it became a leftward
policy and stayed near or behind the spawn point. The first recorded progress
values were `[39.0, 37.0, 33.0, 29.0, 23.0]`, and final progress was `0.0`.

## Blocking Issues

1. Block SMB fresh training still collapses under deterministic validation.
   The new action-collapse gate catches this correctly, but the trainer is not
   producing transferable behavior from the current short MC sweep run.

2. Full SMB imitation warm start is not successfully fitting the scripted
   opening labels. The final action accuracy of `0.0` and the later leftward
   policy indicate the warm-start target/action path needs debugging before it
   can recover a collapsed source checkpoint.

3. Obstacle-window duration labeling is wired into the run, but both save-state
   sweeps produced `no_candidate_progressed`. The save states are generated and
   readable, but the candidate sweep policy/action script is not producing
   usable progress labels from those windows.

4. Full SMB fine-tuning did not improve progress beyond spawn. The final policy
   takes mostly `LEFT` actions and never reaches the first enemy or pipe region.

## Conclusion

This run fully exercised the current Block SMB to Full SMB pipeline, including
MC sweep training, deterministic transfer gates, forced diagnostic transfer,
Full SMB imitation warm start, obstacle-window label attempts, Full SMB
fine-tuning, and a recorded 2400-step level 1-1 evaluation.

The model does not play Full SMB successfully. The immediate engineering focus
should be the Full SMB imitation warm-start target path and the obstacle-window
sweep candidate generation, followed by another gated Block SMB run once those
diagnostics are fixed.
