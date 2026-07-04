# Fresh Block-SMB Duration/Release Analysis

Date: 2026-07-04

This document analyzes the fresh Block-SMB distillation run that trained the
duration/release primitive heads directly from scripted fixed and Monte Carlo
sweeps.

## Artifacts

- Summary: `artifacts/fresh_block_smb_20260704_005140/block_smb_distill_summary.json`
- Epoch log: `artifacts/fresh_block_smb_20260704_005140/block_smb_distill_log.jsonl`
- Stdout capture: `artifacts/fresh_block_smb_20260704_005140/block_smb_distill_stdout.json`
- Checkpoint: `data/full_pipeline_20260704_005140_block_smb/block_smb/policy_duration_release_fresh.pth`
- Checkpoint metadata: `data/full_pipeline_20260704_005140_block_smb/block_smb/policy_duration_release_fresh.json`

## Configuration

- Device: CUDA
- Seed: `20260704`
- Epochs: `80`
- Batch size: `256`
- Learning rate: `8e-4`
- Hidden dim: `64`
- Training mode: independent teacher-forced distillation
- Fixed scenarios: all 16 current Block-SMB fixed scenarios
- MC training: full `block_smb_mc_v1` family x difficulty sweep, 45 train samples
- MC validation/test: 45 validation samples and 45 test samples
- Primitive timing loss weight: `0.5`
- Primitive hazard multiplier: `3.0`
- Fixed evaluation: 3 episodes per scenario, 200 max steps

The first attempt failed before epoch 1 because deterministic CUDA required
`CUBLAS_WORKSPACE_CONFIG`; the successful run used `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
A separate crash exposed during this work was fixed in commit `fca49aa` by
tracking `terminal_outcome` dynamics losses in the distillation accumulators.

## Verdict

This checkpoint is not promotion-quality and should not be transferred to
Full-SMB.

- Fixed Block-SMB success rate: `0.4583333333333333`
- Fixed threshold pass rate: `0.4375`
- Fixed success thresholds met: `false`
- Mean fixed return: `33.155833333333334`
- MC validation success rate: `0.37037037037037035`
- MC validation gate met: `false`
- MC test success rate: `0.37037037037037035`
- MC test gate met: `false`
- Semantic prediction gate met: `true`
- All-NOOP action collapse: `false`

The model is not collapsed to NOOP. It emits mostly `RIGHT` and `RIGHT_JUMP`,
but it does not learn reliable scenario-specific timing, waiting, left recovery,
or multi-obstacle sequencing.

## Dataset And Supervision

- Total examples: `12086`
- Dataset action counts:
  - `NOOP`: `316`
  - `RIGHT`: `7900`
  - `RIGHT_JUMP`: `3364`
  - `LEFT`: `462`
  - `LEFT_JUMP`: `44`
  - `JUMP`: `0`
- Primitive duration labels: `200`
- Primitive release labels: `3408`
- Positive release labels: `200`
- Post-release labels: `3408`
- Weighted primitive supervision count: `10224`

The duration/release heads are receiving labels, but duration labels remain
sparse compared with the full action dataset because each jump run contributes
one duration target. Wait and left-recovery examples are also rare compared with
rightward movement.

## Training Curve

| Epoch | Accuracy | Action Loss | Primitive Loss |
| ---: | ---: | ---: | ---: |
| 1 | `0.2722157868608307` | `2.758801583479951` | `0.46441009052710563` |
| 10 | `0.28669534999172597` | `2.059196494173258` | `0.30703448581760656` |
| 20 | `0.3210325997021347` | `1.7405022807060517` | `0.31716200684191237` |
| 30 | `0.7992718848254179` | `0.6998365774674268` | `0.27671061785641676` |
| 33 | `0.8326989905675989` | `0.6120504341130617` | `0.28832478988216` |
| 40 | `0.6566275028959127` | `1.6179527506890645` | `0.29097310466599074` |
| 50 | `0.42975343372497105` | `2.2428017318613693` | `0.3046677238300958` |
| 60 | `0.46963428760549397` | `1.8085170104451158` | `0.3622884810533071` |
| 70 | `0.73183849081582` | `1.1789287713294825` | `0.26928133681852967` |
| 78 | `0.8311269237133874` | `0.6566766973917164` | `0.2145273430806486` |
| 80 | `0.5820784378619891` | `1.592424301680559` | `0.3310042336846984` |

Training was unstable. The best raw action accuracy was epoch `33`, and the
best primitive loss was epoch `78`, but the saved final epoch degraded
substantially. This is a direct reason to add validation-based checkpoint
selection or early stopping before treating this distillation path as reliable.

## Fixed Scenario Results

Action counts are listed as `NOOP/RIGHT/RIGHT_JUMP/LEFT/LEFT_JUMP/JUMP`.

| Scenario | Success Rate | Threshold Met | Action Counts |
| --- | ---: | --- | --- |
| `level_1_flat.json` | `1.0` | `true` | `0/146/73/0/0/0` |
| `level_2_gap.json` | `0.0` | `false` | `0/69/48/0/0/0` |
| `level_3_stairs.json` | `0.0` | `false` | `0/510/90/0/0/0` |
| `level_4_platforms.json` | `0.0` | `false` | `0/94/59/0/0/0` |
| `level_5_enemy_hop.json` | `1.0` | `true` | `0/154/71/0/0/0` |
| `level_6_enemy_patrol.json` | `1.0` | `true` | `0/166/59/0/0/0` |
| `level_7_moving_bridge.json` | `1.0` | `true` | `0/154/73/0/0/0` |
| `level_8_enemy_gap.json` | `0.0` | `false` | `0/60/57/0/0/0` |
| `level_9_enemy_stomp.json` | `1.0` | `true` | `0/152/69/0/0/0` |
| `level_10_left_retreat.json` | `1.0` | `true` | `0/3/0/123/30/0` |
| `level_11_left_jump_recovery.json` | `0.0` | `false` | `9/561/30/0/0/0` |
| `level_12_wait_bridge.json` | `1.0` | `true` | `0/165/51/0/0/0` |
| `level_13_variable_pits.json` | `0.0` | `false` | `0/69/48/0/0/0` |
| `level_14_under_enemy_platform.json` | `0.0` | `false` | `0/39/48/0/0/0` |
| `level_15_wait_long_bridge.json` | `0.0` | `false` | `0/119/52/0/0/0` |
| `level_16_wait_enemy_gate.json` | `0.3333333333333333` | `false` | `0/356/155/0/0/0` |

The fixed failures show three separate problems:

- Gap/pit/platform timing fails even though `RIGHT_JUMP` is emitted.
- Wait scenarios do not reliably emit `NOOP`; even explicit wait examples are
  being overwhelmed by rightward movement.
- Left jump recovery collapses back to rightward behavior.

## Monte Carlo Family Results

| Family | Validation Success | Test Success |
| --- | ---: | ---: |
| `chained_enemy_gauntlet` | `0.000` | `0.000` |
| `enemy_gap` | `0.000` | `0.000` |
| `mixed_section` | `0.000` | `0.000` |
| `platform_chain` | `0.000` | `0.000` |
| `single_gap` | `0.000` | `0.000` |
| `stair_climb` | `0.000` | `0.111` |
| `full_smb_opening_proxy` | `0.222` | `0.000` |
| `chained_obstacles` | `0.333` | `0.222` |
| `enemy_patrol` | `0.333` | `0.333` |
| `retreat_recovery` | `0.333` | `0.333` |
| `wait_timing` | `0.667` | `0.667` |
| `flat_run` | `0.778` | `0.889` |
| `moving_bridge` | `0.889` | `1.000` |
| `enemy_hop` | `1.000` | `1.000` |
| `enemy_stomp` | `1.000` | `1.000` |

The strongest families are enemy-hop, enemy-stomp, and moving-bridge. The
weakest families are exactly the transfer-critical ones: gaps, stairs,
platforms, chained gauntlets, mixed sections, and Full-SMB-opening proxies.

## Interpretation

The latest primitive timing-head work helped expose timing supervision in the
training metrics, but the policy is still not learning a robust executable
skill. The final model knows that jumping is often needed, but it does not
choose the right jump window, hold duration, release timing, or recovery mode
often enough to solve the gate.

The key evidence:

- Action collapse gates are working: the model is not all-NOOP.
- The policy is not purely passive: `RIGHT` and `RIGHT_JUMP` dominate.
- Duration/release labels are present and primitive loss improves during some
  epochs.
- Scenario execution still fails because per-frame action imitation is not
  enough to preserve the action-span schedule across long obstacle windows.
- Late-epoch instability means the final saved checkpoint can be worse than
  earlier training states.

## Required Follow-Ups

1. Add validation-based checkpoint selection to the distiller.
   Save and evaluate the best epoch by fixed success, MC validation pass rate,
   and primitive timing loss instead of only saving the final epoch.

2. Add explicit fixed/MC validation during training.
   The training log should include scenario success snapshots so overtraining
   or action-mode drift is visible before the final evaluation.

3. Strengthen wait and left-recovery supervision.
   NOOP is only `316 / 12086` examples and LEFT_JUMP is only `44 / 12086`.
   These modes need stronger losses or balanced sampling, otherwise they are
   washed out by rightward movement.

4. Make jump duration labels denser around obstacles.
   There are only `200` duration labels for `12086` examples. Add explicit
   obstacle-window duration labels near pit, platform, stair, and pipe windows,
   not just one label at each scripted jump start.

5. Evaluate best-epoch candidates before Full-SMB transfer.
   A checkpoint should not transfer unless fixed thresholds pass and MC
   validation gates pass. This run does neither.

6. Consider sequence/recurrent distillation for action spans.
   The fresh run used independent teacher-forced batches. The failure pattern is
   duration and horizon-sensitive, so sequence training or span-level primitive
   targets should be part of the next run.

## Promotion Decision

Do not transfer this checkpoint to Full-SMB.

The model fails fixed Block-SMB thresholds, fails held-out MC validation, and
fails held-out MC test. It is a useful diagnostic checkpoint, but not a
full-SMB warm-start candidate.
