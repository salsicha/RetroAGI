# Block SMB Success Thresholds

These thresholds define when a Block SMB policy is considered successful on the
four fixed scenarios used by P3 evaluation. They are intentionally stricter
than a single lucky goal collision: each scenario must be solved repeatedly,
within a bounded time budget, and with enough return to show the policy is not
paying avoidable penalties.

The machine-readable source of truth is
`retroagi.stages.block_smb.success.FIXED_BLOCK_SMB_SUCCESS_THRESHOLDS`.

## Evaluation Protocol

Use deterministic evaluation with:

- at least `3` episodes per fixed scenario,
- at most `200` environment steps per episode,
- frozen Block ViT perception unless a run explicitly documents fine-tuning,
- the standard `MarioScenarioEnv` reward terms from `BlockSMBRewardConfig`.

An evaluation with fewer than `3` episodes or a larger step budget can still be
useful for debugging, but it does not qualify as a known-good Block SMB policy.

## Fixed Scenario Thresholds

| Scenario | Minimum Success Rate | Minimum Mean Return | Required Episodes | Max Steps |
| --- | ---: | ---: | ---: | ---: |
| `level_1_flat.json` | `1.0` | `55.0` | `3` | `200` |
| `level_2_gap.json` | `1.0` | `55.0` | `3` | `200` |
| `level_3_stairs.json` | `1.0` | `55.0` | `3` | `200` |
| `level_4_platforms.json` | `1.0` | `55.0` | `3` | `200` |

`success_rate` is the fraction of deterministic evaluation episodes that end by
reaching the scenario goal. The minimum return floor is paired with the success
rate requirement: a policy must reach the goal and must do it efficiently
enough to stay above the return floor.

With the default reward configuration, a clean goal completion typically earns:

- progress reward for moving right,
- `+50` for reaching the goal,
- optional coin rewards,
- minus the per-frame penalty.

The `55.0` return floor leaves room for different valid action timings while
rejecting policies that only barely complete after excessive delay or other
avoidable penalties.

## How To Check A Run

Run deterministic evaluation through the Block SMB CLI:

```bash
retroagi-block-smb evaluate \
  --checkpoint artifacts/block_smb/known_good/policy.pth \
  --evaluation-episodes 3 \
  --evaluation-max-steps 200
```

Each fixed scenario result includes:

- `threshold`: the scenario threshold,
- `threshold_met`: whether that scenario passed,
- `threshold_diagnostics`: booleans for success rate, return, episode count,
  and step-budget checks.

The top-level evaluation result includes `success_thresholds_met`, which is
true only when every fixed scenario passes its threshold.
