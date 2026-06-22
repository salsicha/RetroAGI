# Known-Good Block SMB Scripted Policy

Date: 2026-06-22

This artifact is a deterministic scripted-policy baseline for the four fixed
Block SMB scenarios. It is intentionally labeled as scripted, not learned. Its
purpose is to provide a known-good checkpoint and evaluation artifact set while
learned-policy training is still being improved.

- Policy checkpoint: `artifacts/block_smb/known_good_scripted_seed20260622/policy.pth`
- Run summary: `artifacts/block_smb/known_good_scripted_seed20260622/run_summary.json`
- Evaluation recordings: `artifacts/block_smb/known_good_scripted_seed20260622/evaluation`
- Seed: `20260622`
- Evaluation episodes per scenario: `3`
- Evaluation max steps: `200`
- Success thresholds met: `true`

## Deterministic Evaluation

| Scenario | Mean Return | Success Rate | Threshold Met |
| --- | ---: | ---: | --- |
| `level_1_flat.json` | `69.26500000000001` | `1.0` | `true` |
| `level_2_gap.json` | `68.70500000000001` | `1.0` | `true` |
| `level_3_stairs.json` | `68.70500000000003` | `1.0` | `true` |
| `level_4_platforms.json` | `69.82500000000002` | `1.0` | `true` |

Mean return: `69.12500000000003`

Overall success rate: `1.0`

## Notes

The action scripts use the shared Block SMB action IDs:

- `1`: `RIGHT`
- `2`: `RIGHT_JUMP`

This checkpoint should be used as a regression baseline for environment,
recording, and threshold validation. It should not be reported as evidence that
the learned actor/world-model/critic policy has solved Block SMB.
