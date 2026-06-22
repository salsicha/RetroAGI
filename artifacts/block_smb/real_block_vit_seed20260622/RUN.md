# Block SMB Real Policy Training Run

Date: 2026-06-22

This run exercised the Block SMB policy trainer with the actual trained Block
ViT perception checkpoint:

- Vision checkpoint: `data/block_vit/block_vit.pth`
- Vision mode: frozen
- Policy checkpoint: `artifacts/block_smb/real_block_vit_seed20260622/policy.pth`
- Run summary: `artifacts/block_smb/real_block_vit_seed20260622/run_summary.json`
- Seed: `20260622`
- Device: `cpu`

Command:

```bash
python -m retroagi.stages.block_smb.cli train \
  --device cpu \
  --seed 20260622 \
  --epochs 1 \
  --episodes-per-epoch 2 \
  --rollout-steps 32 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 120 \
  --generated-scenarios 2 \
  --vision-checkpoint data/block_vit/block_vit.pth \
  --checkpoint artifacts/block_smb/real_block_vit_seed20260622/policy.pth \
  --record \
  --record-dir artifacts/block_smb/real_block_vit_seed20260622/evaluation \
  --output artifacts/block_smb/real_block_vit_seed20260622/run_summary.json
```

Resolved curriculum:

- `level_1_flat.json`
- `level_2_gap.json`
- `level_3_stairs.json`
- `level_4_platforms.json`
- `generated_000`
- `generated_001`

Training metrics:

- `episodes`: `2.0`
- `mean_return`: `-0.3049999999999999`
- `loss_total`: `0.00747208297252655`
- `loss_actor_pass1`: `1.2113982439041138`
- `loss_actor_pass2`: `0.0011231405660510063`
- `loss_world_model`: `0.10793090611696243`
- `loss_critic`: `0.004040311090648174`
- `loss_entropy`: `1.6562169790267944`
- `gradient_norm`: `0.5188508033752441`

Deterministic evaluation:

| Scenario | Return | Success Rate |
| --- | ---: | ---: |
| `level_1_flat.json` | `-1.2` | `0.0` |
| `level_2_gap.json` | `-1.2` | `0.0` |
| `level_3_stairs.json` | `-1.2` | `0.0` |
| `level_4_platforms.json` | `-1.2` | `0.0` |

Evaluation artifacts:

- `evaluation/level_1_flat.json_episode0.npz`
- `evaluation/level_2_gap.json_episode0.npz`
- `evaluation/level_3_stairs.json_episode0.npz`
- `evaluation/level_4_platforms.json_episode0.npz`

The run is a real checkpoint-backed baseline, not a known-good policy. The
zero success rate should feed the next P3 work: define scenario success
thresholds, then tune the objective/rewards/config until those thresholds are
met reproducibly.
