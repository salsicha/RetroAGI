# Full Pipeline Run Summary - 2026-06-26

Environment:

- Python: 3.12.3
- PyTorch: 2.7.0+cu126
- GPU: Tesla V100-SXM2-32GB
- Full SMB backend/content check: passed

Block SMB:

- Retrained Block ViT checkpoint:
  `data/full_pipeline_20260626_1450/block_vit/block_vit.pth`
- Block ViT diagnostic:
  - accuracy: 0.9991373697916667
  - foreground_accuracy: 0.9960039206815954
  - mean_iou: 0.986213022720282
  - bottleneck: false
- Trained Block SMB policy checkpoint:
  `data/full_pipeline_20260626_1450/block_smb/policy.pth`
- Block SMB fixed-scenario evaluation:
  - mean_return: 18.532500000000002
  - success_rate: 0.25
  - threshold_pass_rate: 0.25
  - passed: `level_1_flat.json`
  - failed: `level_2_gap.json`, `level_3_stairs.json`, `level_4_platforms.json`

Full SMB:

- Verified existing asset-mock dataset:
  - train: 5,000 scenes
  - validation: 1,000 scenes
- Retrained Full SMB ViT checkpoint:
  `data/full_pipeline_20260626_1450/full_vit/full_smb_vit.pth`
- Full SMB ViT validation:
  - overall accuracy: 99.95%
  - foreground accuracy: 99.90%
  - mean IoU: 99.23%
- Real-emulator Full SMB perception diagnostic:
  - semantic_confidence: 0.9763607740402221
  - position_rmse: 0.832899759670282
  - bottleneck: true
  - bottleneck_reasons: `position_rmse`, `position_consistency`
- Transferred policy:
  `data/full_pipeline_20260626_1450/full_smb/transferred_policy.pth`
- Fine-tuned Full SMB policy:
  `data/full_pipeline_20260626_1450/full_smb/policy.pth`
- Fixed benchmark evaluation:
  - steps: 2400
  - mean_return: 0.0
  - success_rate: 0.0
  - mean_progress: 0.0
  - threshold_pass_rate: 0.0

Notes:

- Full SMB training initially failed to save a checkpoint when final evaluation
  was disabled because the trainer still attempted a final in-process
  evaluation. `retroagi/stages/full_smb/train.py` was patched so explicit
  `evaluation_episodes=0` and `evaluation_max_steps=0` produce an empty final
  evaluation instead.
- Large local debug dumps were intentionally not committed:
  `artifacts/full_pipeline_20260626_1450/full_smb/train_no_eval_rerun.log`,
  `artifacts/full_pipeline_20260626_1450/full_smb/train_no_eval_summary.json`,
  and `data/full_pipeline_20260626_1450/full_smb/policy.json`.
