# Documented Full SMB Benchmark Run

Run name: `documented_benchmark_seed0`

This is the repository's documented Full SMB benchmark-run artifact. It is a
local replay contract, not a committed emulator-trained policy checkpoint. Full
SMB depends on local ROM-derived content and locally produced policy
checkpoints, so the repository records the exact benchmark manifest and
qualification gates while keeping ROM bytes and checkpoint bytes out of git.

- Manifest: `artifacts/full_smb/documented_benchmark_seed0/benchmark_manifest.json`
- Content metadata: `artifacts/full_smb/documented_benchmark_seed0/content.json`
- Environment check: `artifacts/full_smb/documented_benchmark_seed0/env_check.json`
- Policy checkpoint: `artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth`
- Evaluation report: `artifacts/full_smb/documented_benchmark_seed0/evaluations/evaluation.json`
- Recording manifest: `artifacts/full_smb/documented_benchmark_seed0/recordings/recording_manifest.npz`
- Play summary: `artifacts/full_smb/documented_benchmark_seed0/summaries/play_summary.json`
- Comparison report: `artifacts/full_smb/documented_benchmark_seed0/comparisons/policy_suite_comparison.json`
- Seed: `0`
- Task set: `fixed_benchmark`
- ROM bytes committed: `false`
- Checkpoint bytes committed: `false`

## Local Replay Order

Create the canonical layout:

```bash
python - <<'PY'
import json
from retroagi.stages.full_smb import full_smb_artifact_layout

layout = full_smb_artifact_layout("documented_benchmark_seed0")
layout.ensure_directories()
layout.files()["layout_manifest"].write_text(
    json.dumps(layout.to_manifest(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
```

Verify local content and backend setup:

```bash
retroagi check-env --game smb --stage full \
  --seed 0 \
  --steps 4 \
  --frame-skip 2 \
  --output artifacts/full_smb/documented_benchmark_seed0/env_check.json
```

Record local throughput:

```bash
python -m retroagi.stages.full_smb.benchmark \
  --steps 1000 \
  --warmup-steps 100 \
  --seed 0 \
  --frame-skip 2 \
  --device cpu \
  --output artifacts/full_smb/documented_benchmark_seed0/summaries/throughput_benchmark.json
```

Transfer the Block SMB policy into the Full SMB contract:

```bash
retroagi transfer --game smb --stage full \
  --block-policy-checkpoint data/block_smb/policy.pth \
  --block-vision-checkpoint data/block_vit/block_vit.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --output-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/transferred_policy.pth
```

Train or fine-tune the local policy:

```bash
retroagi train --game smb --stage full \
  --mode fine-tune \
  --seed 0 \
  --init-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/transferred_policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --perception-mode freeze \
  --task-set curriculum \
  --epochs 1 \
  --updates-per-epoch 1 \
  --rollout-steps 64 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 64 \
  --evaluation-interval-epochs 1 \
  --checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --recording-dir artifacts/full_smb/documented_benchmark_seed0/recordings \
  --recording-path artifacts/full_smb/documented_benchmark_seed0/recordings/recording_manifest.npz \
  --log-path artifacts/full_smb/documented_benchmark_seed0/logs/train.jsonl \
  --tracking-log-dir artifacts/full_smb/documented_benchmark_seed0/tracking \
  --output-summary artifacts/full_smb/documented_benchmark_seed0/summaries/train_summary.json
```

Resume once to prove checkpoint continuity:

```bash
retroagi resume --game smb --stage full \
  --seed 0 \
  --checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --save-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/resumed_policy.pth \
  --task-set curriculum \
  --epochs 2 \
  --updates-per-epoch 1 \
  --rollout-steps 64 \
  --evaluation-episodes 1 \
  --evaluation-max-steps 64 \
  --evaluation-interval-epochs 1 \
  --tracking-backend none \
  --tracking-log-dir artifacts/full_smb/documented_benchmark_seed0/tracking \
  --output-summary artifacts/full_smb/documented_benchmark_seed0/summaries/resume_summary.json
```

Evaluate on the fixed benchmark task set:

```bash
retroagi evaluate --game smb --stage full \
  --seed 0 \
  --checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --evaluation-episodes 3 \
  --evaluation-max-steps 2400 \
  --output-summary artifacts/full_smb/documented_benchmark_seed0/evaluations/evaluation.json
```

Record deterministic rollouts:

```bash
retroagi record --game smb --stage full \
  --seed 0 \
  --checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --evaluation-episodes 3 \
  --evaluation-max-steps 2400 \
  --record-dir artifacts/full_smb/documented_benchmark_seed0/recordings \
  --recording-path artifacts/full_smb/documented_benchmark_seed0/recordings/recording_manifest.npz \
  --output-summary artifacts/full_smb/documented_benchmark_seed0/summaries/recording_summary.json
```

Play locally with rendering and diagnostics:

```bash
retroagi play --game smb --stage full \
  --seed 0 \
  --checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --task-set fixed_benchmark \
  --level 1-1 \
  --steps 1000 \
  --frame-skip 4 \
  --action-repeat 2 \
  --render-mode human \
  --deterministic-policy \
  --inspection-overlay \
  --fps 30 \
  --record \
  --record-dir artifacts/full_smb/documented_benchmark_seed0/recordings \
  --record-output artifacts/full_smb/documented_benchmark_seed0/recordings/play_manifest.npz \
  --output-summary artifacts/full_smb/documented_benchmark_seed0/summaries/play_summary.json
```

Compare policy roles on identical seeded task streams:

```bash
retroagi compare --game smb --stage full \
  --transfer-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/transferred_policy.pth \
  --fine-tuned-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --known-good-checkpoint artifacts/full_smb/documented_benchmark_seed0/checkpoints/policy.pth \
  --full-smb-vision-checkpoint data/vit/full_smb_vit.pth \
  --task-set fixed_benchmark \
  --seed 0 \
  --seed 1 \
  --output artifacts/full_smb/documented_benchmark_seed0/comparisons/policy_suite_comparison.json
```

## Qualification Gates

The local run qualifies as the documented Full SMB benchmark only when:

- `env_check.json` reports successful backend/content checks.
- `summaries/throughput_benchmark.json` exists and reports positive
  `emulator_frames_per_second`.
- `checkpoints/policy.pth` loads through `retroagi evaluate`.
- `evaluations/evaluation.json` has `success_thresholds_met: true` and every
  fixed benchmark task has `threshold_met: true`.
- `summaries/recording_summary.json` points to recording manifests and episode
  artifacts under `recordings/`.
- `summaries/play_summary.json` records policy control mode, render settings,
  inspection overlay fields, and play recording output.
- `comparisons/policy_suite_comparison.json` contains named policy results and
  aggregate `action_agreement`.

This run is the Full SMB counterpart to the checked-in Block SMB known-good
artifact, but it remains a documented local benchmark until a redistributable
Full SMB checkpoint can be legally produced and stored outside the repository.
