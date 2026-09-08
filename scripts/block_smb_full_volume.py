"""Reproducible 30-epoch family-only training with optional qualified initialization.

Run with python -m scripts.block_smb_full_volume --output-dir PATH.
--preflight runs real CUDA optimization and frozen perception on representative
long episodes; it writes diagnostics but never resumes or creates a policy run.
"""

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

# CUDA reads this before its first cuBLAS operation. Set it for both the
# deterministic preflight and the detached production run.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from retroagi.core import to_plain_data
from retroagi.stages.block_smb.cli import _make_vision_factory, _normalize_config_values
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.train import (
    BlockSMBTrainingConfig,
    make_block_smb_model,
    make_block_smb_optimizer,
    restore_block_smb_checkpoint,
    train_and_evaluate_block_smb,
    train_block_smb_epoch,
)

CONFIG = Path(__file__).parent / "configs/block_smb_full_volume_revision2.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    values = json.loads(args.config.read_text())
    if args.init_checkpoint:
        values["init_checkpoint"] = args.init_checkpoint
        values["demonstration_bootstrap_updates"] = 0
    values["checkpoint_path"] = args.output_dir / "checkpoints/policy.pth"
    values["log_path"] = args.output_dir / "events.jsonl"
    config = BlockSMBTrainingConfig(**_normalize_config_values(values))
    if config.resume_path is not None:
        raise ValueError("Use a fresh optimizer and curriculum for this recipe.")
    if config.fixed_scenarios:
        raise ValueError("Full-volume training uses generated families only.")
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.preflight and config.log_path.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output_dir}")
    vision_factory, _ = _make_vision_factory(config, None)
    if args.preflight:
        torch.use_deterministic_algorithms(config.deterministic)
        torch.manual_seed(config.seed)
        device = torch.device(config.device)
        # Full-sized model and real observation pipeline, with successful
        # long demonstrations to exercise the complete computation graph.
        probe = replace(
            config, use_oracle_actions=True, update_batch_episodes=1, save_checkpoints=False
        )
        model = make_block_smb_model(probe).to(device)
        if config.init_checkpoint:
            restore_block_smb_checkpoint(
                config.init_checkpoint,
                model,
                map_location=device,
                architecture_name=config.architecture_name,
                architecture_config=config.architecture_config,
                motion_observations=config.motion_observations,
                restore_rng=False,
            )
        optimizer = make_block_smb_optimizer(model, probe)
        samples = [
            sample_block_smb_monte_carlo_scenario(
                split="validation", seed=2, sample_index=0, family=family, difficulty="hard"
            )
            for family in ("retreat_recovery", "moving_bridge", "chained_obstacles")
        ]
        metrics, _ = train_block_smb_epoch(
            model,
            optimizer,
            [(s.scenario_id, s.scenario) for s in samples],
            probe,
            epoch=0,
            device=device,
            vision_factory=vision_factory,
        )
        # Exercise actual actor sampling too: a supplied jump cannot catch
        # action-selection/likelihood mismatches in autonomous traversal.
        on_policy_samples = [
            sample_block_smb_monte_carlo_scenario(
                split="validation", seed=2, sample_index=0, family=family, difficulty="easy"
            )
            for family in ("tall_pipe_jump", "flat_run", "bridge_wait")
        ]
        on_policy_metrics, _ = train_block_smb_epoch(
            model,
            optimizer,
            [(sample.scenario_id, sample.scenario) for sample in on_policy_samples],
            replace(probe, use_oracle_actions=False),
            epoch=1,
            device=device,
            vision_factory=vision_factory,
        )
        if on_policy_metrics["oracle_action_supervised_steps"] != 0:
            raise AssertionError("On-policy preflight unexpectedly used demonstrations")
        if on_policy_metrics["loss_policy"] == 0:
            raise AssertionError("On-policy preflight did not exercise policy credit")
        metrics["on_policy"] = on_policy_metrics
        metrics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        (args.output_dir / "preflight.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
        return
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(to_plain_data(config), indent=2) + "\n"
    )
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sources": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("retroagi").rglob("*.py"))
        },
        "initial_checkpoint_sha256": (
            hashlib.sha256(config.init_checkpoint.read_bytes()).hexdigest()
            if config.init_checkpoint
            else None
        ),
    }
    (args.output_dir / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "full_volume_started", "epochs": config.epochs}), flush=True)
    result = train_and_evaluate_block_smb(config, vision_factory=vision_factory)
    summary = {key: to_plain_data(value) for key, value in result.items() if key != "model"}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
