"""Reproducible fresh full-volume training after the family revision 2 audit.

Run with python -m scripts.block_smb_full_volume --output-dir PATH.
--preflight runs real CUDA optimization and frozen perception on representative
long episodes; it writes diagnostics but never resumes or creates a policy run.
"""

import argparse
import json
import os
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
    train_and_evaluate_block_smb,
    train_block_smb_epoch,
)

CONFIG = Path(__file__).parent / "configs/block_smb_full_volume_revision2.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    values = json.loads(CONFIG.read_text())
    values["checkpoint_path"] = args.output_dir / "checkpoints/policy.pth"
    values["log_path"] = args.output_dir / "events.jsonl"
    config = BlockSMBTrainingConfig(**_normalize_config_values(values))
    if config.resume_path is not None or config.init_checkpoint is not None:
        raise ValueError("This recipe starts from a fresh policy.")
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=probe.learning_rate)
        samples = [
            sample_block_smb_monte_carlo_scenario(
                split="validation", seed=2, sample_index=0, family=family, difficulty="hard"
            )
            for family in ("pipe_mount", "bridge_wait", "chained_obstacles")
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
        metrics["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        (args.output_dir / "preflight.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics), flush=True)
        return
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(to_plain_data(config), indent=2) + "\n"
    )
    result = train_and_evaluate_block_smb(config, vision_factory=vision_factory)
    summary = {key: to_plain_data(value) for key, value in result.items() if key != "model"}
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
