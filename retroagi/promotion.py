"""Progressive-resolution promotion pipeline for architecture concepts."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from retroagi import experiments
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SMB_ACTIONS,
    StageSpec,
    architecture_names,
    build_architecture_variant,
    get_architecture,
    parse_architecture_ablation_item,
    save_checkpoint,
    select_device,
    to_plain_data,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.full_smb.adapter import (
    FULL_SMB_SPEC,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.train import FullSMBTrainingConfig, train_full_smb_policy
from retroagi.stages.full_smb.transfer import (
    load_transferred_full_smb_policy,
    select_transferred_full_smb_action,
    transfer_block_smb_checkpoint_to_full_smb,
)
from retroagi.stages.full_smb.vision import (
    FullSMBVisionTransformer,
    build_full_smb_vit_checkpoint,
)
from retroagi.stages.synthetic_1d.train import SYNTHETIC_1D_SPEC

SUPPORTED_RUNG_STATUS = "supported"
UNSUPPORTED_RUNG_STATUS = "unsupported"


@dataclass(frozen=True)
class PromotionRung:
    name: str
    description: str
    status: str
    reason: str | None = None


PROMOTION_RUNGS: tuple[PromotionRung, ...] = (
    PromotionRung(
        name="interface-smoke",
        description="Instantiate each compatible StageSpec and verify one finite backward pass.",
        status=SUPPORTED_RUNG_STATUS,
    ),
    PromotionRung(
        name="synthetic-concept",
        description="Run deterministic Synthetic 1D concept training with a metric gate.",
        status=SUPPORTED_RUNG_STATUS,
    ),
    PromotionRung(
        name="synthetic-stress",
        description="Raise Synthetic 1D sequence, noise, controller, and holdout difficulty.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="stress-budget presets and baseline thresholds are not defined yet",
    ),
    PromotionRung(
        name="block-smb-smoke",
        description="Run a tiny CPU-compatible Block SMB policy training smoke.",
        status=SUPPORTED_RUNG_STATUS,
    ),
    PromotionRung(
        name="block-smb-fixed-scenario-training",
        description="Train and evaluate across every fixed Block SMB scenario.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="fixed-scenario budget and success thresholds are tracked as follow-up work",
    ),
    PromotionRung(
        name="block-smb-generated-generalization",
        description="Report fixed-vs-generated Block SMB scenario generalization.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="generated-scenario promotion gates are not defined yet",
    ),
    PromotionRung(
        name="full-smb-transfer-smoke",
        description=(
            "Transfer a Block SMB policy into Full SMB, run deterministic inference, "
            "and continue direct Full SMB training."
        ),
        status=SUPPORTED_RUNG_STATUS,
    ),
    PromotionRung(
        name="full-smb-transfer-vs-scratch",
        description="Compare transferred and scratch Full SMB policies.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="Full SMB comparison thresholds are not promoted into this pipeline yet",
    ),
    PromotionRung(
        name="full-smb-fine-tuning",
        description="Continue training in the emulator and compare against transfer baselines.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="direct Full SMB fine-tuning is not implemented yet",
    ),
)

PROMOTION_BUDGETS: dict[str, dict[str, dict[str, int | float]]] = {
    "small": {
        "interface-smoke": {"batch_size": 2, "runtime_seconds": 10.0},
        "synthetic-concept": {
            "epochs": 1,
            "train_samples": 16,
            "validation_samples": 8,
            "test_samples": 8,
            "controller_mse_threshold": 10.0,
            "runtime_seconds": 60.0,
        },
        "synthetic-stress": {
            "epochs": 2,
            "train_samples": 64,
            "validation_samples": 32,
            "test_samples": 32,
            "runtime_seconds": 120.0,
        },
        "block-smb-smoke": {
            "epochs": 1,
            "episodes_per_epoch": 1,
            "rollout_steps": 2,
            "evaluation_episodes": 1,
            "evaluation_max_steps": 2,
            "success_rate_threshold": 0.0,
            "runtime_seconds": 60.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "rollout_steps": 16,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 200,
            "runtime_seconds": 300.0,
        },
        "block-smb-generated-generalization": {
            "epochs": 3,
            "episodes_per_epoch": 3,
            "rollout_steps": 24,
            "generated_scenarios": 2,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 200,
            "runtime_seconds": 600.0,
        },
        "full-smb-transfer-smoke": {"steps": 32, "seeds": 1, "runtime_seconds": 120.0},
        "full-smb-transfer-vs-scratch": {"steps": 128, "seeds": 2, "runtime_seconds": 300.0},
        "full-smb-fine-tuning": {
            "epochs": 1,
            "rollout_steps": 128,
            "evaluation_episodes": 2,
            "runtime_seconds": 600.0,
        },
    },
    "medium": {
        "interface-smoke": {"batch_size": 4, "runtime_seconds": 20.0},
        "synthetic-concept": {
            "epochs": 3,
            "train_samples": 128,
            "validation_samples": 64,
            "test_samples": 64,
            "controller_mse_threshold": 5.0,
            "runtime_seconds": 180.0,
        },
        "synthetic-stress": {
            "epochs": 5,
            "train_samples": 512,
            "validation_samples": 128,
            "test_samples": 128,
            "runtime_seconds": 600.0,
        },
        "block-smb-smoke": {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "rollout_steps": 8,
            "evaluation_episodes": 2,
            "evaluation_max_steps": 64,
            "success_rate_threshold": 0.0,
            "runtime_seconds": 180.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 5,
            "episodes_per_epoch": 4,
            "rollout_steps": 64,
            "evaluation_episodes": 5,
            "evaluation_max_steps": 200,
            "runtime_seconds": 1200.0,
        },
        "block-smb-generated-generalization": {
            "epochs": 8,
            "episodes_per_epoch": 6,
            "rollout_steps": 96,
            "generated_scenarios": 8,
            "evaluation_episodes": 5,
            "evaluation_max_steps": 200,
            "runtime_seconds": 2400.0,
        },
        "full-smb-transfer-smoke": {"steps": 128, "seeds": 2, "runtime_seconds": 300.0},
        "full-smb-transfer-vs-scratch": {"steps": 512, "seeds": 3, "runtime_seconds": 900.0},
        "full-smb-fine-tuning": {
            "epochs": 3,
            "rollout_steps": 512,
            "evaluation_episodes": 5,
            "runtime_seconds": 3600.0,
        },
    },
    "full": {
        "interface-smoke": {"batch_size": 8, "runtime_seconds": 30.0},
        "synthetic-concept": {
            "epochs": 10,
            "train_samples": 1024,
            "validation_samples": 256,
            "test_samples": 256,
            "controller_mse_threshold": 2.0,
            "runtime_seconds": 900.0,
        },
        "synthetic-stress": {
            "epochs": 20,
            "train_samples": 4096,
            "validation_samples": 1024,
            "test_samples": 1024,
            "runtime_seconds": 3600.0,
        },
        "block-smb-smoke": {
            "epochs": 3,
            "episodes_per_epoch": 4,
            "rollout_steps": 16,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 128,
            "success_rate_threshold": 0.0,
            "runtime_seconds": 300.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 20,
            "episodes_per_epoch": 8,
            "rollout_steps": 128,
            "evaluation_episodes": 10,
            "evaluation_max_steps": 200,
            "runtime_seconds": 7200.0,
        },
        "block-smb-generated-generalization": {
            "epochs": 30,
            "episodes_per_epoch": 12,
            "rollout_steps": 160,
            "generated_scenarios": 32,
            "evaluation_episodes": 10,
            "evaluation_max_steps": 200,
            "runtime_seconds": 14400.0,
        },
        "full-smb-transfer-smoke": {"steps": 512, "seeds": 3, "runtime_seconds": 900.0},
        "full-smb-transfer-vs-scratch": {"steps": 2048, "seeds": 5, "runtime_seconds": 3600.0},
        "full-smb-fine-tuning": {
            "epochs": 10,
            "rollout_steps": 2048,
            "evaluation_episodes": 10,
            "runtime_seconds": 14400.0,
        },
    },
}

REQUIRED_RUNG_METRICS = {
    "synthetic-concept": ("controller_mse",),
    "block-smb-smoke": ("eval_success_rate", "gradient_norm"),
}

RUNG_ALIASES = {rung.name: rung.name for rung in PROMOTION_RUNGS}
RUNG_ALIASES.update(
    {
        "interface": "interface-smoke",
        "synthetic": "synthetic-concept",
        "synthetic-1d": "synthetic-concept",
        "block": "block-smb-smoke",
        "block-smb": "block-smb-smoke",
        "full-smb": "full-smb-transfer-smoke",
    }
)
STAGE_SPECS = (SYNTHETIC_1D_SPEC, BLOCK_SMB_SPEC, FULL_SMB_SPEC)


def _architecture_name(value: str) -> str:
    normalized = value.strip()
    if normalized.lower() == "baseline":
        return BASELINE_ARCHITECTURE_NAME
    available = set(architecture_names())
    if normalized in available:
        return normalized
    choices = ", ".join(sorted({"baseline", *available}))
    raise argparse.ArgumentTypeError(f"unknown architecture {value!r}; expected one of: {choices}")


def _architecture_config_item(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("must use KEY=VALUE syntax")
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("architecture config key must be non-empty")
    return key, _parse_config_value(raw_value.strip())


def _parse_config_value(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _rung_name(value: str) -> str:
    try:
        return RUNG_ALIASES[value.lower()]
    except KeyError as exc:
        choices = ", ".join(rung.name for rung in PROMOTION_RUNGS)
        raise argparse.ArgumentTypeError(
            f"unknown promotion rung {value!r}; expected one of: {choices}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi promote",
        description="Run an architecture through progressive-resolution promotion checks.",
    )
    parser.add_argument(
        "--rung",
        action="append",
        type=_rung_name,
        help="promotion rung to include; repeat to run a subset",
    )
    parser.add_argument("--output", required=True, type=Path, help="promotion manifest JSON path")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/promotions/latest"),
        help="directory for rung summaries, checkpoints, and logs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--budget",
        choices=tuple(PROMOTION_BUDGETS),
        default="small",
        help="promotion budget preset to use before explicit per-rung overrides",
    )
    parser.add_argument("--interface-batch-size", type=int, default=None)
    parser.add_argument("--interface-runtime-seconds", type=float, default=None)
    parser.add_argument(
        "--architecture",
        dest="architecture_name",
        type=_architecture_name,
        default=BASELINE_ARCHITECTURE_NAME,
    )
    parser.add_argument(
        "--architecture-config",
        action="append",
        default=None,
        type=_architecture_config_item,
        metavar="KEY=VALUE",
        help="architecture-specific config override; may be repeated",
    )
    parser.add_argument(
        "--ablation",
        action="append",
        default=None,
        type=parse_architecture_ablation_item,
        metavar="KEY=VALUE",
        help="architecture-level ablation override; may be repeated",
    )
    parser.add_argument("--synthetic-concept-epochs", type=int, default=None)
    parser.add_argument("--synthetic-concept-train-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-validation-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-test-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-controller-mse", type=float, default=None)
    parser.add_argument("--synthetic-concept-runtime-seconds", type=float, default=None)
    parser.add_argument("--block-smoke-epochs", type=int, default=None)
    parser.add_argument("--block-smoke-episodes-per-epoch", type=int, default=None)
    parser.add_argument("--block-smoke-rollout-steps", type=int, default=None)
    parser.add_argument("--block-smoke-evaluation-episodes", type=int, default=None)
    parser.add_argument("--block-smoke-evaluation-max-steps", type=int, default=None)
    parser.add_argument("--block-smoke-success-rate", type=float, default=None)
    parser.add_argument("--block-smoke-runtime-seconds", type=float, default=None)
    return parser


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    selected = set(args.rung or [rung.name for rung in PROMOTION_RUNGS])
    budgets = _resolve_budgets(args)
    variant = build_architecture_variant(args.architecture_config or (), args.ablation or ())
    architecture_config = dict(variant.architecture_config)
    results = []
    stopping_reason = None
    for rung in PROMOTION_RUNGS:
        if rung.name not in selected:
            continue
        if stopping_reason is not None:
            results.append(_stopped_rung(rung, budgets[rung.name], stopping_reason))
            continue
        if rung.status == UNSUPPORTED_RUNG_STATUS:
            results.append(_skipped_rung(rung, budgets[rung.name]))
        elif rung.name == "interface-smoke":
            results.append(_run_interface_smoke(args, variant, budgets[rung.name]))
        elif rung.name == "synthetic-concept":
            results.append(_run_synthetic_concept(args, variant, budgets[rung.name]))
        elif rung.name == "block-smb-smoke":
            results.append(_run_block_smb_smoke(args, variant, budgets[rung.name]))
        elif rung.name == "full-smb-transfer-smoke":
            results.append(_run_full_smb_transfer_smoke(args, variant, budgets[rung.name]))
        else:
            raise ValueError(f"unsupported promotion rung {rung.name!r}")
        if results[-1]["status"] == "failed":
            stopping_reason = f"{rung.name} failed automatic promotion gates"

    failed = [rung for rung in results if rung["status"] == "failed"]
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": architecture_config,
        },
        "architecture_variant": variant.metadata(),
        "seed": args.seed,
        "device": args.device,
        "budget": {
            "name": args.budget,
            "rungs": budgets,
        },
        "artifacts_dir": str(args.artifacts_dir),
        "rungs": results,
        "passed": not failed,
    }


def _resolve_budgets(args: argparse.Namespace) -> dict[str, dict[str, int | float]]:
    budgets = deepcopy(PROMOTION_BUDGETS[args.budget])
    _apply_budget_override(
        budgets["interface-smoke"],
        "batch_size",
        args.interface_batch_size,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "epochs",
        args.synthetic_concept_epochs,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "train_samples",
        args.synthetic_concept_train_samples,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "validation_samples",
        args.synthetic_concept_validation_samples,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "test_samples",
        args.synthetic_concept_test_samples,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "controller_mse_threshold",
        args.synthetic_concept_controller_mse,
    )
    _apply_budget_override(
        budgets["synthetic-concept"],
        "runtime_seconds",
        args.synthetic_concept_runtime_seconds,
    )
    _apply_budget_override(
        budgets["interface-smoke"],
        "runtime_seconds",
        args.interface_runtime_seconds,
    )
    _apply_budget_override(budgets["block-smb-smoke"], "epochs", args.block_smoke_epochs)
    _apply_budget_override(
        budgets["block-smb-smoke"],
        "episodes_per_epoch",
        args.block_smoke_episodes_per_epoch,
    )
    _apply_budget_override(
        budgets["block-smb-smoke"], "rollout_steps", args.block_smoke_rollout_steps
    )
    _apply_budget_override(
        budgets["block-smb-smoke"],
        "evaluation_episodes",
        args.block_smoke_evaluation_episodes,
    )
    _apply_budget_override(
        budgets["block-smb-smoke"],
        "evaluation_max_steps",
        args.block_smoke_evaluation_max_steps,
    )
    _apply_budget_override(
        budgets["block-smb-smoke"],
        "success_rate_threshold",
        args.block_smoke_success_rate,
    )
    _apply_budget_override(
        budgets["block-smb-smoke"],
        "runtime_seconds",
        args.block_smoke_runtime_seconds,
    )
    return budgets


def _apply_budget_override(
    budget: dict[str, int | float], key: str, value: int | float | None
) -> None:
    if value is not None:
        budget[key] = value


def _skipped_rung(rung: PromotionRung, budget: Mapping[str, int | float]) -> dict[str, Any]:
    return {
        "name": rung.name,
        "description": rung.description,
        "status": "skipped",
        "passed": None,
        "reason": rung.reason,
        "budget": dict(budget),
    }


def _stopped_rung(
    rung: PromotionRung, budget: Mapping[str, int | float], reason: str
) -> dict[str, Any]:
    return {
        "name": rung.name,
        "description": rung.description,
        "status": "stopped",
        "passed": None,
        "reason": reason,
        "budget": dict(budget),
    }


def _run_interface_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    architecture = get_architecture(args.architecture_name)
    stage_results = []
    start_time = time.perf_counter()
    for spec in STAGE_SPECS:
        if not architecture.supports_stage(spec):
            stage_results.append(
                {
                    "stage": spec.name,
                    "status": "skipped",
                    "passed": None,
                    "reason": "architecture does not declare support for this StageSpec",
                }
            )
            continue
        stage_results.append(
            _run_stage_interface_smoke(architecture, spec, variant, budget, device)
        )

    runnable = [stage for stage in stage_results if stage["status"] != "skipped"]
    elapsed_seconds = time.perf_counter() - start_time
    automatic_gates = _interface_automatic_gates(stage_results, budget, elapsed_seconds)
    passed = (
        bool(runnable)
        and all(stage["passed"] for stage in runnable)
        and all(gate["passed"] for gate in automatic_gates)
    )
    return {
        "name": "interface-smoke",
        "description": "Instantiate compatible StageSpecs and verify finite gradients.",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "budget": dict(budget),
        "runtime_seconds": elapsed_seconds,
        "automatic_gates": automatic_gates,
        "device": str(device),
        "stages": stage_results,
    }


def _run_stage_interface_smoke(
    architecture: Any,
    spec: StageSpec,
    variant: Any,
    budget: Mapping[str, int | float],
    device: torch.device,
) -> dict[str, Any]:
    try:
        model = architecture.build(spec, variant.architecture_config).to(device)
        model.train()
        batch_size = int(budget["batch_size"])
        src_a = torch.randint(0, spec.vocab_size, (batch_size, spec.seq_len_a), device=device)
        src_b = torch.randint(0, spec.vocab_size, (batch_size, spec.seq_len_b), device=device)
        src_c = torch.randn(batch_size, spec.seq_len_c, device=device)
        outputs = model(src_a, src_b, src_c, tau=1.0, **variant.forward_kwargs)
        tensors = [
            output for output in outputs if torch.is_tensor(output) and output.is_floating_point()
        ]
        loss = sum(tensor.float().pow(2).mean() for tensor in tensors)
        loss_value = float(loss.detach().cpu())
        if not torch.isfinite(loss).item():
            raise ValueError("forward pass produced a non-finite loss")
        loss.backward()
        gradients = [
            parameter.grad for parameter in model.parameters() if parameter.grad is not None
        ]
        finite_gradients = bool(gradients) and all(
            bool(torch.isfinite(gradient).all().item()) for gradient in gradients
        )
        if not finite_gradients:
            raise ValueError("backward pass produced missing or non-finite gradients")
        return {
            "stage": spec.name,
            "status": "passed",
            "passed": True,
            "loss": loss_value,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "output_tensors": len(tensors),
        }
    except Exception as exc:  # pragma: no cover - exercised through manifest assertions.
        return {
            "stage": spec.name,
            "status": "failed",
            "passed": False,
            "reason": str(exc),
        }


def _run_synthetic_concept(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["synthetic-1d"],
        output=args.artifacts_dir / "synthetic_concept" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "synthetic_concept",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(variant.architecture_config.items()),
        ablation=list(variant.ablation_items),
        gate=[
            experiments.MetricGate(
                stage="synthetic-1d",
                metric="controller_mse",
                operator="<=",
                threshold=float(budget["controller_mse_threshold"]),
            )
        ],
        synthetic_epochs=int(budget["epochs"]),
        synthetic_train_samples=int(budget["train_samples"]),
        synthetic_validation_samples=int(budget["validation_samples"]),
        synthetic_test_samples=int(budget["test_samples"]),
        block_epochs=1,
        block_episodes_per_epoch=1,
        block_rollout_steps=2,
        block_evaluation_episodes=1,
        block_evaluation_max_steps=2,
        block_fixed_scenario=None,
        enable_block_checkpoint_transfer=False,
    )
    return _run_experiment_rung("synthetic-concept", experiment_args, budget)


def _run_block_smb_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["block-smb"],
        output=args.artifacts_dir / "block_smb_smoke" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "block_smb_smoke",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(variant.architecture_config.items()),
        ablation=list(variant.ablation_items),
        gate=[
            experiments.MetricGate(
                stage="block-smb",
                metric="eval_success_rate",
                operator=">=",
                threshold=float(budget["success_rate_threshold"]),
            )
        ],
        synthetic_epochs=1,
        synthetic_train_samples=16,
        synthetic_validation_samples=8,
        synthetic_test_samples=8,
        block_epochs=int(budget["epochs"]),
        block_episodes_per_epoch=int(budget["episodes_per_epoch"]),
        block_rollout_steps=int(budget["rollout_steps"]),
        block_evaluation_episodes=int(budget["evaluation_episodes"]),
        block_evaluation_max_steps=int(budget["evaluation_max_steps"]),
        block_fixed_scenario=None,
        enable_block_checkpoint_transfer=False,
    )
    return _run_experiment_rung("block-smb-smoke", experiment_args, budget)


def _run_full_smb_transfer_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    start_time = time.perf_counter()
    device = select_device(args.device)
    rung_dir = args.artifacts_dir / "full_smb_transfer_smoke"
    summary_path = rung_dir / "handoff_summary.json"
    full_smb_vision_path = rung_dir / "full_smb_vit.pth"
    transfer_path = rung_dir / "transferred_policy.pth"
    continued_checkpoint_path = rung_dir / "continued_policy.pth"
    rung_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint_path = _ensure_handoff_source_checkpoint(args, variant, rung_dir)
    _write_promotion_full_smb_vit_checkpoint(full_smb_vision_path)
    transfer_result = transfer_block_smb_checkpoint_to_full_smb(
        source_checkpoint_path,
        output_checkpoint=transfer_path,
        full_smb_vision_checkpoint=full_smb_vision_path,
        block_vision_checkpoint=None,
        device=device,
    )
    transfer_controller_metrics = _controller_transfer_metrics(
        transfer_result.source_checkpoint["states"]["model"],
        transfer_result.checkpoint["states"]["model"],
    )
    loaded = load_transferred_full_smb_policy(
        transfer_path,
        full_smb_vision_checkpoint=full_smb_vision_path,
        device=device,
    )
    inference = _run_deterministic_full_smb_inference(
        loaded.model,
        loaded.vision,
        seed=args.seed,
        device=device,
    )
    training_result = train_full_smb_policy(
        FullSMBTrainingConfig(
            seed=args.seed,
            epochs=1,
            episodes_per_epoch=max(1, int(budget["seeds"])),
            max_steps_per_episode=max(1, int(budget["steps"])),
            evaluation_episodes=max(1, int(budget["seeds"])),
            evaluation_max_steps=max(1, int(budget["steps"])),
            device=str(device),
            init_checkpoint=transfer_path,
            full_smb_vision_checkpoint=full_smb_vision_path,
            checkpoint_path=continued_checkpoint_path,
            save_checkpoints=True,
        ),
        make_stage=_make_promotion_full_smb_stage,
    )
    continued_controller_metrics = _controller_adaptation_metrics(
        transfer_result.checkpoint["states"]["model"],
        training_result.checkpoint["states"]["model"],
    )
    elapsed_seconds = time.perf_counter() - start_time
    metrics = {
        "deterministic_action": float(inference["action"]),
        "deterministic_entropy": float(inference["entropy"]),
        "deterministic_margin": float(inference["margin"]),
        **{
            f"controller_inference_{key}": float(value)
            for key, value in inference["controller"].items()
        },
        "continued_global_step": float(training_result.checkpoint["global_step"]),
        "continued_mean_train_return": float(
            training_result.checkpoint["metrics"]["mean_train_return"]
        ),
        "continued_evaluation_mean_return": float(
            training_result.checkpoint["metrics"]["evaluation_mean_return"]
        ),
        "continued_evaluation_success_rate": float(
            training_result.checkpoint["metrics"]["evaluation_success_rate"]
        ),
        **transfer_controller_metrics,
        **continued_controller_metrics,
    }
    artifacts = {
        "source_block_checkpoint_path": str(source_checkpoint_path),
        "full_smb_vision_checkpoint_path": str(full_smb_vision_path),
        "transfer_checkpoint_path": str(transfer_path),
        "continued_checkpoint_path": str(continued_checkpoint_path),
        "summary_path": str(summary_path),
    }
    summary = {
        "stage": "full-smb",
        "source_stage": "block-smb",
        "seed": args.seed,
        "device": str(device),
        "architecture": {
            "name": transfer_result.checkpoint["architecture"]["name"],
            "config": transfer_result.checkpoint["architecture"]["config"],
        },
        "budget": dict(budget),
        "metrics": metrics,
        "inference": inference,
        "controller_transfer": {
            "state_prefixes": list(_CONTROLLER_STATE_PREFIXES),
            "transfer_metrics": transfer_controller_metrics,
            "adaptation_metrics": continued_controller_metrics,
        },
        "training": training_result.as_dict(),
        "artifacts": artifacts,
    }
    summary_path.write_text(
        json.dumps(to_plain_data(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    automatic_gates = _handoff_automatic_gates(
        metrics=metrics,
        artifacts=artifacts,
        budget=budget,
        elapsed_seconds=elapsed_seconds,
    )
    passed = all(gate["passed"] for gate in automatic_gates)
    return {
        "name": "full-smb-transfer-smoke",
        "description": (
            "Transfer Block SMB policy weights into Full SMB, run deterministic "
            "inference, and continue direct Full SMB training."
        ),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "budget": dict(budget),
        "runtime_seconds": elapsed_seconds,
        "automatic_gates": automatic_gates,
        "device": str(device),
        "metrics": metrics,
        "artifacts": artifacts,
        "summary_path": str(summary_path),
    }


def _ensure_handoff_source_checkpoint(
    args: argparse.Namespace,
    variant: Any,
    rung_dir: Path,
) -> Path:
    previous_checkpoint = args.artifacts_dir / "block_smb_smoke" / "checkpoint.pth"
    if previous_checkpoint.exists():
        return previous_checkpoint

    block_budget = PROMOTION_BUDGETS[args.budget]["block-smb-smoke"]
    experiment_args = argparse.Namespace(
        stage=["block-smb"],
        output=rung_dir / "source_block_smb" / "manifest.json",
        artifacts_dir=rung_dir / "source_block_smb",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(variant.architecture_config.items()),
        ablation=list(variant.ablation_items),
        gate=[],
        synthetic_epochs=1,
        synthetic_train_samples=16,
        synthetic_validation_samples=8,
        synthetic_test_samples=8,
        block_epochs=int(block_budget["epochs"]),
        block_episodes_per_epoch=int(block_budget["episodes_per_epoch"]),
        block_rollout_steps=int(block_budget["rollout_steps"]),
        block_evaluation_episodes=int(block_budget["evaluation_episodes"]),
        block_evaluation_max_steps=int(block_budget["evaluation_max_steps"]),
        block_fixed_scenario=None,
        enable_block_checkpoint_transfer=False,
    )
    with redirect_stdout(io.StringIO()):
        manifest = experiments.run_experiment(experiment_args)
    experiment_args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment_args.output.write_text(
        json.dumps(to_plain_data(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_checkpoint = experiment_args.artifacts_dir / "block_smb" / "checkpoint.pth"
    if not source_checkpoint.exists():
        raise FileNotFoundError(
            "Full SMB handoff source Block SMB checkpoint was not created at "
            f"{source_checkpoint}"
        )
    return source_checkpoint


def _write_promotion_full_smb_vit_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model = FullSMBVisionTransformer(dim=16, depth=1, heads=4, drop=0.0)
    checkpoint = build_full_smb_vit_checkpoint(
        model,
        epoch=1,
        metrics={"mean_iou": 1.0, "pixel_accuracy": 1.0},
        config={
            "model": {
                "hidden_dim": 16,
                "depth": 1,
                "heads": 4,
                "patch_size": 16,
                "dropout": 0.0,
            },
            "source": "promotion_full_smb_transfer_smoke",
        },
    )
    save_checkpoint(path, checkpoint)


def _run_deterministic_full_smb_inference(
    model: torch.nn.Module,
    vision: Any,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    stage = _make_promotion_full_smb_stage(vision)
    try:
        observation = stage.reset(seed=seed)
        batch = stage.encode_observation(observation)
        selection = select_transferred_full_smb_action(
            model,
            batch,
            deterministic=True,
            device=device,
        )
    finally:
        stage.close()
    logits = selection.logits.float()
    probabilities = torch.softmax(logits, dim=-1)
    entropy = torch.distributions.Categorical(probs=probabilities).entropy()
    top_values = torch.topk(logits, k=min(2, logits.shape[-1]), dim=-1).values
    margin = (
        float((top_values[:, 0] - top_values[:, 1]).mean().item())
        if top_values.shape[-1] > 1
        else 0.0
    )
    return {
        "action": selection.action,
        "action_name": selection.action_name,
        "entropy": float(entropy.mean().item()),
        "margin": margin,
        "logits_shape": list(selection.logits.shape),
        "controller": _controller_output_metrics(model, batch, device=device),
    }


_CONTROLLER_STATE_PREFIXES = ("agent.transformer_B.", "agent.fc_controller_params.")


def _controller_state_items(
    state: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().float()
        for key, value in state.items()
        if key.startswith(_CONTROLLER_STATE_PREFIXES) and torch.is_tensor(value)
    }


def _controller_transfer_metrics(
    source_state: Mapping[str, Any],
    transferred_state: Mapping[str, Any],
) -> dict[str, float]:
    source = _controller_state_items(source_state)
    transferred = _controller_state_items(transferred_state)
    keys = sorted(set(source) & set(transferred))
    if not keys:
        return {
            "controller_transfer_key_count": 0.0,
            "controller_transfer_max_abs_delta": math.inf,
            "controller_transfer_mean_abs_delta": math.inf,
        }
    deltas = [(source[key] - transferred[key]).abs().reshape(-1) for key in keys]
    concatenated = torch.cat(deltas)
    return {
        "controller_transfer_key_count": float(len(keys)),
        "controller_transfer_max_abs_delta": float(concatenated.max().item()),
        "controller_transfer_mean_abs_delta": float(concatenated.mean().item()),
    }


def _controller_adaptation_metrics(
    transferred_state: Mapping[str, Any],
    continued_state: Mapping[str, Any],
) -> dict[str, float]:
    transferred = _controller_state_items(transferred_state)
    continued = _controller_state_items(continued_state)
    keys = sorted(set(transferred) & set(continued))
    if not keys:
        return {
            "controller_adaptation_key_count": 0.0,
            "controller_adaptation_changed_tensors": 0.0,
            "controller_adaptation_max_abs_delta": math.inf,
            "controller_adaptation_mean_abs_delta": math.inf,
        }
    deltas = [(transferred[key] - continued[key]).abs().reshape(-1) for key in keys]
    concatenated = torch.cat(deltas)
    changed_tensors = sum(
        1 for key in keys if not torch.equal(transferred[key], continued[key])
    )
    return {
        "controller_adaptation_key_count": float(len(keys)),
        "controller_adaptation_changed_tensors": float(changed_tensors),
        "controller_adaptation_max_abs_delta": float(concatenated.max().item()),
        "controller_adaptation_mean_abs_delta": float(concatenated.mean().item()),
    }


@torch.no_grad()
def _controller_output_metrics(
    model: torch.nn.Module,
    batch: Any,
    *,
    device: torch.device,
) -> dict[str, float]:
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    outputs = model(src_a, src_b, src_c, tau=1.0, episode_mask=episode_mask)
    w = outputs[5].detach().cpu().float()
    b = outputs[6].detach().cpu().float()
    return {
        "w_mean": float(w.mean().item()),
        "w_std": float(w.std(unbiased=False).item()),
        "b_mean": float(b.mean().item()),
        "b_std": float(b.std(unbiased=False).item()),
    }


def _make_promotion_full_smb_stage(vision: Any) -> FullSMBStage:
    return FullSMBStage(
        env=_PromotionTinyFullSMBEnv(),
        vision=vision,
        observation_config=FullSMBObservationConfig(
            frame_skip=1,
            frame_stack=2,
            resize_shape=(16, 20),
        ),
    )


class _PromotionTinyFullSMBEnv:
    buttons = ("B", "A", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT")

    def __init__(self) -> None:
        self.seed = 0
        self.step_count = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.seed = 0 if seed is None else int(seed)
        self.step_count = 0
        return self._observation(0), {
            "x_pos": self.seed,
            "y_pos": 96,
            "score": 0,
            "coins": 0,
            "lives": 3,
        }

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        button_vector = np.asarray(action, dtype=np.int8)
        action_value = int(np.flatnonzero(button_vector).sum())
        self.step_count += 1
        terminated = self.step_count >= 8
        return (
            self._observation(action_value),
            float(action_value + self.step_count),
            terminated,
            False,
            {
                "x_pos": self.seed + self.step_count,
                "y_pos": 96,
                "score": self.step_count * 10,
                "coins": self.step_count % 3,
                "lives": 3,
                "level_complete": terminated,
            },
        )

    def close(self) -> None:
        pass

    def _observation(self, action_value: int) -> np.ndarray:
        base = np.arange(16 * 20, dtype=np.uint16).reshape(16, 20)
        return np.stack(
            (
                (base + self.seed + self.step_count + action_value) % 256,
                (base * 2 + self.seed + self.step_count) % 256,
                (base * 3 + self.seed + action_value) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)


def _run_experiment_rung(
    name: str,
    experiment_args: argparse.Namespace,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    start_time = time.perf_counter()
    manifest = experiments.run_experiment(experiment_args)
    elapsed_seconds = time.perf_counter() - start_time
    output = json.dumps(to_plain_data(manifest), indent=2, sort_keys=True)
    experiment_args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment_args.output.write_text(output + "\n", encoding="utf-8")
    automatic_gates = _experiment_automatic_gates(name, manifest, budget, elapsed_seconds)
    passed = bool(manifest["passed"]) and all(gate["passed"] for gate in automatic_gates)
    return {
        "name": name,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "budget": dict(budget),
        "runtime_seconds": elapsed_seconds,
        "automatic_gates": automatic_gates,
        "experiment_manifest_path": str(experiment_args.output),
        "experiment": manifest,
    }


def _interface_automatic_gates(
    stage_results: Sequence[Mapping[str, Any]],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    gates = [_runtime_gate(budget, elapsed_seconds)]
    for stage in stage_results:
        if stage.get("status") == "skipped":
            continue
        gates.append(
            {
                "name": f"{stage['stage']}:finite-loss",
                "kind": "numerical",
                "passed": _is_finite_number(stage.get("loss")),
                "actual": stage.get("loss"),
                "threshold": "finite",
                "reason": (
                    None
                    if _is_finite_number(stage.get("loss"))
                    else "loss is missing or non-finite"
                ),
            }
        )
        gates.append(
            {
                "name": f"{stage['stage']}:finite-gradients",
                "kind": "numerical",
                "passed": bool(stage.get("passed")),
                "actual": stage.get("passed"),
                "threshold": True,
                "reason": None if stage.get("passed") else stage.get("reason", "stage failed"),
            }
        )
    return gates


def _experiment_automatic_gates(
    name: str,
    manifest: Mapping[str, Any],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    gates = [_runtime_gate(budget, elapsed_seconds)]
    stages = manifest.get("stages", [])
    if not isinstance(stages, Sequence):
        stages = []
    gates.extend(_required_metric_gates(name, stages))
    gates.extend(_finite_metric_gates(stages))
    gates.extend(_artifact_gates(stages))
    return gates


def _handoff_automatic_gates(
    *,
    metrics: Mapping[str, float],
    artifacts: Mapping[str, str],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    gates = [_runtime_gate(budget, elapsed_seconds)]
    gates.append(
        {
            "name": "deterministic-action-in-vocabulary",
            "kind": "contract",
            "passed": 0 <= int(metrics["deterministic_action"]) < len(SMB_ACTIONS),
            "actual": int(metrics["deterministic_action"]),
            "threshold": f"0..{len(SMB_ACTIONS) - 1}",
            "reason": (
                None
                if 0 <= int(metrics["deterministic_action"]) < len(SMB_ACTIONS)
                else "selected action is outside the shared SMB action vocabulary"
            ),
        }
    )
    gates.append(
        {
            "name": "continued-training-advanced",
            "kind": "metric",
            "passed": metrics["continued_global_step"] > 0,
            "actual": metrics["continued_global_step"],
            "threshold": ">0",
            "reason": (
                None
                if metrics["continued_global_step"] > 0
                else "continued Full SMB training did not advance global_step"
            ),
        }
    )
    gates.append(
        {
            "name": "controller-transfer-exact",
            "kind": "contract",
            "passed": (
                metrics["controller_transfer_key_count"] > 0
                and metrics["controller_transfer_max_abs_delta"] == 0.0
            ),
            "actual": {
                "key_count": metrics["controller_transfer_key_count"],
                "max_abs_delta": metrics["controller_transfer_max_abs_delta"],
            },
            "threshold": "key_count>0 and max_abs_delta==0",
            "reason": (
                None
                if (
                    metrics["controller_transfer_key_count"] > 0
                    and metrics["controller_transfer_max_abs_delta"] == 0.0
                )
                else "Block SMB controller weights were not copied exactly into Full SMB"
            ),
        }
    )
    gates.append(
        {
            "name": "controller-adapted-during-full-smb-training",
            "kind": "metric",
            "passed": (
                metrics["controller_adaptation_key_count"] > 0
                and metrics["controller_adaptation_changed_tensors"] > 0
            ),
            "actual": {
                "key_count": metrics["controller_adaptation_key_count"],
                "changed_tensors": metrics["controller_adaptation_changed_tensors"],
                "max_abs_delta": metrics["controller_adaptation_max_abs_delta"],
            },
            "threshold": "key_count>0 and changed_tensors>0",
            "reason": (
                None
                if (
                    metrics["controller_adaptation_key_count"] > 0
                    and metrics["controller_adaptation_changed_tensors"] > 0
                )
                else "continued Full SMB training did not update controller weights"
            ),
        }
    )
    for metric, value in metrics.items():
        gates.append(
            {
                "name": f"metric-finite:{metric}",
                "kind": "numerical",
                "passed": _is_finite_number(value),
                "actual": value,
                "threshold": "finite",
                "reason": None if _is_finite_number(value) else "metric is non-finite",
            }
        )
    for field, artifact_path in artifacts.items():
        exists = Path(artifact_path).exists()
        gates.append(
            {
                "name": f"artifact:{field}",
                "kind": "artifact",
                "passed": exists,
                "actual": artifact_path,
                "threshold": "exists",
                "reason": None if exists else "artifact path does not exist",
            }
        )
    return gates


def _runtime_gate(
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    threshold = float(budget["runtime_seconds"])
    passed = elapsed_seconds <= threshold
    return {
        "name": "runtime-seconds",
        "kind": "runtime",
        "passed": passed,
        "actual": elapsed_seconds,
        "threshold": threshold,
        "reason": None if passed else "runtime exceeded promotion budget",
    }


def _required_metric_gates(name: str, stages: Sequence[Any]) -> list[dict[str, Any]]:
    required_metrics = REQUIRED_RUNG_METRICS.get(name, ())
    gates = []
    for metric in required_metrics:
        stage_metric = _find_metric(stages, metric)
        gates.append(
            {
                "name": f"required-metric:{metric}",
                "kind": "metric",
                "passed": stage_metric is not None,
                "actual": stage_metric,
                "threshold": "present",
                "reason": None if stage_metric is not None else "required metric is missing",
            }
        )
    return gates


def _finite_metric_gates(stages: Sequence[Any]) -> list[dict[str, Any]]:
    gates = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        stage_name = stage.get("stage", "unknown-stage")
        metrics = stage.get("metrics", {})
        if not isinstance(metrics, Mapping):
            gates.append(
                {
                    "name": f"{stage_name}:metrics-finite",
                    "kind": "numerical",
                    "passed": False,
                    "actual": None,
                    "threshold": "finite numeric metrics",
                    "reason": "metrics payload is not a mapping",
                }
            )
            continue
        non_finite = [
            key
            for key, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(value)
        ]
        gates.append(
            {
                "name": f"{stage_name}:metrics-finite",
                "kind": "numerical",
                "passed": not non_finite,
                "actual": non_finite,
                "threshold": "finite numeric metrics",
                "reason": None if not non_finite else "one or more numeric metrics are non-finite",
            }
        )
    return gates


def _artifact_gates(stages: Sequence[Any]) -> list[dict[str, Any]]:
    gates = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        stage_name = stage.get("stage", "unknown-stage")
        for field in ("summary_path", "checkpoint_path", "log_path"):
            artifact_path = stage.get(field)
            if artifact_path is None:
                continue
            exists = Path(str(artifact_path)).exists()
            gates.append(
                {
                    "name": f"{stage_name}:artifact:{field}",
                    "kind": "artifact",
                    "passed": exists,
                    "actual": str(artifact_path),
                    "threshold": "exists",
                    "reason": None if exists else "artifact path does not exist",
                }
            )
    return gates


def _find_metric(stages: Sequence[Any], metric: str) -> float | None:
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        metrics = stage.get("metrics", {})
        if not isinstance(metrics, Mapping):
            continue
        value = metrics.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = run_promotion(args)
    output = json.dumps(to_plain_data(manifest), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
