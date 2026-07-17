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
import torch.nn.functional as F

from retroagi import experiments
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    COMPARISON_OPERATORS,
    GamePromotionGateSpec,
    PromotionMetricGateSpec,
    StageSpec,
    architecture_names,
    build_architecture_variant,
    build_game_promotion_plan,
    game_plugin_names,
    get_architecture,
    get_game_plugin,
    load_checkpoint,
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
    policy_architecture_from_checkpoint,
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
        name="full-smb-asset-mock-perception",
        description=(
            "Train the Full SMB ViT on deterministic mock scenes using the Full SMB "
            "asset vocabulary and gate policy transfer on held-out perception metrics."
        ),
        status=SUPPORTED_RUNG_STATUS,
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
            "monte_carlo_validation_samples": 2,
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
        "full-smb-asset-mock-perception": {
            "train_scenes": 8,
            "validation_scenes": 4,
            "epochs": 8,
            "semantic_accuracy_threshold": 0.05,
            "foreground_accuracy_threshold": 0.01,
            "mean_iou_threshold": 0.01,
            "position_within_tolerance_threshold": 0.0,
            "position_tolerance": 0.35,
            "runtime_seconds": 120.0,
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
            "monte_carlo_validation_samples": 4,
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
        "full-smb-asset-mock-perception": {
            "train_scenes": 16,
            "validation_scenes": 8,
            "epochs": 12,
            "semantic_accuracy_threshold": 0.10,
            "foreground_accuracy_threshold": 0.05,
            "mean_iou_threshold": 0.02,
            "position_within_tolerance_threshold": 0.0,
            "position_tolerance": 0.30,
            "runtime_seconds": 300.0,
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
            "monte_carlo_validation_samples": 8,
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
        "full-smb-asset-mock-perception": {
            "train_scenes": 64,
            "validation_scenes": 16,
            "epochs": 20,
            "semantic_accuracy_threshold": 0.25,
            "foreground_accuracy_threshold": 0.10,
            "mean_iou_threshold": 0.05,
            "position_within_tolerance_threshold": 0.10,
            "position_tolerance": 0.25,
            "runtime_seconds": 900.0,
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

RUNG_ALIASES = {rung.name: (rung.name,) for rung in PROMOTION_RUNGS}
RUNG_ALIASES.update(
    {
        "architecture-smoke": ("interface-smoke",),
        "game-synthetic": ("synthetic-concept",),
        "game-block": ("block-smb-smoke",),
        "game-full-smoke": (
            "full-smb-asset-mock-perception",
            "full-smb-transfer-smoke",
        ),
        "game-transfer": ("full-smb-transfer-smoke",),
        "game-full-comparison": ("full-smb-transfer-vs-scratch",),
        "game-full-training": ("full-smb-fine-tuning",),
        "interface": ("interface-smoke",),
        "synthetic": ("synthetic-concept",),
        "synthetic-1d": ("synthetic-concept",),
        "block": ("block-smb-smoke",),
        "block-smb": ("block-smb-smoke",),
        "full-smb": ("full-smb-transfer-smoke",),
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


def _rung_name(value: str) -> tuple[str, ...]:
    try:
        return RUNG_ALIASES[value.lower()]
    except KeyError as exc:
        choices = ", ".join(rung.name for rung in PROMOTION_RUNGS)
        raise argparse.ArgumentTypeError(
            f"unknown promotion rung {value!r}; expected one of: {choices}"
        ) from exc


def _game_name(value: str) -> str:
    name = value.lower()
    if name in game_plugin_names():
        return name
    available = ", ".join(game_plugin_names())
    raise argparse.ArgumentTypeError(f"unknown game {value!r}; available game plugins: {available}")


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
        "--game",
        default="smb",
        type=_game_name,
        help="game plugin to promote through; default: smb",
    )
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
    parser.add_argument("--block-smoke-monte-carlo-validation-samples", type=int, default=None)
    parser.add_argument("--block-smoke-success-rate", type=float, default=None)
    parser.add_argument("--block-smoke-runtime-seconds", type=float, default=None)
    return parser


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    selected = _selected_rung_names(args)
    budgets = _resolve_budgets(args)
    variant = build_architecture_variant(args.architecture_config or (), args.ablation or ())
    architecture_config = dict(variant.architecture_config)
    game_plugin = get_game_plugin(args.game)
    game_promotion_plan = build_game_promotion_plan(game_plugin)
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
        else:
            try:
                if rung.name == "interface-smoke":
                    result = _run_interface_smoke(
                        args,
                        variant,
                        budgets[rung.name],
                        game_plugin.promotion_gate(rung.name),
                    )
                elif rung.name == "synthetic-concept":
                    result = _run_synthetic_concept(
                        args,
                        variant,
                        budgets[rung.name],
                        game_plugin.promotion_gate(rung.name),
                    )
                elif rung.name == "block-smb-smoke":
                    result = _run_block_smb_smoke(
                        args,
                        variant,
                        budgets[rung.name],
                        game_plugin.promotion_gate(rung.name),
                    )
                elif rung.name == "full-smb-asset-mock-perception":
                    result = _run_full_smb_asset_mock_perception(
                        args,
                        budgets[rung.name],
                        game_plugin.promotion_gate(rung.name),
                    )
                elif rung.name == "full-smb-transfer-smoke":
                    result = _run_full_smb_transfer_smoke(
                        args,
                        variant,
                        budgets[rung.name],
                        game_plugin.promotion_gate(rung.name),
                    )
                else:
                    raise ValueError(f"unsupported promotion rung {rung.name!r}")
            except Exception as error:
                results.append(_errored_rung(rung, budgets[rung.name], error))
            else:
                results.append(result)
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
        "game": {
            "name": game_plugin.name,
            "family": game_plugin.game.family,
            "stage_ladder": [
                {
                    "name": stage.name,
                    "stage_spec_name": stage.stage_spec_name,
                    "role": stage.role,
                }
                for stage in game_plugin.game.stage_ladder
            ],
        },
        "game_promotion": game_promotion_plan.to_manifest(
            {rung["name"]: rung["status"] for rung in results},
            game_plugin.promotion_gates,
        ),
        "budget": {
            "name": args.budget,
            "rungs": budgets,
        },
        "artifacts_dir": str(args.artifacts_dir),
        "rungs": results,
        "passed": not failed,
    }


def _selected_rung_names(args: argparse.Namespace) -> set[str]:
    if args.rung is None:
        return {rung.name for rung in PROMOTION_RUNGS}
    return {rung for selection in args.rung for rung in selection}


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
        "monte_carlo_validation_samples",
        args.block_smoke_monte_carlo_validation_samples,
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


def _errored_rung(
    rung: PromotionRung, budget: Mapping[str, int | float], error: Exception
) -> dict[str, Any]:
    return {
        "name": rung.name,
        "description": rung.description,
        "status": "failed",
        "passed": False,
        "error": f"{type(error).__name__}: {error}",
        "budget": dict(budget),
    }


def _run_interface_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
    gate_spec: GamePromotionGateSpec | None,
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
    automatic_gates = _interface_automatic_gates(
        stage_results,
        budget,
        elapsed_seconds,
        gate_spec,
    )
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
    gate_spec: GamePromotionGateSpec | None,
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["synthetic-1d"],
        game=args.game,
        output=args.artifacts_dir / "synthetic_concept" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "synthetic_concept",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(variant.architecture_config.items()),
        ablation=list(variant.ablation_items),
        gate=_experiment_metric_gates_from_game_spec(
            "synthetic-1d",
            gate_spec,
            budget,
        ),
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
    return _run_experiment_rung(
        "synthetic-concept",
        experiment_args,
        budget,
        gate_spec,
    )


def _run_block_smb_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
    gate_spec: GamePromotionGateSpec | None,
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["block-smb"],
        game=args.game,
        output=args.artifacts_dir / "block_smb_smoke" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "block_smb_smoke",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(variant.architecture_config.items()),
        ablation=list(variant.ablation_items),
        gate=_experiment_metric_gates_from_game_spec(
            "block-smb",
            gate_spec,
            budget,
        ),
        synthetic_epochs=1,
        synthetic_train_samples=16,
        synthetic_validation_samples=8,
        synthetic_test_samples=8,
        block_epochs=int(budget["epochs"]),
        block_episodes_per_epoch=int(budget["episodes_per_epoch"]),
        block_rollout_steps=int(budget["rollout_steps"]),
        block_evaluation_episodes=int(budget["evaluation_episodes"]),
        block_evaluation_max_steps=int(budget["evaluation_max_steps"]),
        block_monte_carlo_validation_samples=int(budget.get("monte_carlo_validation_samples", 0)),
        block_fixed_scenario=None,
        enable_block_checkpoint_transfer=False,
    )
    return _run_experiment_rung(
        "block-smb-smoke",
        experiment_args,
        budget,
        gate_spec,
    )


def _run_full_smb_asset_mock_perception(
    args: argparse.Namespace,
    budget: Mapping[str, int | float],
    gate_spec: GamePromotionGateSpec | None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    device = select_device(args.device)
    rung_dir = args.artifacts_dir / "full_smb_asset_mock_perception"
    checkpoint_path = rung_dir / "full_smb_vit.pth"
    summary_path = rung_dir / "asset_mock_summary.json"
    rung_dir.mkdir(parents=True, exist_ok=True)

    result = _train_full_smb_asset_mock_vit(
        train_scenes=int(budget["train_scenes"]),
        validation_scenes=int(budget["validation_scenes"]),
        epochs=int(budget["epochs"]),
        seed=args.seed,
        device=device,
        checkpoint_path=checkpoint_path,
        position_tolerance=float(budget["position_tolerance"]),
    )
    elapsed_seconds = time.perf_counter() - start_time
    artifacts = {
        "full_smb_vision_checkpoint_path": str(checkpoint_path),
        "summary_path": str(summary_path),
    }
    summary = {
        "stage": "full-smb",
        "source_stage": "asset-mock",
        "seed": args.seed,
        "device": str(device),
        "budget": dict(budget),
        "metrics": result["metrics"],
        "training": result["training"],
        "mock_assets": result["mock_assets"],
        "artifacts": artifacts,
    }
    summary_path.write_text(
        json.dumps(to_plain_data(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    automatic_gates = _asset_mock_automatic_gates(
        metrics=result["metrics"],
        artifacts=artifacts,
        budget=budget,
        elapsed_seconds=elapsed_seconds,
        gate_spec=gate_spec,
    )
    passed = all(gate["passed"] for gate in automatic_gates)
    _asset_mock_gate_status_path(checkpoint_path).write_text(
        json.dumps(
            {"passed": passed, "automatic_gates": to_plain_data(automatic_gates)},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "name": "full-smb-asset-mock-perception",
        "description": (
            "Train Full SMB ViT on deterministic asset-mock scenes and gate "
            "policy transfer on held-out perception metrics."
        ),
        "status": "passed" if passed else "failed",
        "passed": passed,
        "budget": dict(budget),
        "runtime_seconds": elapsed_seconds,
        "automatic_gates": automatic_gates,
        "device": str(device),
        "metrics": result["metrics"],
        "artifacts": artifacts,
        "summary_path": str(summary_path),
    }


def _train_full_smb_asset_mock_vit(
    *,
    train_scenes: int,
    validation_scenes: int,
    epochs: int,
    seed: int,
    device: torch.device,
    checkpoint_path: Path,
    position_tolerance: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    train_images, train_labels, _train_positions, train_coverage = (
        _compose_full_smb_asset_mock_batch(train_scenes, seed=seed)
    )
    val_images, val_labels, val_positions, val_coverage = _compose_full_smb_asset_mock_batch(
        validation_scenes,
        seed=seed + 10_000,
    )
    model = FullSMBVisionTransformer(dim=16, depth=1, heads=4, drop=0.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=0.0)
    train_images = train_images.to(device)
    train_labels = train_labels.to(device)
    losses: list[float] = []
    model.train()
    weights = _asset_mock_class_weights(train_labels).to(device)
    for _epoch in range(max(1, epochs)):
        output = model(train_images).semantic_logits
        loss = F.cross_entropy(output, train_labels, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    metrics = _evaluate_full_smb_asset_mock_vit(
        model,
        val_images.to(device),
        val_labels.to(device),
        val_positions.to(device),
        position_tolerance=position_tolerance,
    )
    checkpoint = build_full_smb_vit_checkpoint(
        model,
        epoch=max(1, epochs),
        metrics=metrics,
        config={
            "model": {
                "hidden_dim": 16,
                "depth": 1,
                "heads": 4,
                "patch_size": 16,
                "dropout": 0.0,
            },
            "training": {
                "source": "promotion_full_smb_asset_mock_perception",
                "train_scenes": train_scenes,
                "validation_scenes": validation_scenes,
                "epochs": max(1, epochs),
                "learning_rate": 5e-3,
            },
            "data": {
                "semantic_classes": list(FULL_SMB_ASSET_MOCK_CLASSES),
                "mock_asset_kind": "deterministic_colored_rectangles",
                "mock_scene_size": [240, 256],
                "patch_size": 16,
            },
        },
        metadata={
            "promotion_rung": "full-smb-asset-mock-perception",
            "asset_mock_note": (
                "Fast promotion fixture using deterministic Full SMB class-colored "
                "mock assets; high-fidelity reproduction still uses scripts/vit."
            ),
        },
    )
    save_checkpoint(checkpoint_path, checkpoint)
    return {
        "metrics": metrics,
        "training": {
            "losses": losses,
            "final_loss": losses[-1] if losses else 0.0,
        },
        "mock_assets": {
            "classes": list(FULL_SMB_ASSET_MOCK_CLASSES),
            "train_class_coverage": train_coverage,
            "validation_class_coverage": val_coverage,
        },
    }


FULL_SMB_ASSET_MOCK_CLASSES = (
    "sky",
    "ground",
    "brick",
    "question_block",
    "pipe",
    "coin",
    "goomba",
    "koopa",
    "mario",
    "mushroom",
    "hill",
    "cloud",
    "bush",
)
FULL_SMB_ASSET_MOCK_COLORS = {
    "sky": (107, 140, 255),
    "ground": (141, 83, 29),
    "brick": (188, 74, 32),
    "question_block": (230, 166, 40),
    "pipe": (34, 166, 62),
    "coin": (248, 220, 69),
    "goomba": (119, 72, 40),
    "koopa": (88, 186, 77),
    "mario": (214, 34, 34),
    "mushroom": (245, 72, 72),
    "hill": (91, 184, 80),
    "cloud": (242, 242, 255),
    "bush": (45, 148, 63),
}


def _asset_mock_class_weights(labels: torch.Tensor) -> torch.Tensor:
    counts = torch.bincount(
        labels.reshape(-1),
        minlength=len(FULL_SMB_ASSET_MOCK_CLASSES),
    ).float()
    weights = 1.0 / torch.sqrt(counts + 1.0)
    return weights / weights.mean().clamp_min(1e-6)


def _compose_full_smb_asset_mock_batch(
    count: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int]]:
    rng = np.random.default_rng(seed)
    height, width, patch = 240, 256, 16
    grid_h, grid_w = height // patch, width // patch
    images = np.zeros((count, height, width, 3), dtype=np.uint8)
    labels = np.zeros((count, grid_h, grid_w), dtype=np.int64)
    positions = np.zeros((count, 2), dtype=np.float32)
    coverage = {name: 0 for name in FULL_SMB_ASSET_MOCK_CLASSES}
    for index in range(count):
        image, label, mario_position = _compose_full_smb_asset_mock_scene(rng)
        images[index] = image
        labels[index] = label
        positions[index] = mario_position
        present = np.unique(label)
        for class_id in present:
            coverage[FULL_SMB_ASSET_MOCK_CLASSES[int(class_id)]] += 1
    return (
        torch.as_tensor(images),
        torch.as_tensor(labels, dtype=torch.long),
        torch.as_tensor(positions, dtype=torch.float32),
        coverage,
    )


def _compose_full_smb_asset_mock_scene(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width, patch = 240, 256, 16
    grid_h, grid_w = height // patch, width // patch
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[...] = FULL_SMB_ASSET_MOCK_COLORS["sky"]
    label = np.zeros((grid_h, grid_w), dtype=np.int64)

    def paint(name: str, x0: int, y0: int, w: int = 1, h: int = 1) -> None:
        class_id = FULL_SMB_ASSET_MOCK_CLASSES.index(name)
        x1 = min(grid_w, max(0, x0) + w)
        y1 = min(grid_h, max(0, y0) + h)
        x0_clamped = max(0, x0)
        y0_clamped = max(0, y0)
        if x0_clamped >= x1 or y0_clamped >= y1:
            return
        label[y0_clamped:y1, x0_clamped:x1] = class_id
        image[
            y0_clamped * patch : y1 * patch,
            x0_clamped * patch : x1 * patch,
        ] = FULL_SMB_ASSET_MOCK_COLORS[name]

    for gx in range(grid_w):
        paint("ground", gx, grid_h - 2, 1, 2)
    paint("hill", int(rng.integers(0, 4)), grid_h - 5, 3, 3)
    paint("bush", int(rng.integers(8, 13)), grid_h - 3, 2, 1)
    paint("cloud", int(rng.integers(1, 10)), int(rng.integers(1, 4)), 2, 1)
    paint("pipe", int(rng.integers(9, 14)), grid_h - 4, 2, 2)
    row = int(rng.integers(4, 8))
    start = int(rng.integers(2, 6))
    for offset, name in enumerate(("brick", "question_block", "brick")):
        paint(name, start + offset, row)
    paint("coin", start + 1, max(1, row - 2))
    paint("goomba", int(rng.integers(5, 9)), grid_h - 3)
    paint("koopa", int(rng.integers(11, 15)), grid_h - 3)
    paint("mushroom", int(rng.integers(1, 6)), grid_h - 3)

    # Tiny promotion batches need balanced foreground coverage; this atlas
    # stands in for cropped full-game assets before the larger scripts/vit run.
    atlas_positions = {
        "ground": (0, 0),
        "brick": (2, 0),
        "question_block": (4, 0),
        "pipe": (6, 0),
        "coin": (8, 0),
        "goomba": (10, 0),
        "koopa": (12, 0),
        "mushroom": (14, 0),
        "hill": (0, 2),
        "cloud": (2, 2),
        "bush": (4, 2),
    }
    for name, (x0, y0) in atlas_positions.items():
        paint(name, x0, y0)
    mario_x = int(rng.integers(1, 7))
    mario_y = grid_h - 3
    paint("mario", mario_x, mario_y)
    paint("mario", 6, 2)
    return (
        image,
        label,
        np.asarray(
            [
                mario_x / max(grid_w - 1, 1),
                mario_y / max(grid_h - 1, 1),
            ],
            dtype=np.float32,
        ),
    )


@torch.no_grad()
def _evaluate_full_smb_asset_mock_vit(
    model: FullSMBVisionTransformer,
    images: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    *,
    position_tolerance: float,
) -> dict[str, float]:
    model.eval()
    output = model(images)
    predictions = output.semantic_ids
    accuracy = (predictions == labels).float().mean().item()
    foreground = labels != 0
    foreground_accuracy = (
        (predictions[foreground] == labels[foreground]).float().mean().item()
        if foreground.any()
        else 0.0
    )
    ious = []
    for class_id in range(len(FULL_SMB_ASSET_MOCK_CLASSES)):
        pred_class = predictions == class_id
        true_class = labels == class_id
        union = torch.logical_or(pred_class, true_class).sum()
        if int(union.item()) == 0:
            continue
        intersection = torch.logical_and(pred_class, true_class).sum()
        ious.append((intersection.float() / union.float()).item())
    position_error = torch.linalg.vector_norm(output.position - positions, dim=-1)
    return {
        "accuracy": float(accuracy),
        "foreground_accuracy": float(foreground_accuracy),
        "mean_iou": float(sum(ious) / len(ious)) if ious else 0.0,
        "position_rmse": float(torch.sqrt((position_error**2).mean()).item()),
        "position_within_tolerance": float(
            (position_error <= position_tolerance).float().mean().item()
        ),
        "position_tolerance": float(position_tolerance),
        "class_coverage": float(len(torch.unique(labels))),
    }


def _run_full_smb_transfer_smoke(
    args: argparse.Namespace,
    variant: Any,
    budget: Mapping[str, int | float],
    gate_spec: GamePromotionGateSpec | None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    device = select_device(args.device)
    rung_dir = args.artifacts_dir / "full_smb_transfer_smoke"
    summary_path = rung_dir / "handoff_summary.json"
    transfer_path = rung_dir / "transferred_policy.pth"
    continued_checkpoint_path = rung_dir / "continued_policy.pth"
    rung_dir.mkdir(parents=True, exist_ok=True)

    source_checkpoint_path = _ensure_handoff_source_checkpoint(args, variant, rung_dir)
    full_smb_vision_path = _ensure_asset_mock_perception_checkpoint(args)
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
        gate_spec=gate_spec,
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


def _handoff_checkpoint_matches_run(
    checkpoint_path: Path,
    args: argparse.Namespace,
    variant: Any,
) -> bool:
    """Only reuse an existing Block SMB checkpoint with matching architecture.

    A shared artifacts dir can hold a checkpoint from a different architecture
    variant; reusing it would attribute that run's weights to this variant.
    """

    try:
        checkpoint = load_checkpoint(checkpoint_path)
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
    except (ValueError, KeyError, RuntimeError, OSError):
        return False
    if architecture_name != args.architecture_name:
        return False
    variant_config = dict(getattr(variant, "architecture_config", {}) or {})
    resolved_config = dict(architecture_config or {})
    for key, value in variant_config.items():
        if key not in resolved_config or resolved_config[key] != value:
            return False
    return True


def _ensure_handoff_source_checkpoint(
    args: argparse.Namespace,
    variant: Any,
    rung_dir: Path,
) -> Path:
    previous_checkpoint = args.artifacts_dir / "block_smb_smoke" / "block_smb" / "checkpoint.pth"
    if previous_checkpoint.exists() and _handoff_checkpoint_matches_run(
        previous_checkpoint,
        args,
        variant,
    ):
        return previous_checkpoint

    block_budget = _resolve_budgets(args)["block-smb-smoke"]
    experiment_args = argparse.Namespace(
        stage=["block-smb"],
        game=args.game,
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
        block_monte_carlo_validation_samples=int(
            block_budget.get("monte_carlo_validation_samples", 0)
        ),
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


def _asset_mock_gate_status_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".gates.json")


def _asset_mock_checkpoint_gates_passed(checkpoint_path: Path) -> bool:
    status_path = _asset_mock_gate_status_path(checkpoint_path)
    if not status_path.exists():
        return False
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(status, Mapping) and bool(status.get("passed"))


def _ensure_asset_mock_perception_checkpoint(args: argparse.Namespace) -> Path:
    checkpoint_path = args.artifacts_dir / "full_smb_asset_mock_perception" / "full_smb_vit.pth"
    if checkpoint_path.exists() and _asset_mock_checkpoint_gates_passed(checkpoint_path):
        return checkpoint_path
    budget = PROMOTION_BUDGETS[args.budget]["full-smb-asset-mock-perception"]
    gate_spec = get_game_plugin(args.game).promotion_gate("full-smb-asset-mock-perception")
    rung = _run_full_smb_asset_mock_perception(args, budget, gate_spec)
    if not rung["passed"]:
        raise RuntimeError(
            "Full SMB transfer requires a passing asset-mock perception rung: "
            f"{rung['automatic_gates']}"
        )
    return checkpoint_path


def _run_deterministic_full_smb_inference(
    model: torch.nn.Module,
    vision: Any,
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    # The model forward samples Gumbel noise even in eval mode, so the reported
    # "deterministic" metrics are only reproducible if torch RNG is pinned here.
    torch.manual_seed(seed)
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
    changed_tensors = sum(1 for key in keys if not torch.equal(transferred[key], continued[key]))
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
    gate_spec: GamePromotionGateSpec | None,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    manifest = experiments.run_experiment(experiment_args)
    elapsed_seconds = time.perf_counter() - start_time
    output = json.dumps(to_plain_data(manifest), indent=2, sort_keys=True)
    experiment_args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment_args.output.write_text(output + "\n", encoding="utf-8")
    automatic_gates = _experiment_automatic_gates(
        manifest,
        budget,
        elapsed_seconds,
        gate_spec,
    )
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
    gate_spec: GamePromotionGateSpec | None,
) -> list[dict[str, Any]]:
    gates = _runtime_gates(gate_spec, budget, elapsed_seconds)
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
    manifest: Mapping[str, Any],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
    gate_spec: GamePromotionGateSpec | None,
) -> list[dict[str, Any]]:
    gates = _runtime_gates(gate_spec, budget, elapsed_seconds)
    stages = manifest.get("stages", [])
    if not isinstance(stages, Sequence):
        stages = []
    gates.extend(
        _game_metric_gates(
            gate_spec,
            _collect_stage_metrics(stages),
            budget,
        )
    )
    gates.extend(_finite_metric_gates(stages))
    gates.extend(_game_artifact_gates(gate_spec, _collect_stage_artifacts(stages)))
    return gates


def _asset_mock_automatic_gates(
    *,
    metrics: Mapping[str, float],
    artifacts: Mapping[str, str],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
    gate_spec: GamePromotionGateSpec | None,
) -> list[dict[str, Any]]:
    gates = _runtime_gates(gate_spec, budget, elapsed_seconds)
    gates.extend(_game_metric_gates(gate_spec, metrics, budget))
    gates.extend(_game_artifact_gates(gate_spec, artifacts))
    return gates


def _handoff_automatic_gates(
    *,
    metrics: Mapping[str, float],
    artifacts: Mapping[str, str],
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
    gate_spec: GamePromotionGateSpec | None,
) -> list[dict[str, Any]]:
    gates = _runtime_gates(gate_spec, budget, elapsed_seconds)
    gates.extend(_game_metric_gates(gate_spec, metrics, budget))
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
    gates.extend(_game_artifact_gates(gate_spec, artifacts))
    return gates


def _runtime_gates(
    gate_spec: GamePromotionGateSpec | None,
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
) -> list[dict[str, Any]]:
    if gate_spec is not None and gate_spec.runtime is None:
        return []
    budget_key = gate_spec.runtime.budget_key if gate_spec is not None else "runtime_seconds"
    reason = (
        gate_spec.runtime.reason if gate_spec is not None else "runtime exceeded promotion budget"
    )
    return [_runtime_gate(budget, elapsed_seconds, budget_key, reason)]


def _runtime_gate(
    budget: Mapping[str, int | float],
    elapsed_seconds: float,
    budget_key: str = "runtime_seconds",
    failure_reason: str = "runtime exceeded promotion budget",
) -> dict[str, Any]:
    if budget_key not in budget:
        return {
            "name": "runtime-seconds",
            "kind": "runtime",
            "passed": False,
            "actual": elapsed_seconds,
            "threshold": None,
            "reason": f"runtime budget key {budget_key!r} is missing",
            "source": "game-promotion",
        }
    threshold = float(budget[budget_key])
    passed = elapsed_seconds <= threshold
    return {
        "name": "runtime-seconds",
        "kind": "runtime",
        "passed": passed,
        "actual": elapsed_seconds,
        "threshold": threshold,
        "reason": None if passed else failure_reason,
        "source": "game-promotion",
    }


def _game_metric_gates(
    gate_spec: GamePromotionGateSpec | None,
    metrics: Mapping[str, Any],
    budget: Mapping[str, int | float],
) -> list[dict[str, Any]]:
    if gate_spec is None:
        return []
    gates = []
    for metric_gate in gate_spec.metric_gates:
        gates.append(_evaluate_game_metric_gate(metric_gate, metrics, budget, gate_spec))
    return gates


def _evaluate_game_metric_gate(
    metric_gate: PromotionMetricGateSpec,
    metrics: Mapping[str, Any],
    budget: Mapping[str, int | float],
    gate_spec: GamePromotionGateSpec,
) -> dict[str, Any]:
    actual = metrics.get(metric_gate.metric)
    threshold: float | str
    if metric_gate.operator == "present":
        passed = actual is not None
        threshold = "present"
        kind = "metric"
    elif metric_gate.operator == "finite":
        passed = _is_finite_number(actual)
        threshold = "finite"
        kind = "numerical"
    else:
        threshold = _metric_gate_threshold(metric_gate, budget)
        passed = (
            _is_finite_number(actual)
            and isinstance(threshold, (int, float))
            and _compare_metric(float(actual), metric_gate.operator, float(threshold))
        )
        kind = "metric"
    gate_name = metric_gate.name or f"game-metric:{metric_gate.metric}"
    return {
        "name": gate_name,
        "kind": kind,
        "passed": passed,
        "actual": actual,
        "threshold": (
            threshold
            if metric_gate.operator in {"present", "finite"}
            else f"{metric_gate.operator}{threshold}"
        ),
        "reason": None if passed else metric_gate.reason,
        "source": "game-promotion",
        "failure_reason": gate_spec.failure_reason,
    }


def _metric_gate_threshold(
    metric_gate: PromotionMetricGateSpec,
    budget: Mapping[str, int | float],
) -> float:
    if metric_gate.threshold is not None:
        return float(metric_gate.threshold)
    assert metric_gate.threshold_key is not None
    return float(budget[metric_gate.threshold_key])


def _experiment_metric_gates_from_game_spec(
    stage: str,
    gate_spec: GamePromotionGateSpec | None,
    budget: Mapping[str, int | float],
) -> list[experiments.MetricGate]:
    if gate_spec is None:
        return []
    gates = []
    for metric_gate in gate_spec.metric_gates:
        if metric_gate.operator not in COMPARISON_OPERATORS:
            continue
        gates.append(
            experiments.MetricGate(
                stage=stage,
                metric=metric_gate.metric,
                operator=metric_gate.operator,
                threshold=_metric_gate_threshold(metric_gate, budget),
            )
        )
    return gates


def _game_artifact_gates(
    gate_spec: GamePromotionGateSpec | None,
    artifacts: Mapping[str, str],
) -> list[dict[str, Any]]:
    if gate_spec is None:
        return []
    gates = []
    for artifact_gate in gate_spec.artifact_gates:
        artifact_path = artifacts.get(artifact_gate.field)
        exists = artifact_path is not None and Path(str(artifact_path)).exists()
        gates.append(
            {
                "name": artifact_gate.name or f"artifact:{artifact_gate.field}",
                "kind": "artifact",
                "passed": exists,
                "actual": str(artifact_path) if artifact_path is not None else None,
                "threshold": "exists",
                "reason": None if exists else artifact_gate.reason,
                "source": "game-promotion",
                "failure_reason": gate_spec.failure_reason,
            }
        )
    return gates


def _compare_metric(actual: float, operator: str, threshold: float) -> bool:
    if operator == ">=":
        return actual >= threshold
    if operator == "<=":
        return actual <= threshold
    if operator == ">":
        return actual > threshold
    if operator == "<":
        return actual < threshold
    if operator == "==":
        return actual == threshold
    raise ValueError(f"unknown metric gate operator {operator!r}")


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


def _collect_stage_metrics(stages: Sequence[Any]) -> dict[str, Any]:
    collected = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        metrics = stage.get("metrics", {})
        if isinstance(metrics, Mapping):
            collected.update(metrics)
    return collected


def _collect_stage_artifacts(stages: Sequence[Any]) -> dict[str, str]:
    artifacts = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        for field in ("summary_path", "checkpoint_path", "log_path"):
            artifact_path = stage.get(field)
            if artifact_path is not None:
                artifacts[field] = str(artifact_path)
    return artifacts


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
