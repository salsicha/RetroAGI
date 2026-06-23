"""Progressive-resolution promotion pipeline for architecture concepts."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from retroagi import experiments
from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    StageSpec,
    architecture_names,
    get_architecture,
    select_device,
    to_plain_data,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.full_smb.adapter import FULL_SMB_SPEC
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
        description="Transfer into Full SMB and run headless seeded observation checks.",
        status=UNSUPPORTED_RUNG_STATUS,
        reason="Full SMB promotion smoke requires emulator artifact plumbing",
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
        "interface-smoke": {"batch_size": 2},
        "synthetic-concept": {
            "epochs": 1,
            "train_samples": 16,
            "validation_samples": 8,
            "test_samples": 8,
            "controller_mse_threshold": 10.0,
        },
        "synthetic-stress": {
            "epochs": 2,
            "train_samples": 64,
            "validation_samples": 32,
            "test_samples": 32,
        },
        "block-smb-smoke": {
            "epochs": 1,
            "episodes_per_epoch": 1,
            "rollout_steps": 2,
            "evaluation_episodes": 1,
            "evaluation_max_steps": 2,
            "success_rate_threshold": 0.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "rollout_steps": 16,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 200,
        },
        "block-smb-generated-generalization": {
            "epochs": 3,
            "episodes_per_epoch": 3,
            "rollout_steps": 24,
            "generated_scenarios": 2,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 200,
        },
        "full-smb-transfer-smoke": {"steps": 32, "seeds": 1},
        "full-smb-transfer-vs-scratch": {"steps": 128, "seeds": 2},
        "full-smb-fine-tuning": {
            "epochs": 1,
            "rollout_steps": 128,
            "evaluation_episodes": 2,
        },
    },
    "medium": {
        "interface-smoke": {"batch_size": 4},
        "synthetic-concept": {
            "epochs": 3,
            "train_samples": 128,
            "validation_samples": 64,
            "test_samples": 64,
            "controller_mse_threshold": 5.0,
        },
        "synthetic-stress": {
            "epochs": 5,
            "train_samples": 512,
            "validation_samples": 128,
            "test_samples": 128,
        },
        "block-smb-smoke": {
            "epochs": 2,
            "episodes_per_epoch": 2,
            "rollout_steps": 8,
            "evaluation_episodes": 2,
            "evaluation_max_steps": 64,
            "success_rate_threshold": 0.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 5,
            "episodes_per_epoch": 4,
            "rollout_steps": 64,
            "evaluation_episodes": 5,
            "evaluation_max_steps": 200,
        },
        "block-smb-generated-generalization": {
            "epochs": 8,
            "episodes_per_epoch": 6,
            "rollout_steps": 96,
            "generated_scenarios": 8,
            "evaluation_episodes": 5,
            "evaluation_max_steps": 200,
        },
        "full-smb-transfer-smoke": {"steps": 128, "seeds": 2},
        "full-smb-transfer-vs-scratch": {"steps": 512, "seeds": 3},
        "full-smb-fine-tuning": {
            "epochs": 3,
            "rollout_steps": 512,
            "evaluation_episodes": 5,
        },
    },
    "full": {
        "interface-smoke": {"batch_size": 8},
        "synthetic-concept": {
            "epochs": 10,
            "train_samples": 1024,
            "validation_samples": 256,
            "test_samples": 256,
            "controller_mse_threshold": 2.0,
        },
        "synthetic-stress": {
            "epochs": 20,
            "train_samples": 4096,
            "validation_samples": 1024,
            "test_samples": 1024,
        },
        "block-smb-smoke": {
            "epochs": 3,
            "episodes_per_epoch": 4,
            "rollout_steps": 16,
            "evaluation_episodes": 3,
            "evaluation_max_steps": 128,
            "success_rate_threshold": 0.0,
        },
        "block-smb-fixed-scenario-training": {
            "epochs": 20,
            "episodes_per_epoch": 8,
            "rollout_steps": 128,
            "evaluation_episodes": 10,
            "evaluation_max_steps": 200,
        },
        "block-smb-generated-generalization": {
            "epochs": 30,
            "episodes_per_epoch": 12,
            "rollout_steps": 160,
            "generated_scenarios": 32,
            "evaluation_episodes": 10,
            "evaluation_max_steps": 200,
        },
        "full-smb-transfer-smoke": {"steps": 512, "seeds": 3},
        "full-smb-transfer-vs-scratch": {"steps": 2048, "seeds": 5},
        "full-smb-fine-tuning": {
            "epochs": 10,
            "rollout_steps": 2048,
            "evaluation_episodes": 10,
        },
    },
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
    parser.add_argument("--synthetic-concept-epochs", type=int, default=None)
    parser.add_argument("--synthetic-concept-train-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-validation-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-test-samples", type=int, default=None)
    parser.add_argument("--synthetic-concept-controller-mse", type=float, default=None)
    parser.add_argument("--block-smoke-epochs", type=int, default=None)
    parser.add_argument("--block-smoke-episodes-per-epoch", type=int, default=None)
    parser.add_argument("--block-smoke-rollout-steps", type=int, default=None)
    parser.add_argument("--block-smoke-evaluation-episodes", type=int, default=None)
    parser.add_argument("--block-smoke-evaluation-max-steps", type=int, default=None)
    parser.add_argument("--block-smoke-success-rate", type=float, default=None)
    return parser


def run_promotion(args: argparse.Namespace) -> dict[str, Any]:
    selected = set(args.rung or [rung.name for rung in PROMOTION_RUNGS])
    budgets = _resolve_budgets(args)
    architecture_config = dict(args.architecture_config or ())
    results = []
    for rung in PROMOTION_RUNGS:
        if rung.name not in selected:
            continue
        if rung.status == UNSUPPORTED_RUNG_STATUS:
            results.append(_skipped_rung(rung, budgets[rung.name]))
        elif rung.name == "interface-smoke":
            results.append(_run_interface_smoke(args, architecture_config, budgets[rung.name]))
        elif rung.name == "synthetic-concept":
            results.append(_run_synthetic_concept(args, architecture_config, budgets[rung.name]))
        elif rung.name == "block-smb-smoke":
            results.append(_run_block_smb_smoke(args, architecture_config, budgets[rung.name]))
        else:
            raise ValueError(f"unsupported promotion rung {rung.name!r}")

    failed = [rung for rung in results if rung["status"] == "failed"]
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": architecture_config,
        },
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


def _run_interface_smoke(
    args: argparse.Namespace,
    architecture_config: Mapping[str, Any],
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    architecture = get_architecture(args.architecture_name)
    stage_results = []
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
            _run_stage_interface_smoke(architecture, spec, architecture_config, budget, device)
        )

    runnable = [stage for stage in stage_results if stage["status"] != "skipped"]
    passed = bool(runnable) and all(stage["passed"] for stage in runnable)
    return {
        "name": "interface-smoke",
        "description": "Instantiate compatible StageSpecs and verify finite gradients.",
        "status": "passed" if passed else "failed",
        "passed": passed,
        "budget": dict(budget),
        "device": str(device),
        "stages": stage_results,
    }


def _run_stage_interface_smoke(
    architecture: Any,
    spec: StageSpec,
    architecture_config: Mapping[str, Any],
    budget: Mapping[str, int | float],
    device: torch.device,
) -> dict[str, Any]:
    try:
        model = architecture.build(spec, architecture_config).to(device)
        model.train()
        batch_size = int(budget["batch_size"])
        src_a = torch.randint(0, spec.vocab_size, (batch_size, spec.seq_len_a), device=device)
        src_b = torch.randint(0, spec.vocab_size, (batch_size, spec.seq_len_b), device=device)
        src_c = torch.randn(batch_size, spec.seq_len_c, device=device)
        outputs = model(src_a, src_b, src_c, tau=1.0)
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
    architecture_config: Mapping[str, Any],
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["synthetic-1d"],
        output=args.artifacts_dir / "synthetic_concept" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "synthetic_concept",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(architecture_config.items()),
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
    architecture_config: Mapping[str, Any],
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    experiment_args = argparse.Namespace(
        stage=["block-smb"],
        output=args.artifacts_dir / "block_smb_smoke" / "manifest.json",
        artifacts_dir=args.artifacts_dir / "block_smb_smoke",
        seed=args.seed,
        device=args.device,
        architecture_name=args.architecture_name,
        architecture_config=list(architecture_config.items()),
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


def _run_experiment_rung(
    name: str,
    experiment_args: argparse.Namespace,
    budget: Mapping[str, int | float],
) -> dict[str, Any]:
    manifest = experiments.run_experiment(experiment_args)
    output = json.dumps(to_plain_data(manifest), indent=2, sort_keys=True)
    experiment_args.output.parent.mkdir(parents=True, exist_ok=True)
    experiment_args.output.write_text(output + "\n", encoding="utf-8")
    return {
        "name": name,
        "status": "passed" if manifest["passed"] else "failed",
        "passed": bool(manifest["passed"]),
        "budget": dict(budget),
        "experiment_manifest_path": str(experiment_args.output),
        "experiment": manifest,
    }


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
