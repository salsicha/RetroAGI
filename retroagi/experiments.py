"""Stage-agnostic experiment runner for architecture comparisons."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    architecture_names,
    build_architecture_variant,
    build_game_promotion_plan,
    game_plugin_names,
    get_game_plugin,
    parse_architecture_ablation_item,
    to_plain_data,
)

STAGE_ALIASES = {
    "synthetic-1d": "synthetic-1d",
    "synthetic_1d": "synthetic-1d",
    "synthetic": "synthetic-1d",
    "block-smb": "block-smb",
    "block_smb": "block-smb",
    "block": "block-smb",
}
SUPPORTED_EXPERIMENT_STAGES = tuple(sorted({"synthetic-1d", "block-smb"}))
EXPERIMENT_STAGE_GAME_STAGE = {
    "synthetic-1d": "synthetic",
    "block-smb": "block",
}
EXPERIMENT_STAGE_PROMOTION_RUNGS = {
    "synthetic-1d": ("synthetic-concept",),
    "block-smb": ("block-smb-smoke",),
}
GATE_PATTERN = re.compile(
    r"^(?:(?P<stage>[A-Za-z0-9_-]+):)?"
    r"(?P<metric>[A-Za-z0-9_./-]+)"
    r"(?P<operator>>=|<=|>|<|==)"
    r"(?P<threshold>-?(?:\d+(?:\.\d*)?|\.\d+))$"
)


@dataclass(frozen=True)
class MetricGate:
    metric: str
    operator: str
    threshold: float
    stage: str | None = None

    def applies_to(self, stage: str) -> bool:
        return self.stage is None or self.stage == stage

    def evaluate(self, metrics: Mapping[str, Any]) -> dict[str, Any]:
        actual = metrics.get(self.metric)
        if not isinstance(actual, (int, float)) or isinstance(actual, bool):
            return {
                "metric": self.metric,
                "operator": self.operator,
                "threshold": self.threshold,
                "actual": actual,
                "passed": False,
                "reason": "metric missing or non-numeric",
            }
        actual_value = float(actual)
        passed = _compare_metric(actual_value, self.operator, self.threshold)
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "actual": actual_value,
            "passed": passed,
            "reason": None if passed else "metric gate failed",
        }


@dataclass(frozen=True)
class StageRunPlan:
    stage: str
    command: list[str]
    stage_args: list[str]
    summary_path: Path
    checkpoint_path: Path
    log_path: Path | None = None


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


def _stage_name(value: str) -> str:
    try:
        return STAGE_ALIASES[value.lower()]
    except KeyError as exc:
        choices = ", ".join(SUPPORTED_EXPERIMENT_STAGES)
        raise argparse.ArgumentTypeError(
            f"unknown experiment stage {value!r}; expected one of: {choices}"
        ) from exc


def _architecture_name(value: str) -> str:
    normalized = value.strip()
    if normalized.lower() == "baseline":
        return BASELINE_ARCHITECTURE_NAME
    available = set(architecture_names())
    if normalized in available:
        return normalized
    choices = ", ".join(sorted({"baseline", *available}))
    raise argparse.ArgumentTypeError(f"unknown architecture {value!r}; expected one of: {choices}")


def _game_name(value: str) -> str:
    name = value.lower()
    if name in game_plugin_names():
        return name
    available = ", ".join(game_plugin_names())
    raise argparse.ArgumentTypeError(f"unknown game {value!r}; available game plugins: {available}")


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


def _metric_gate(value: str) -> MetricGate:
    match = GATE_PATTERN.match(value.strip())
    if match is None:
        raise argparse.ArgumentTypeError(
            "gate must use [STAGE:]METRIC>=VALUE syntax, with operators >= <= > < =="
        )
    stage = match.group("stage")
    return MetricGate(
        stage=_stage_name(stage) if stage is not None else None,
        metric=match.group("metric"),
        operator=match.group("operator"),
        threshold=float(match.group("threshold")),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi experiment",
        description="Run one architecture through selected stages and write a combined manifest.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        required=True,
        type=_stage_name,
        help="stage to run; repeat for multiple stages",
    )
    parser.add_argument("--output", required=True, type=Path, help="combined manifest JSON path")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/experiments/latest"),
        help="directory for per-stage summaries, checkpoints, and logs",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--game",
        default="smb",
        type=_game_name,
        help="game profile to attach to the experiment manifest; default: smb",
    )
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
    parser.add_argument(
        "--gate",
        action="append",
        default=None,
        type=_metric_gate,
        metavar="[STAGE:]METRIC>=VALUE",
        help="optional metric gate; may be repeated",
    )
    parser.add_argument("--synthetic-epochs", type=int, default=1)
    parser.add_argument("--synthetic-train-samples", type=int, default=16)
    parser.add_argument("--synthetic-validation-samples", type=int, default=8)
    parser.add_argument("--synthetic-test-samples", type=int, default=8)
    parser.add_argument("--block-epochs", type=int, default=1)
    parser.add_argument("--block-episodes-per-epoch", type=int, default=1)
    parser.add_argument("--block-rollout-steps", type=int, default=2)
    parser.add_argument("--block-evaluation-episodes", type=int, default=1)
    parser.add_argument("--block-evaluation-max-steps", type=int, default=2)
    parser.add_argument("--block-fixed-scenario", action="append", default=None)
    parser.add_argument(
        "--enable-block-checkpoint-transfer",
        action="store_true",
        help="load the Block ViT checkpoint instead of using an untrained frozen vision encoder",
    )
    return parser


def build_experiment_plans(args: argparse.Namespace) -> list[StageRunPlan]:
    variant = build_architecture_variant(args.architecture_config or (), args.ablation or ())
    architecture_config = dict(variant.architecture_config)
    plugin = _experiment_game_plugin(args)
    plans = []
    for stage in args.stage:
        _resolve_experiment_game_stage(plugin, stage)
        if stage == "synthetic-1d":
            plans.append(_synthetic_plan(args, architecture_config, variant.args_for_stage(stage)))
        elif stage == "block-smb":
            plans.append(
                _block_smb_plan(
                    args,
                    architecture_config,
                    variant.args_for_stage(stage),
                    disable_checkpoint_transfer=(
                        not args.enable_block_checkpoint_transfer
                        and variant.ablation.checkpoint_transfer_enabled is None
                    ),
                )
            )
        else:
            raise ValueError(f"unsupported experiment stage {stage!r}")
    return plans


def _architecture_args(
    args: argparse.Namespace, architecture_config: Mapping[str, Any]
) -> list[str]:
    values = ["--architecture", args.architecture_name]
    for key, value in architecture_config.items():
        values.extend(["--architecture-config", f"{key}={value}"])
    return values


def _synthetic_plan(
    args: argparse.Namespace,
    architecture_config: Mapping[str, Any],
    stage_variant_args: Sequence[str] = (),
) -> StageRunPlan:
    stage_dir = args.artifacts_dir / "synthetic_1d"
    summary_path = stage_dir / "run_summary.json"
    checkpoint_path = stage_dir / "checkpoint.pth"
    stage_args = [
        "--output",
        str(summary_path),
        "--checkpoint",
        str(checkpoint_path),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--epochs",
        str(args.synthetic_epochs),
        "--train-samples",
        str(args.synthetic_train_samples),
        "--validation-samples",
        str(args.synthetic_validation_samples),
        "--test-samples",
        str(args.synthetic_test_samples),
        *_architecture_args(args, architecture_config),
        *stage_variant_args,
    ]
    return StageRunPlan(
        stage="synthetic-1d",
        command=[
            "retroagi",
            "train",
            "--game",
            getattr(args, "game", "smb"),
            "--stage",
            "synthetic-1d",
            *stage_args,
        ],
        stage_args=["train", *stage_args],
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
    )


def _block_smb_plan(
    args: argparse.Namespace,
    architecture_config: Mapping[str, Any],
    stage_variant_args: Sequence[str] = (),
    *,
    disable_checkpoint_transfer: bool = True,
) -> StageRunPlan:
    stage_dir = args.artifacts_dir / "block_smb"
    summary_path = stage_dir / "run_summary.json"
    checkpoint_path = stage_dir / "checkpoint.pth"
    log_path = stage_dir / "events.jsonl"
    fixed_scenarios = args.block_fixed_scenario or ["level_1_flat.json"]
    stage_args = [
        "--output",
        str(summary_path),
        "--checkpoint",
        str(checkpoint_path),
        "--log-path",
        str(log_path),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--epochs",
        str(args.block_epochs),
        "--episodes-per-epoch",
        str(args.block_episodes_per_epoch),
        "--rollout-steps",
        str(args.block_rollout_steps),
        "--generated-scenarios",
        "0",
        "--monte-carlo-train-samples-per-epoch",
        "0",
        "--monte-carlo-validation-samples",
        "0",
        "--monte-carlo-test-samples",
        "0",
        "--evaluation-episodes",
        str(args.block_evaluation_episodes),
        "--evaluation-max-steps",
        str(args.block_evaluation_max_steps),
        "--evaluation-interval-epochs",
        "1",
        *_architecture_args(args, architecture_config),
        *stage_variant_args,
    ]
    for scenario in fixed_scenarios:
        stage_args.extend(["--fixed-scenario", scenario])
    if disable_checkpoint_transfer:
        stage_args.append("--disable-checkpoint-transfer")
    return StageRunPlan(
        stage="block-smb",
        command=[
            "retroagi",
            "train",
            "--game",
            getattr(args, "game", "smb"),
            "--stage",
            "block-smb",
            *stage_args,
        ],
        stage_args=["train", *stage_args],
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
    )


def run_experiment(
    args: argparse.Namespace,
    *,
    runners: Mapping[str, Callable[[Sequence[str]], int]] | None = None,
) -> dict[str, Any]:
    plans = build_experiment_plans(args)
    plugin = _experiment_game_plugin(args)
    runners = dict(runners or _default_runners())
    gates = list(args.gate or ())
    stage_results = []
    for plan in plans:
        plan.summary_path.parent.mkdir(parents=True, exist_ok=True)
        runner = runners[plan.stage]
        exit_code = int(runner(plan.stage_args))
        summary = _load_stage_summary(plan.summary_path)
        metrics = summary.get("metrics", {})
        if not isinstance(metrics, Mapping):
            metrics = {}
        gate_results = [gate.evaluate(metrics) for gate in gates if gate.applies_to(plan.stage)]
        stage_result = {
            "stage": plan.stage,
            "game_stage": _stage_game_manifest(plugin, plan.stage),
            "command": plan.command,
            "summary_path": str(plan.summary_path),
            "checkpoint_path": str(plan.checkpoint_path),
            "log_path": str(plan.log_path) if plan.log_path is not None else None,
            "recordings": [],
            "exit_code": exit_code,
            "config": summary.get("config", {}),
            "metrics": dict(metrics),
            "gates": gate_results,
            "passed": exit_code == 0 and all(result["passed"] for result in gate_results),
        }
        stage_result["promotion_decision"] = _stage_promotion_decision(
            plugin,
            plan.stage,
            stage_result,
        )
        stage_results.append(stage_result)
    return _manifest(args, stage_results, gates)


def _default_runners() -> dict[str, Callable[[Sequence[str]], int]]:
    from retroagi.stages.block_smb import cli as block_smb_cli
    from retroagi.stages.synthetic_1d import cli as synthetic_cli

    return {
        "synthetic-1d": synthetic_cli.main,
        "block-smb": block_smb_cli.main,
    }


def _load_stage_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _manifest(
    args: argparse.Namespace,
    stage_results: list[dict[str, Any]],
    gates: Sequence[MetricGate],
) -> dict[str, Any]:
    variant = build_architecture_variant(args.architecture_config or (), args.ablation or ())
    plugin = _experiment_game_plugin(args)
    promotion_decisions = [
        stage["promotion_decision"] for stage in stage_results if "promotion_decision" in stage
    ]
    rung_statuses = {
        rung: decision["status"]
        for decision in promotion_decisions
        for rung in decision["architecture_rungs"]
    }
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": dict(variant.architecture_config),
        },
        "architecture_variant": variant.metadata(),
        "seed": args.seed,
        "device": args.device,
        "game": _game_manifest(plugin),
        "game_promotion": build_game_promotion_plan(plugin).to_manifest(
            rung_statuses,
            plugin.promotion_gates,
        ),
        "artifacts_dir": str(args.artifacts_dir),
        "stages": stage_results,
        "gates": [to_plain_data(gate) for gate in gates],
        "promotion_decisions": promotion_decisions,
        "passed": all(stage["passed"] for stage in stage_results),
    }


def _experiment_game_plugin(args: argparse.Namespace):
    return get_game_plugin(getattr(args, "game", "smb"))


def _resolve_experiment_game_stage(plugin, experiment_stage: str):
    game_stage = EXPERIMENT_STAGE_GAME_STAGE[experiment_stage]
    resolution = plugin.resolve_stage(game_stage)
    expected_stage_spec = experiment_stage.replace("-", "_")
    if resolution.stage_spec_name != expected_stage_spec:
        raise ValueError(
            f"game {plugin.name!r} stage {game_stage!r} resolves to "
            f"{resolution.stage_spec_name!r}, but this experiment runner supports "
            f"{expected_stage_spec!r}"
        )
    return resolution


def _game_manifest(plugin) -> dict[str, Any]:
    game = plugin.game
    return {
        "name": plugin.name,
        "family": game.family,
        "backend": {
            "name": game.emulator_backend,
            "version": _backend_version(game.emulator_backend),
            "contract": game.backend_spec().to_manifest(),
        },
        "stage_ladder": [
            {
                "name": stage.name,
                "stage_spec_name": stage.stage_spec_name,
                "role": stage.role,
                "required_artifacts": list(stage.required_artifacts),
                "promotion_gate_summary": stage.promotion_gate_summary,
            }
            for stage in game.stage_ladder
        ],
        "content_identifiers": [
            {
                "name": asset.name,
                "required": asset.required,
                "local_path": asset.local_path,
            }
            for asset in game.asset_requirements
        ],
        "asset_provenance": [
            {
                "name": asset.name,
                "provenance": asset.provenance,
                "license_notes": asset.license_notes,
            }
            for asset in game.asset_requirements
        ],
        "asset_checklist": [item.to_manifest() for item in game.asset_checklist],
        "licensing": dict(game.licensing),
    }


def _stage_game_manifest(plugin, experiment_stage: str) -> dict[str, Any]:
    resolution = _resolve_experiment_game_stage(plugin, experiment_stage)
    stage = next(stage for stage in plugin.game.stage_ladder if stage.name == resolution.name)
    try:
        perception_pipeline = plugin.perception_pipeline(resolution.name)
    except KeyError:
        perception_pipeline = None
    return {
        "name": resolution.name,
        "stage_spec_name": resolution.stage_spec_name,
        "role": resolution.role,
        "required_artifacts": list(stage.required_artifacts),
        "promotion_gate_summary": stage.promotion_gate_summary,
        "stage_adapter": plugin.stage_adapters.get(resolution.name),
        "vision_encoder": plugin.vision_encoders.get(resolution.name),
        "perception_pipeline": (
            perception_pipeline.to_manifest() if perception_pipeline is not None else None
        ),
    }


def _stage_promotion_decision(
    plugin,
    experiment_stage: str,
    stage_result: Mapping[str, Any],
) -> dict[str, Any]:
    rungs = EXPERIMENT_STAGE_PROMOTION_RUNGS[experiment_stage]
    passed = bool(stage_result.get("passed"))
    status = "passed" if passed else "failed"
    return {
        "stage": experiment_stage,
        "game_stage": stage_result["game_stage"]["name"],
        "architecture_rungs": list(rungs),
        "status": status,
        "passed": passed,
        "promotion_gates": [
            plugin.promotion_gates[rung].to_manifest()
            for rung in rungs
            if rung in plugin.promotion_gates
        ],
        "reason": None if passed else "stage failed experiment or metric gates",
    }


def _backend_version(backend_name: str) -> str | None:
    packages = {
        "stable-retro": "stable-retro",
    }
    package_name = packages.get(backend_name)
    if package_name is None:
        return None
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = run_experiment(args)
    output = json.dumps(to_plain_data(manifest), indent=2, sort_keys=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if manifest["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
