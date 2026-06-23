"""Stage-agnostic experiment runner for architecture comparisons."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from retroagi.core import BASELINE_ARCHITECTURE_NAME, architecture_names, to_plain_data

STAGE_ALIASES = {
    "synthetic-1d": "synthetic-1d",
    "synthetic_1d": "synthetic-1d",
    "synthetic": "synthetic-1d",
    "block-smb": "block-smb",
    "block_smb": "block-smb",
    "block": "block-smb",
}
SUPPORTED_EXPERIMENT_STAGES = tuple(sorted({"synthetic-1d", "block-smb"}))
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
    architecture_config = dict(args.architecture_config or ())
    plans = []
    for stage in args.stage:
        if stage == "synthetic-1d":
            plans.append(_synthetic_plan(args, architecture_config))
        elif stage == "block-smb":
            plans.append(_block_smb_plan(args, architecture_config))
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
    args: argparse.Namespace, architecture_config: Mapping[str, Any]
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
    ]
    return StageRunPlan(
        stage="synthetic-1d",
        command=["retroagi", "train", "--stage", "synthetic-1d", *stage_args],
        stage_args=["train", *stage_args],
        summary_path=summary_path,
        checkpoint_path=checkpoint_path,
    )


def _block_smb_plan(
    args: argparse.Namespace, architecture_config: Mapping[str, Any]
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
        "--evaluation-episodes",
        str(args.block_evaluation_episodes),
        "--evaluation-max-steps",
        str(args.block_evaluation_max_steps),
        "--evaluation-interval-epochs",
        "1",
        *_architecture_args(args, architecture_config),
    ]
    for scenario in fixed_scenarios:
        stage_args.extend(["--fixed-scenario", scenario])
    if not args.enable_block_checkpoint_transfer:
        stage_args.append("--disable-checkpoint-transfer")
    return StageRunPlan(
        stage="block-smb",
        command=["retroagi", "train", "--stage", "block-smb", *stage_args],
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
        stage_results.append(
            {
                "stage": plan.stage,
                "command": plan.command,
                "summary_path": str(plan.summary_path),
                "checkpoint_path": str(plan.checkpoint_path),
                "log_path": str(plan.log_path) if plan.log_path is not None else None,
                "exit_code": exit_code,
                "config": summary.get("config", {}),
                "metrics": dict(metrics),
                "gates": gate_results,
                "passed": exit_code == 0 and all(result["passed"] for result in gate_results),
            }
        )
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
    return {
        "architecture": {
            "name": args.architecture_name,
            "config": dict(args.architecture_config or ()),
        },
        "seed": args.seed,
        "device": args.device,
        "artifacts_dir": str(args.artifacts_dir),
        "stages": stage_results,
        "gates": [to_plain_data(gate) for gate in gates],
        "passed": all(stage["passed"] for stage in stage_results),
    }


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
