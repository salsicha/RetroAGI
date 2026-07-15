"""Command line entry point for Synthetic 1D training and resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    SUPPORTED_CONTROLLER_SCHEDULES,
    architecture_names,
    load_checkpoint,
    to_plain_data,
)

from .train import (
    SyntheticSplitSeeds,
    SyntheticSplitSizes,
    SyntheticTrainingConfig,
    train_and_evaluate,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


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
    return key, _parse_architecture_config_value(raw_value.strip())


def _parse_architecture_config_value(value: str) -> Any:
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi-synthetic-1d",
        description="Train or resume the Synthetic 1D architecture validation stage.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train or resume Synthetic 1D")
    train.add_argument("--output", type=Path, help="write the resolved run summary JSON")
    train.add_argument("--seed", type=int, help="base random seed")
    train.add_argument("--epochs", type=_positive_int)
    train.add_argument("--batch-size", type=_positive_int)
    train.add_argument("--learning-rate", type=_positive_float)
    train.add_argument("--critic-loss-weight", type=_non_negative_float)
    train.add_argument("--primitive-loss-weight", type=_non_negative_float)
    train.add_argument("--primitive-outcome-loss-weight", type=_non_negative_float)
    train.add_argument("--primitive-outcome-horizon", type=_positive_int)
    train.add_argument("--tau-start", type=_positive_float)
    train.add_argument("--tau-end", type=_positive_float)
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--checkpoint", type=Path, help="checkpoint path to save")
    train.add_argument("--resume", type=Path, help="checkpoint path to resume")
    train.add_argument("--train-samples", type=_positive_int)
    train.add_argument("--validation-samples", type=_positive_int)
    train.add_argument("--test-samples", type=_positive_int)
    train.add_argument("--train-seed", type=int)
    train.add_argument("--validation-seed", type=int)
    train.add_argument("--test-seed", type=int)
    train.add_argument("--hidden-dim", type=_positive_int)
    train.add_argument(
        "--controller-schedule",
        choices=SUPPORTED_CONTROLLER_SCHEDULES,
        help="low-level controller gain schedule",
    )
    train.add_argument(
        "--architecture",
        dest="architecture_name",
        type=_architecture_name,
        help="model architecture to instantiate; use 'baseline' for the default",
    )
    train.add_argument(
        "--architecture-config",
        action="append",
        default=None,
        type=_architecture_config_item,
        metavar="KEY=VALUE",
        help="architecture-specific config override; may be repeated",
    )
    train.set_defaults(deterministic=None)
    train.add_argument(
        "--deterministic",
        action="store_true",
        dest="deterministic",
        help="enable deterministic Torch algorithms",
    )
    train.add_argument(
        "--nondeterministic",
        action="store_false",
        dest="deterministic",
        help="disable deterministic Torch algorithms",
    )
    return parser


def _config_overrides(args: argparse.Namespace) -> dict[str, Any]:
    names = (
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "critic_loss_weight",
        "primitive_loss_weight",
        "primitive_outcome_loss_weight",
        "primitive_outcome_horizon",
        "tau_start",
        "tau_end",
        "device",
        "deterministic",
    )
    return {
        name: getattr(args, name)
        for name in names
        if hasattr(args, name) and getattr(args, name) is not None
    }


def _normalize_config_values(values: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    for name in ("checkpoint_path", "resume_path"):
        if normalized.get(name) is not None:
            normalized[name] = Path(normalized[name])
    if normalized.get("split_sizes") is not None:
        split_sizes = normalized["split_sizes"]
        if not isinstance(split_sizes, SyntheticSplitSizes):
            normalized["split_sizes"] = SyntheticSplitSizes(**dict(split_sizes))
    if normalized.get("split_seeds") is not None:
        split_seeds = normalized["split_seeds"]
        if not isinstance(split_seeds, SyntheticSplitSeeds):
            normalized["split_seeds"] = SyntheticSplitSeeds(**dict(split_seeds))
    if normalized.get("architecture_config") is not None:
        architecture_config = normalized["architecture_config"]
        if not isinstance(architecture_config, Mapping):
            raise TypeError("architecture_config must be a mapping")
        normalized["architecture_config"] = dict(architecture_config)
    return normalized


def _checkpoint_config(path: Path, *, epochs_override: int | None = None) -> dict[str, Any]:
    checkpoint = load_checkpoint(path)
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError(f"checkpoint {path} does not contain a configuration mapping")
    values = _normalize_config_values(config)
    values["resume_path"] = path
    values["save_checkpoints"] = False
    completed_epochs = int(checkpoint.get("epoch", 0))
    if epochs_override is not None and int(epochs_override) <= completed_epochs:
        raise ValueError(
            f"--epochs {int(epochs_override)} must exceed the {completed_epochs} "
            f"epoch(s) already completed by checkpoint {path}; resuming would "
            "train zero epochs"
        )
    if "epochs" not in values:
        values["epochs"] = max(1, int(checkpoint.get("epoch", 1)))
    else:
        values["epochs"] = max(int(values["epochs"]), completed_epochs)
    return values


def _apply_split_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
    size_fields = {
        "train": getattr(args, "train_samples", None),
        "validation": getattr(args, "validation_samples", None),
        "test": getattr(args, "test_samples", None),
    }
    size_overrides = {name: value for name, value in size_fields.items() if value is not None}
    if size_overrides:
        current_sizes = values.get("split_sizes", SyntheticSplitSizes())
        if isinstance(current_sizes, SyntheticSplitSizes):
            sizes = {
                "train": current_sizes.train,
                "validation": current_sizes.validation,
                "test": current_sizes.test,
            }
        elif isinstance(current_sizes, Mapping):
            sizes = dict(current_sizes)
        else:
            raise TypeError("split_sizes must be a SyntheticSplitSizes or mapping")
        sizes.update(size_overrides)
        values["split_sizes"] = SyntheticSplitSizes(**sizes)

    seed_fields = {
        "train": getattr(args, "train_seed", None),
        "validation": getattr(args, "validation_seed", None),
        "test": getattr(args, "test_seed", None),
    }
    seed_overrides = {name: value for name, value in seed_fields.items() if value is not None}
    if not seed_overrides:
        return
    current_seeds = values.get("split_seeds")
    if current_seeds is None:
        default_seeds = SyntheticSplitSeeds()
        seeds = {
            "train": default_seeds.train,
            "validation": default_seeds.validation,
            "test": default_seeds.test,
        }
    elif isinstance(current_seeds, SyntheticSplitSeeds):
        seeds = {
            "train": current_seeds.train,
            "validation": current_seeds.validation,
            "test": current_seeds.test,
        }
    elif isinstance(current_seeds, Mapping):
        seeds = dict(current_seeds)
    else:
        raise TypeError("split_seeds must be a SyntheticSplitSeeds, mapping, or None")
    seeds.update(seed_overrides)
    values["split_seeds"] = SyntheticSplitSeeds(**seeds)


def _apply_architecture_overrides(values: dict[str, Any], args: argparse.Namespace) -> None:
    architecture_name = getattr(args, "architecture_name", None)
    if architecture_name is not None:
        values["architecture_name"] = architecture_name

    explicit_overrides = dict(getattr(args, "architecture_config", None) or ())
    legacy_overrides = {
        name: getattr(args, name)
        for name in ("hidden_dim", "controller_schedule")
        if hasattr(args, name)
        and getattr(args, name) is not None
        and name not in explicit_overrides
    }
    if not explicit_overrides and not legacy_overrides:
        return
    current = values.get("architecture_config", {})
    if current is None:
        current_values: dict[str, Any] = {}
    elif isinstance(current, Mapping):
        current_values = dict(current)
    else:
        raise TypeError("architecture_config must be a mapping")
    current_values.update(legacy_overrides)
    current_values.update(explicit_overrides)
    values["architecture_config"] = current_values


def _make_train_config(args: argparse.Namespace) -> SyntheticTrainingConfig:
    values = (
        _checkpoint_config(args.resume, epochs_override=args.epochs)
        if args.resume is not None
        else {}
    )
    values.update(_config_overrides(args))
    _apply_split_overrides(values, args)
    _apply_architecture_overrides(values, args)
    if args.checkpoint is not None:
        values["checkpoint_path"] = args.checkpoint
        values["save_checkpoints"] = True
    elif args.resume is not None:
        values["save_checkpoints"] = False
    if args.resume is not None:
        values["resume_path"] = args.resume
    return SyntheticTrainingConfig(**values)


def _public_result(
    history: Mapping[str, list[float]], config: SyntheticTrainingConfig
) -> dict[str, Any]:
    metrics = {
        key: values[-1]
        for key, values in history.items()
        if key.startswith(("controller_", "error_", "accuracy_")) and values
    }
    return {
        "config": to_plain_data(config),
        "history": to_plain_data(history),
        "metrics": metrics,
        "architecture": {
            "name": config.architecture_name,
            "config": dict(config.architecture_config),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command != "train":
        raise ValueError(f"unknown command {args.command!r}")
    config = _make_train_config(args)
    history = train_and_evaluate(config)
    return _public_result(history, config)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run(args)
    output = json.dumps(result, indent=2, sort_keys=True)
    if getattr(args, "output", None) is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
