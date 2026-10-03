"""Command line for the four-layer Block SMB agent: train one layer, or examine one."""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

from .monte_carlo import BLOCK_SMB_MC_FAMILIES


def _add_layer_args(parser: argparse.ArgumentParser) -> None:
    """One option per LayeredTrainConfig setting, with the same name and default."""
    import dataclasses

    from .layered_train import LayeredTrainConfig

    for setting in dataclasses.fields(LayeredTrainConfig):
        option = "--" + setting.name.replace("_", "-")
        if setting.name == "families":
            parser.add_argument(option, nargs="+", choices=BLOCK_SMB_MC_FAMILIES)
        elif isinstance(setting.default, tuple):
            parser.add_argument(option, nargs=len(setting.default), type=float, default=None)
        elif setting.default is None or isinstance(setting.default, str):
            parser.add_argument(option, type=str, default=setting.default)
        else:
            parser.add_argument(option, type=type(setting.default), default=setting.default)


def _run_train_layer(args: argparse.Namespace) -> dict[str, Any]:
    import dataclasses

    from .layered_train import LayeredTrainConfig, train_layer

    settings = {
        setting.name: getattr(args, setting.name)
        for setting in dataclasses.fields(LayeredTrainConfig)
        if getattr(args, setting.name) is not None or setting.default is None
    }
    for name, value in list(settings.items()):
        if isinstance(value, list):
            settings[name] = tuple(value)
        elif value is None and name != "init":
            settings.pop(name)
    summary = train_layer(LayeredTrainConfig(**settings))
    return {"best_validation_success": summary["best_validation_success"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi-block-smb",
        description="Train and examine the layers of the four-layer Block SMB agent.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_layer = subparsers.add_parser(
        "train-layer",
        help="train one layer of the four-layer agent (action, then skill, then tactic)",
    )
    _add_layer_args(train_layer)

    exam_layer = subparsers.add_parser(
        "exam-layer",
        help="play fresh held-out layouts with a saved four-layer policy",
    )
    exam_layer.add_argument("--checkpoint", required=True)
    exam_layer.add_argument(
        "--learner",
        choices=("action", "skill", "tactic", "deployed"),
        default="deployed",
        help="the layer under test (the teacher gives its token from above), or the whole agent",
    )
    exam_layer.add_argument("--layouts-per-difficulty", type=int, default=6)
    exam_layer.add_argument("--first-layout", type=int, default=100)
    exam_layer.add_argument("--workers", type=int, default=12)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "train-layer":
        return _run_train_layer(args)
    if args.command == "exam-layer":
        from .layered_train import examine_layer

        return examine_layer(
            args.checkpoint,
            None if args.learner == "deployed" else args.learner,
            layouts_per_difficulty=args.layouts_per_difficulty,
            first_layout=args.first_layout,
            workers=args.workers,
        )
    raise ValueError(f"unknown command {args.command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # train-layer's --output is its run folder, which holds its own history.
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
