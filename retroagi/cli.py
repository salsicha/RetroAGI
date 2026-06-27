"""Project-level command line interface for RetroAGI stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from retroagi.core import game_plugin_names, get_game_plugin, normalize_stage_name


def _stage_name(value: str) -> str:
    try:
        return normalize_stage_name(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _game_name(value: str) -> str:
    name = value.lower()
    if name in game_plugin_names():
        return name
    available = ", ".join(game_plugin_names())
    raise argparse.ArgumentTypeError(f"unknown game {value!r}; available game plugins: {available}")


def _add_game_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--game",
        default="smb",
        type=_game_name,
        help="game profile to run; default: smb",
    )


def _add_stage_arg(parser: argparse.ArgumentParser) -> None:
    _add_game_arg(parser)
    parser.add_argument(
        "--stage",
        "--env",
        dest="stage",
        required=True,
        type=_stage_name,
        help=(
            "game-neutral stage to run: synthetic, block, or full; legacy "
            "synthetic-1d, block-smb, and full-smb aliases are accepted"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi",
        description="Run RetroAGI training, evaluation, resume, and stage utilities.",
        epilog=(
            "Select the game with --game and the fidelity rung with --stage. "
            "Stage-specific options after --stage/--env are forwarded to the "
            "selected implementation. Use retroagi-block-smb for the legacy "
            "Block SMB-only entry point."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="train the selected stage")
    _add_stage_arg(train)

    evaluate = subparsers.add_parser("evaluate", help="evaluate the selected stage")
    _add_stage_arg(evaluate)

    resume = subparsers.add_parser("resume", help="resume training for the selected stage")
    _add_stage_arg(resume)
    resume.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="existing checkpoint to resume from",
    )
    resume.add_argument(
        "--save-checkpoint",
        type=Path,
        help="optional destination for the resumed checkpoint",
    )

    record = subparsers.add_parser("record", help="record evaluation artifacts")
    _add_stage_arg(record)

    play = subparsers.add_parser("play", help="play a saved policy for the selected stage")
    _add_stage_arg(play)

    imitate = subparsers.add_parser("imitate", help="run imitation warm starts")
    _add_stage_arg(imitate)

    gate = subparsers.add_parser(
        "gate",
        help="run short curriculum gates before expensive benchmark evaluation",
    )
    _add_stage_arg(gate)

    diagnose = subparsers.add_parser(
        "diagnose-vision",
        help="run perception diagnostics for stages that expose them",
    )
    _add_stage_arg(diagnose)

    diagnose_actions = subparsers.add_parser(
        "diagnose-actions",
        help="run policy action-contract diagnostics for stages that expose them",
    )
    _add_stage_arg(diagnose_actions)

    transfer = subparsers.add_parser("transfer", help="transfer checkpoints between stages")
    _add_stage_arg(transfer)

    compare = subparsers.add_parser("compare", help="compare checkpoints for a stage")
    _add_stage_arg(compare)

    benchmark_architecture = subparsers.add_parser(
        "benchmark-architecture",
        help="benchmark trainer-compatible architecture variants for a stage",
    )
    _add_stage_arg(benchmark_architecture)

    check_env = subparsers.add_parser(
        "check-env",
        help="check the selected stage backend and local content setup",
    )
    _add_stage_arg(check_env)

    subparsers.add_parser(
        "experiment",
        help="run one architecture through selected stages and write a combined manifest",
    )
    subparsers.add_parser(
        "promote",
        help="run progressive-resolution promotion checks for an architecture",
    )
    subparsers.add_parser(
        "report",
        help="build an architecture sweep comparison report from saved manifests",
    )

    return parser


def run(args: argparse.Namespace, stage_args: Sequence[str]) -> int:
    command = str(args.command)
    if command == "experiment":
        return _run_experiment(stage_args)
    if command == "promote":
        return _run_promotion(stage_args)
    if command == "report":
        return _run_report(stage_args)
    game_plugin = get_game_plugin(str(args.game))
    stage = game_plugin.resolve_stage(str(args.stage)).name
    game_plugin.stage_adapter(stage)

    if game_plugin.name != "smb":
        raise ValueError(f"game {game_plugin.name!r} does not have CLI runners yet")
    if stage == "block":
        return _run_block_smb(args, stage_args)
    if stage == "full":
        return _run_full_smb(args, stage_args)
    if stage == "synthetic":
        return _run_synthetic_1d(args, stage_args)
    raise ValueError(f"unsupported stage {stage!r} for game {game_plugin.name!r}")


def _run_block_smb(args: argparse.Namespace, stage_args: Sequence[str]) -> int:
    from retroagi.stages.block_smb import cli as block_smb_cli

    if args.command == "resume":
        block_args = ["train", "--resume", str(args.checkpoint)]
        if args.save_checkpoint is not None:
            block_args.extend(["--checkpoint", str(args.save_checkpoint)])
        block_args.extend(stage_args)
    elif args.command in {"train", "evaluate", "record", "diagnose-vision"}:
        block_args = [str(args.command), *stage_args]
    else:
        raise ValueError(f"Block SMB does not support retroagi {args.command!r}")
    return int(block_smb_cli.main(block_args))


def _run_full_smb(args: argparse.Namespace, stage_args: Sequence[str]) -> int:
    command = str(args.command)
    if command == "train":
        from retroagi.stages.full_smb.train import main as train_main

        return int(train_main(["train", *stage_args]))
    if command == "resume":
        from retroagi.stages.full_smb.train import main as train_main

        train_args = ["resume", "--checkpoint", str(args.checkpoint)]
        if args.save_checkpoint is not None:
            train_args.extend(["--save-checkpoint", str(args.save_checkpoint)])
        train_args.extend(stage_args)
        return int(train_main(train_args))
    if command == "evaluate":
        return _run_full_smb_evaluate(stage_args)
    if command == "record":
        from retroagi.stages.full_smb.train import main as train_main

        return int(train_main(["record", *stage_args]))
    if command == "play":
        from retroagi.stages.full_smb.train import main as train_main

        return int(train_main(["play", *stage_args]))
    if command == "imitate":
        from retroagi.stages.full_smb.imitation import main as imitation_main

        return int(imitation_main(list(stage_args)))
    if command == "gate":
        from retroagi.stages.full_smb.curriculum_gates import main as gate_main

        return int(gate_main(list(stage_args)))
    if command == "diagnose-vision":
        from retroagi.stages.full_smb.diagnostics import main as diagnostics_main

        return int(diagnostics_main(list(stage_args)))
    if command == "diagnose-actions":
        from retroagi.stages.full_smb.action_diagnostics import main as diagnostics_main

        return int(diagnostics_main(list(stage_args)))
    if command == "transfer":
        from retroagi.stages.full_smb.transfer import main as transfer_main

        transfer_main(list(stage_args))
        return 0
    if command == "compare":
        from retroagi.stages.full_smb.compare import main as compare_main

        compare_main(list(stage_args))
        return 0
    if command == "benchmark-architecture":
        from retroagi.stages.full_smb.architecture_benchmark import main as benchmark_main

        return int(benchmark_main(list(stage_args)))
    if command == "check-env":
        from retroagi.stages.full_smb.capabilities import main as capabilities_main

        return int(capabilities_main(list(stage_args)))
    raise ValueError(
        "Full SMB currently supports train, resume, evaluate, diagnose-vision, "
        "diagnose-actions, record, play, imitate, gate, transfer, compare, and "
        "benchmark-architecture, and check-env through the top-level CLI"
    )


def _run_full_smb_evaluate(stage_args: Sequence[str]) -> int:
    if any(
        arg in {"--policy-checkpoint", "--checkpoint"}
        or arg.startswith("--policy-checkpoint=")
        or arg.startswith("--checkpoint=")
        for arg in stage_args
    ):
        from retroagi.stages.full_smb.train import main as train_main

        return int(train_main(["evaluate", *stage_args]))

    from retroagi.stages.full_smb.run import main as run_full_smb

    parser = argparse.ArgumentParser(prog="retroagi evaluate --stage full-smb")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--encode-observations", action="store_true")
    args = parser.parse_args(stage_args)
    run_full_smb(
        num_steps=args.steps,
        seed=args.seed,
        render=args.render,
        encode_observations=args.encode_observations,
    )
    return 0


def _run_synthetic_1d(args: argparse.Namespace, stage_args: Sequence[str]) -> int:
    from retroagi.stages.synthetic_1d import cli as synthetic_cli

    if args.command == "train":
        synthetic_args = ["train", *stage_args]
    elif args.command == "resume":
        synthetic_args = ["train", "--resume", str(args.checkpoint)]
        if args.save_checkpoint is not None:
            synthetic_args.extend(["--checkpoint", str(args.save_checkpoint)])
        synthetic_args.extend(stage_args)
    else:
        raise ValueError(
            "Synthetic 1D currently supports train and resume through the top-level CLI"
        )
    return int(synthetic_cli.main(synthetic_args))


def _run_experiment(stage_args: Sequence[str]) -> int:
    from retroagi import experiments

    return int(experiments.main(list(stage_args)))


def _run_promotion(stage_args: Sequence[str]) -> int:
    from retroagi import promotion

    return int(promotion.main(list(stage_args)))


def _run_report(stage_args: Sequence[str]) -> int:
    from retroagi import reports

    return int(reports.main(list(stage_args)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, stage_args = parser.parse_known_args(argv)
    try:
        return run(args, stage_args)
    except ValueError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
