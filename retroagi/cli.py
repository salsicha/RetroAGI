"""Project-level command line interface for RetroAGI stages."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

STAGE_ALIASES = {
    "synthetic-1d": "synthetic-1d",
    "synthetic_1d": "synthetic-1d",
    "synthetic": "synthetic-1d",
    "block-smb": "block-smb",
    "block_smb": "block-smb",
    "block": "block-smb",
    "full-smb": "full-smb",
    "full_smb": "full-smb",
    "full": "full-smb",
}


def _stage_name(value: str) -> str:
    try:
        return STAGE_ALIASES[value.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted({"synthetic-1d", "block-smb", "full-smb"}))
        raise argparse.ArgumentTypeError(
            f"unknown stage {value!r}; expected one of: {choices}"
        ) from exc


def _add_stage_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stage",
        "--env",
        dest="stage",
        required=True,
        type=_stage_name,
        help="stage/environment to run: synthetic-1d, block-smb, or full-smb",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi",
        description="Run RetroAGI training, evaluation, resume, and stage utilities.",
        epilog=(
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

    diagnose = subparsers.add_parser(
        "diagnose-vision",
        help="run perception diagnostics for stages that expose them",
    )
    _add_stage_arg(diagnose)

    transfer = subparsers.add_parser("transfer", help="transfer checkpoints between stages")
    _add_stage_arg(transfer)

    compare = subparsers.add_parser("compare", help="compare checkpoints for a stage")
    _add_stage_arg(compare)

    return parser


def run(args: argparse.Namespace, stage_args: Sequence[str]) -> int:
    command = str(args.command)
    stage = str(args.stage)

    if stage == "block-smb":
        return _run_block_smb(args, stage_args)
    if stage == "full-smb":
        return _run_full_smb(command, stage_args)
    if stage == "synthetic-1d":
        return _run_synthetic_1d(args, stage_args)
    raise ValueError(f"unsupported stage {stage!r}")


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


def _run_full_smb(command: str, stage_args: Sequence[str]) -> int:
    if command == "evaluate":
        return _run_full_smb_evaluate(stage_args)
    if command == "transfer":
        from retroagi.stages.full_smb.transfer import main as transfer_main

        transfer_main(list(stage_args))
        return 0
    if command == "compare":
        from retroagi.stages.full_smb.compare import main as compare_main

        compare_main(list(stage_args))
        return 0
    raise ValueError(
        "Full SMB currently supports evaluate, transfer, and compare through "
        "the top-level CLI; direct training/resume is not implemented yet"
    )


def _run_full_smb_evaluate(stage_args: Sequence[str]) -> int:
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
