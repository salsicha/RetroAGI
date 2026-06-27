"""Benchmark Full SMB policy architecture variants."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F

from retroagi.core import (
    BASELINE_ARCHITECTURE_NAME,
    POLICY_TUPLE_OUTPUT_CONTRACTS,
    SMB_ACTIONS,
    SINGLE_PASS_LSTM_ARCHITECTURE_NAME,
    StageBatch,
    WorldModelState,
    get_architecture,
    load_checkpoint,
    select_device,
    to_plain_data,
)
from retroagi.stages.full_smb.adapter import FULL_SMB_SPEC
from retroagi.stages.full_smb.transfer import make_full_smb_policy_model

FULL_SMB_ARCHITECTURE_BENCHMARK_SCHEMA_VERSION = 1
DEFAULT_FULL_SMB_ARCHITECTURE_BENCHMARK_ARCHITECTURES = (
    BASELINE_ARCHITECTURE_NAME,
    SINGLE_PASS_LSTM_ARCHITECTURE_NAME,
)


def benchmark_full_smb_policy_architectures(
    *,
    architecture_names: Sequence[str] = DEFAULT_FULL_SMB_ARCHITECTURE_BENCHMARK_ARCHITECTURES,
    architecture_config: Optional[Mapping[str, Any]] = None,
    block_policy_checkpoint: Optional[Path] = None,
    device: str | torch.device = "auto",
    seed: int = 0,
    batch_size: int = 2,
    iterations: int = 8,
    warmup: int = 2,
) -> dict[str, Any]:
    """Compare trainer-compatible Full SMB policy architectures on one batch."""

    if not architecture_names:
        raise ValueError("architecture_names must be non-empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    resolved_device = select_device(device)
    torch.manual_seed(int(seed))
    batch = _benchmark_batch(batch_size=batch_size, device=resolved_device)
    source_state = _block_source_state(block_policy_checkpoint, map_location=resolved_device)
    results: dict[str, Any] = {}
    models: dict[str, torch.nn.Module] = {}
    for name in architecture_names:
        architecture = get_architecture(name)
        if architecture.output_contract not in POLICY_TUPLE_OUTPUT_CONTRACTS:
            raise ValueError(
                f"architecture {name!r} has unsupported output contract "
                f"{architecture.output_contract!r}"
            )
        config = _architecture_config_for_source(
            name,
            architecture_config=architecture_config,
            source_state=source_state,
        )
        model = make_full_smb_policy_model(
            architecture_name=name,
            architecture_config=config,
        ).to(resolved_device)
        transfer = _load_source_state(model, source_state)
        models[name] = model
        results[name] = {
            "architecture": architecture.metadata(),
            "architecture_config": dict(config),
            "transfer": transfer,
            "train_step": _benchmark_train_step(
                model,
                batch,
                device=resolved_device,
                iterations=iterations,
                warmup=warmup,
            ),
            "inference": _benchmark_inference(
                model,
                batch,
                device=resolved_device,
                iterations=iterations,
                warmup=warmup,
            ),
            "action_quality": _action_quality(model, batch, device=resolved_device),
            "recurrent_state": _recurrent_state_probe(model, batch, device=resolved_device),
        }
    return {
        "schema_version": FULL_SMB_ARCHITECTURE_BENCHMARK_SCHEMA_VERSION,
        "config": {
            "architectures": list(architecture_names),
            "architecture_config": dict(architecture_config or {}),
            "block_policy_checkpoint": (
                str(block_policy_checkpoint) if block_policy_checkpoint is not None else None
            ),
            "device": str(resolved_device),
            "seed": int(seed),
            "batch_size": int(batch_size),
            "iterations": int(iterations),
            "warmup": int(warmup),
        },
        "architectures": results,
        "comparison": _pairwise_comparison(models, batch, device=resolved_device),
    }


def _benchmark_batch(*, batch_size: int, device: torch.device) -> StageBatch:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(41_001 + batch_size)
    src_a = torch.randint(
        0,
        FULL_SMB_SPEC.vocab_size,
        (batch_size, FULL_SMB_SPEC.seq_len_a),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    src_b = torch.randint(
        0,
        FULL_SMB_SPEC.vocab_size,
        (batch_size, FULL_SMB_SPEC.seq_len_b),
        generator=generator,
        dtype=torch.long,
    ).to(device)
    src_c = torch.randn(
        (batch_size, FULL_SMB_SPEC.seq_len_c),
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    return StageBatch(
        src_a=src_a,
        target_a=None,
        src_b=src_b,
        target_b=None,
        src_c=src_c,
        target_c=None,
        metadata={"episode": {"mask": torch.ones(batch_size, dtype=torch.float32, device=device)}},
    )


def _architecture_config_for_source(
    architecture_name: str,
    *,
    architecture_config: Optional[Mapping[str, Any]],
    source_state: Optional[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    config = dict(architecture_config or {})
    if source_state is not None and "agent.embedding.weight" in source_state:
        config.setdefault("hidden_dim", int(source_state["agent.embedding.weight"].shape[1]))
    if architecture_name == SINGLE_PASS_LSTM_ARCHITECTURE_NAME:
        config.setdefault("world_context_scale", 1.0)
    return config


def _block_source_state(
    checkpoint_path: Optional[Path],
    *,
    map_location: torch.device,
) -> Optional[Mapping[str, torch.Tensor]]:
    if checkpoint_path is None:
        return None
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=map_location)
    state = checkpoint.get("states", {}).get("model")
    if not isinstance(state, Mapping):
        raise ValueError("block policy checkpoint is missing states.model")
    return state


def _load_source_state(
    model: torch.nn.Module,
    source_state: Optional[Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    if source_state is None:
        return {
            "source": None,
            "loaded": False,
            "missing_keys": [],
            "unexpected_keys": [],
            "shared_key_count": 0,
        }
    load_result = model.load_state_dict(source_state, strict=False)
    shared_key_count = sum(1 for key in source_state if key in model.state_dict())
    return {
        "source": "block_smb_policy_checkpoint",
        "loaded": True,
        "missing_keys": list(load_result.missing_keys),
        "unexpected_keys": list(load_result.unexpected_keys),
        "shared_key_count": int(shared_key_count),
    }


def _benchmark_train_step(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        _train_step(model, optimizer, batch, device=device)
    _sync_device(device)
    start = time.perf_counter()
    losses = []
    for _ in range(iterations):
        losses.append(_train_step(model, optimizer, batch, device=device))
    _sync_device(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(iterations / elapsed),
        "samples_per_second": float(batch.src_a.size(0) * iterations / elapsed),
        "final_loss": float(losses[-1]) if losses else 0.0,
    }


def _train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: StageBatch,
    *,
    device: torch.device,
) -> float:
    outputs = _forward(model, batch, device=device, return_world_model_state=False)
    logits = outputs[4][..., : len(SMB_ACTIONS)]
    next_state_pred = outputs[1]
    target_actions = batch.src_a[:, -1].to(device).remainder(len(SMB_ACTIONS))
    loss = F.cross_entropy(logits[:, -1, :], target_actions)
    loss = loss + 0.01 * F.mse_loss(next_state_pred, batch.src_c.to(device))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


@torch.no_grad()
def _benchmark_inference(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    model.eval()
    for _ in range(warmup):
        _forward(model, batch, device=device, return_world_model_state=True)
    _sync_device(device)
    start = time.perf_counter()
    for _ in range(iterations):
        _forward(model, batch, device=device, return_world_model_state=True)
    _sync_device(device)
    elapsed = max(time.perf_counter() - start, 1e-9)
    return {
        "elapsed_seconds": float(elapsed),
        "steps_per_second": float(iterations / elapsed),
        "samples_per_second": float(batch.src_a.size(0) * iterations / elapsed),
    }


@torch.no_grad()
def _action_quality(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    outputs = _forward(model, batch, device=device, return_world_model_state=True)
    logits = outputs[4].detach().float()
    final_logits = logits[:, -1, : len(SMB_ACTIONS)]
    probabilities = final_logits.softmax(dim=-1)
    top2 = torch.topk(probabilities, k=min(2, probabilities.size(-1)), dim=-1).values
    margin = top2[:, 0] - (top2[:, 1] if top2.size(-1) > 1 else 0.0)
    actions = final_logits.argmax(dim=-1)
    entropy = torch.distributions.Categorical(logits=final_logits).entropy()
    return {
        "mean_entropy": float(entropy.mean().cpu().item()),
        "mean_top1_margin": float(margin.mean().cpu().item()),
        "unique_actions": int(actions.unique().numel()),
        "action_histogram": {
            str(int(action)): int((actions == action).sum().cpu().item())
            for action in actions.unique()
        },
        "logit_abs_max": float(final_logits.abs().max().cpu().item()),
    }


@torch.no_grad()
def _recurrent_state_probe(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    first = _forward(model, batch, device=device, return_world_model_state=True)
    state = first[-1]
    second = _forward(
        model,
        batch,
        device=device,
        world_model_state=state if isinstance(state, WorldModelState) else None,
        return_world_model_state=True,
    )
    next_state = second[-1]
    first_logits = first[4][:, -1, :].detach()
    first_logits = first_logits[:, : len(SMB_ACTIONS)]
    second_logits = second[4][:, -1, : len(SMB_ACTIONS)].detach()
    return {
        "state_available": isinstance(state, WorldModelState),
        "state_hidden_norm": _state_norm(state),
        "next_state_hidden_norm": _state_norm(next_state),
        "carried_logits_delta_mean_abs": float(
            (second_logits - first_logits).abs().mean().cpu().item()
        ),
    }


@torch.no_grad()
def _pairwise_comparison(
    models: Mapping[str, torch.nn.Module],
    batch: StageBatch,
    *,
    device: torch.device,
) -> dict[str, Any]:
    names = list(models)
    if len(names) < 2:
        return {}
    baseline_name, candidate_name = names[0], names[1]
    first_logits = _forward(
        models[baseline_name],
        batch,
        device=device,
        return_world_model_state=True,
    )[4][:, -1, : len(SMB_ACTIONS)]
    second_logits = _forward(
        models[candidate_name],
        batch,
        device=device,
        return_world_model_state=True,
    )[4][:, -1, : len(SMB_ACTIONS)]
    first_actions = first_logits.argmax(dim=-1)
    second_actions = second_logits.argmax(dim=-1)
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "action_agreement": float((first_actions == second_actions).float().mean().cpu().item()),
        "mean_abs_logit_delta": float((second_logits - first_logits).abs().mean().cpu().item()),
    }


def _forward(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: torch.device,
    world_model_state: WorldModelState | None = None,
    return_world_model_state: bool,
) -> tuple[Any, ...]:
    return model(
        batch.src_a.to(device),
        batch.src_b.to(device),
        batch.src_c.to(device),
        tau=1.0,
        world_model_state=world_model_state,
        episode_mask=torch.ones(batch.src_a.size(0), dtype=torch.float32, device=device),
        return_world_model_state=return_world_model_state,
    )


def _state_norm(state: Any) -> float:
    if not isinstance(state, WorldModelState):
        return 0.0
    return float(state.hidden.detach().norm().cpu().item())


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi benchmark-architecture --stage full")
    parser.add_argument(
        "--architecture",
        dest="architecture_names",
        action="append",
        default=[],
        help="architecture name to benchmark; defaults to baseline and single-pass",
    )
    parser.add_argument("--architecture-config", action="append", default=[])
    parser.add_argument("--block-policy-checkpoint", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    architecture_config = dict(_config_item(item) for item in args.architecture_config)
    result = benchmark_full_smb_policy_architectures(
        architecture_names=tuple(args.architecture_names)
        or DEFAULT_FULL_SMB_ARCHITECTURE_BENCHMARK_ARCHITECTURES,
        architecture_config=architecture_config,
        block_policy_checkpoint=args.block_policy_checkpoint,
        device=args.device,
        seed=args.seed,
        batch_size=args.batch_size,
        iterations=args.iterations,
        warmup=args.warmup,
    )
    output = json.dumps(to_plain_data(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


def _config_item(value: str) -> tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("architecture config must use KEY=VALUE syntax")
    key, raw = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("architecture config key must be non-empty")
    try:
        parsed: Any = int(raw)
    except ValueError:
        try:
            parsed = float(raw)
        except ValueError:
            parsed = raw
    return key, parsed


if __name__ == "__main__":
    raise SystemExit(main())
