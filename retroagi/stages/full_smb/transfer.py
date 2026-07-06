"""Transfer Block SMB checkpoints into the Full SMB stage contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from retroagi.core import (
    ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    BASELINE_ARCHITECTURE_NAME,
    BASELINE_ARCHITECTURE_SPEC,
    POLICY_TUPLE_OUTPUT_CONTRACTS,
    SMB_ACTIONS,
    StageBatch,
    action_level_world_model_state_dict,
    build_architecture,
    build_checkpoint,
    get_architecture,
    load_checkpoint,
    save_checkpoint,
    validate_checkpoint_compatibility,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
from retroagi.stages.block_smb.monte_carlo import DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID
from retroagi.stages.block_smb.train import (
    BLOCK_SMB_CHECKPOINT_KIND,
    BLOCK_SMB_MODEL_NAME,
)
from retroagi.stages.block_smb.vision import (
    DEFAULT_BLOCK_VIT_CHECKPOINT,
    load_block_vit_checkpoint,
)
from retroagi.stages.full_smb.adapter import FULL_SMB_SPEC
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    FullSMBSegmentationVision,
)

FULL_SMB_TRANSFER_MODEL_NAME = "full_smb_transferred_block_policy"
FULL_SMB_TRANSFER_CHECKPOINT_KIND = "full_smb_transfer"


@dataclass(frozen=True)
class FullSMBTransferConfig:
    """Inputs for transferring Block SMB policy weights to Full SMB."""

    block_policy_checkpoint: Path
    output_checkpoint: Optional[Path] = None
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT
    block_vision_checkpoint: Optional[Path] = DEFAULT_BLOCK_VIT_CHECKPOINT
    device: str = "cpu"
    freeze_vision: bool = True
    require_transfer_source_gate: bool = True


@dataclass(frozen=True)
class FullSMBTransferResult:
    """Loaded Full SMB policy, perception, and transfer checkpoint metadata."""

    model: torch.nn.Module
    vision: FullSMBSegmentationVision
    checkpoint: dict[str, Any]
    source_checkpoint: dict[str, Any]
    source_policy_path: Path
    source_vision_path: Optional[Path]
    full_smb_vision_path: Optional[Path]
    output_path: Optional[Path]
    missing_model_keys: tuple[str, ...]
    source_transfer_gate: Mapping[str, Any]


@dataclass(frozen=True)
class FullSMBActionSelection:
    action: int
    action_name: str
    logits: torch.Tensor


def make_full_smb_policy_model(
    *,
    architecture_name: str = BASELINE_ARCHITECTURE_NAME,
    architecture_config: Optional[Mapping[str, Any]] = None,
    hidden_dim: Optional[int] = None,
    controller_schedule: str = "constant",
) -> torch.nn.Module:
    """Build a registered policy architecture under the Full SMB spec."""

    values = dict(architecture_config or {})
    if hidden_dim is not None:
        values.setdefault("hidden_dim", hidden_dim)
    if architecture_name == BASELINE_ARCHITECTURE_NAME or "controller_schedule" in values:
        values.setdefault("controller_schedule", controller_schedule)
    architecture = get_architecture(architecture_name)
    if architecture.output_contract not in POLICY_TUPLE_OUTPUT_CONTRACTS:
        raise ValueError(
            "Full SMB policy transfer requires a trainer-compatible architecture "
            f"output contract in {POLICY_TUPLE_OUTPUT_CONTRACTS!r}, got "
            f"{architecture.output_contract!r}"
        )
    return build_architecture(
        architecture_name,
        FULL_SMB_SPEC,
        values,
    )


def transfer_block_smb_checkpoint_to_full_smb(
    block_policy_checkpoint: Path,
    *,
    output_checkpoint: Optional[Path] = None,
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    block_vision_checkpoint: Optional[Path] = DEFAULT_BLOCK_VIT_CHECKPOINT,
    device: str | torch.device = "cpu",
    freeze_vision: bool = True,
    require_transfer_source_gate: bool = True,
) -> FullSMBTransferResult:
    """Load Block SMB policy weights and save a Full SMB transfer checkpoint.

    The actor/world-model/critic weights transfer because Block SMB and Full SMB
    share hierarchy lengths and the `SMBAction` vocabulary. Block ViT weights
    are validated for provenance but are not reused directly because Full SMB
    uses a different semantic vocabulary; the returned stage vision is the Full
    SMB ViT checkpoint.
    """

    source_path = Path(block_policy_checkpoint)
    source_checkpoint = _load_block_policy_source(
        source_path,
        map_location=device,
        require_transfer_source_gate=require_transfer_source_gate,
    )
    source_transfer_gate = block_smb_checkpoint_transfer_source_gate(source_checkpoint)
    architecture_name, architecture_config = policy_architecture_from_checkpoint(source_checkpoint)
    model = make_full_smb_policy_model(
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    ).to(device)
    model_state, skipped_world_model_keys = action_level_world_model_state_dict(
        model,
        source_checkpoint["states"]["model"],
    )
    load_result = model.load_state_dict(model_state, strict=False)
    missing_keys = _validate_policy_load_result(load_result)
    if skipped_world_model_keys:
        missing_keys = tuple((*missing_keys, *skipped_world_model_keys))
    model.eval()

    source_vision_path = None
    if block_vision_checkpoint is not None:
        source_vision = load_block_vit_checkpoint(
            block_vision_checkpoint,
            device=device,
            freeze=True,
        )
        source_vision_path = source_vision.path

    vision = FullSMBSegmentationVision(
        checkpoint=full_smb_vision_checkpoint,
        device=device,
        freeze=freeze_vision,
    )
    checkpoint = _build_transfer_checkpoint(
        model,
        vision,
        source_checkpoint=source_checkpoint,
        source_policy_path=source_path,
        source_vision_path=source_vision_path,
        full_smb_vision_path=vision.checkpoint_path,
        architecture_name=architecture_name,
        architecture_config=architecture_config,
        missing_keys=missing_keys,
        source_transfer_gate=source_transfer_gate,
    )

    output_path = Path(output_checkpoint) if output_checkpoint is not None else None
    if output_path is not None:
        save_checkpoint(output_path, checkpoint)

    return FullSMBTransferResult(
        model=model,
        vision=vision,
        checkpoint=checkpoint,
        source_checkpoint=source_checkpoint,
        source_policy_path=source_path,
        source_vision_path=source_vision_path,
        full_smb_vision_path=vision.checkpoint_path,
        output_path=output_path,
        missing_model_keys=missing_keys,
        source_transfer_gate=source_transfer_gate,
    )


def load_transferred_full_smb_policy(
    checkpoint_path: Path,
    *,
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    device: str | torch.device = "cpu",
    freeze_vision: bool = True,
) -> FullSMBTransferResult:
    """Load a checkpoint produced by `transfer_block_smb_checkpoint_to_full_smb`."""

    path = Path(checkpoint_path)
    checkpoint = load_checkpoint(path, map_location=device)
    _validate_transfer_checkpoint(checkpoint, path)
    architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
    model = make_full_smb_policy_model(
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    ).to(device)
    model_state, skipped_world_model_keys = action_level_world_model_state_dict(
        model,
        checkpoint["states"]["model"],
    )
    load_result = model.load_state_dict(model_state, strict=False)
    missing_keys = _validate_policy_load_result(load_result)
    if skipped_world_model_keys:
        missing_keys = tuple((*missing_keys, *skipped_world_model_keys))
    model.eval()
    vision = FullSMBSegmentationVision(
        checkpoint=full_smb_vision_checkpoint,
        device=device,
        freeze=freeze_vision,
    )
    metadata = checkpoint.get("metadata", {})
    source = metadata.get("source", {}) if isinstance(metadata, Mapping) else {}
    return FullSMBTransferResult(
        model=model,
        vision=vision,
        checkpoint=checkpoint,
        source_checkpoint={},
        source_policy_path=Path(source.get("policy_checkpoint", "")),
        source_vision_path=_optional_path(source.get("block_vision_checkpoint")),
        full_smb_vision_path=vision.checkpoint_path,
        output_path=path,
        missing_model_keys=missing_keys,
        source_transfer_gate=dict(metadata.get("source_transfer_gate", {})),
    )


@torch.no_grad()
def select_transferred_full_smb_action(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    deterministic: bool = True,
    device: str | torch.device = "cpu",
) -> FullSMBActionSelection:
    """Select an SMB action from a Full SMB batch with transferred policy weights."""

    model.eval()
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    *_prefix, logits_a = model(
        src_a,
        src_b,
        src_c,
        tau=1.0,
        episode_mask=episode_mask,
    )[:5]
    action_logits = logits_a[:, -1, : len(SMB_ACTIONS)]
    if deterministic:
        action_tensor = action_logits.argmax(dim=-1)
    else:
        distribution = torch.distributions.Categorical(logits=action_logits)
        action_tensor = distribution.sample()
    action = int(action_tensor.item())
    return FullSMBActionSelection(
        action=action,
        action_name=SMB_ACTIONS[action].name,
        logits=action_logits.detach().cpu(),
    )


def _load_block_policy_source(
    path: Path,
    *,
    map_location: str | torch.device,
    require_transfer_source_gate: bool = True,
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["stage"] != BLOCK_SMB_SPEC.name:
        raise ValueError(
            f"source policy stage must be {BLOCK_SMB_SPEC.name!r}, " f"got {checkpoint['stage']!r}"
        )
    if checkpoint["model_name"] != BLOCK_SMB_MODEL_NAME:
        raise ValueError(
            f"source policy model must be {BLOCK_SMB_MODEL_NAME!r}, "
            f"got {checkpoint['model_name']!r}"
        )
    if checkpoint["checkpoint_kind"] != BLOCK_SMB_CHECKPOINT_KIND:
        raise ValueError(
            f"source policy checkpoint kind must be {BLOCK_SMB_CHECKPOINT_KIND!r}, "
            f"got {checkpoint['checkpoint_kind']!r}"
        )
    architecture = _validate_policy_checkpoint_architecture(
        checkpoint,
        expected_stage=BLOCK_SMB_SPEC,
        checkpoint_kind=BLOCK_SMB_CHECKPOINT_KIND,
        context="source policy checkpoint",
    )
    if not architecture.supports_stage(FULL_SMB_SPEC):
        raise ValueError(
            "source policy checkpoint architecture "
            f"{architecture.name!r} does not support transfer to {FULL_SMB_SPEC.name!r}"
        )
    _validate_transfer_dimensions(checkpoint)
    transfer_gate = block_smb_checkpoint_transfer_source_gate(checkpoint)
    if require_transfer_source_gate and not transfer_gate["transfer_source_gate_met"]:
        reasons = ", ".join(transfer_gate["failure_reasons"])
        raise ValueError(
            "Block SMB checkpoint is not eligible for Full SMB transfer: "
            f"{reasons or 'transfer source gate failed'}"
        )
    return checkpoint


def block_smb_checkpoint_transfer_source_gate(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    """Return fixed plus Monte Carlo transfer-source gate diagnostics."""

    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    fixed_pass_rate = _optional_float(metrics.get("eval_threshold_pass_rate"))
    fixed_gate_met = fixed_pass_rate is not None and fixed_pass_rate >= 1.0
    semantic_gate = _optional_float(metrics.get("semantic_prediction_gate_met"))
    if semantic_gate is None:
        semantic_gate = _optional_float(
            metrics.get("eval_success_thresholds_met_after_semantic_gate")
        )
    semantic_gate_met = semantic_gate is None or semantic_gate >= 1.0
    monte_carlo_validation_samples = int(config.get("monte_carlo_validation_samples", 0) or 0)
    monte_carlo_gate_metric = _optional_float(metrics.get("eval_monte_carlo_validation_gate_met"))
    monte_carlo_gate_met = (
        monte_carlo_validation_samples > 0
        and monte_carlo_gate_metric is not None
        and monte_carlo_gate_metric >= 1.0
    )
    fixed_action_counts = _action_counts_from_metric_prefix(metrics, "eval_fixed")
    fixed_action_collapse = _all_noop_action_collapse_from_metrics(
        metrics,
        "eval_fixed",
        fixed_action_counts,
    )
    fixed_action_gate_met = fixed_action_collapse is False
    monte_carlo_validation_action_counts = _action_counts_from_metric_prefix(
        metrics,
        "eval_monte_carlo_validation",
    )
    monte_carlo_validation_action_collapse = _all_noop_action_collapse_from_metrics(
        metrics,
        "eval_monte_carlo_validation",
        monte_carlo_validation_action_counts,
    )
    monte_carlo_validation_action_gate_met = monte_carlo_validation_action_collapse is False
    failure_reasons: list[str] = []
    if not fixed_gate_met:
        failure_reasons.append("fixed scenario threshold pass rate is missing or below 1.0")
    if not semantic_gate_met:
        failure_reasons.append("Block SMB semantic prediction gate is not met")
    if not monte_carlo_gate_met:
        failure_reasons.append("held-out Monte Carlo validation gate is missing or failed")
    if fixed_action_collapse is None:
        failure_reasons.append("fixed deterministic action counts are missing")
    elif fixed_action_collapse:
        failure_reasons.append("fixed deterministic policy collapsed to all NOOP actions")
    if monte_carlo_validation_samples > 0:
        if monte_carlo_validation_action_collapse is None:
            failure_reasons.append("Monte Carlo validation deterministic action counts are missing")
        elif monte_carlo_validation_action_collapse:
            failure_reasons.append(
                "Monte Carlo validation deterministic policy collapsed to all NOOP actions"
            )
    return {
        "fixed_threshold_pass_rate": fixed_pass_rate,
        "fixed_gate_met": bool(fixed_gate_met),
        "fixed_action_counts": fixed_action_counts,
        "fixed_all_noop_action_collapse": fixed_action_collapse,
        "fixed_action_collapse_gate_met": bool(fixed_action_gate_met),
        "semantic_prediction_gate_met": bool(semantic_gate_met),
        "monte_carlo_distribution_id": str(
            config.get("monte_carlo_distribution_id", DEFAULT_BLOCK_SMB_MC_DISTRIBUTION_ID)
        ),
        "monte_carlo_validation_samples": monte_carlo_validation_samples,
        "monte_carlo_validation_success_rate": _optional_float(
            metrics.get("eval_monte_carlo_validation_success_rate")
        ),
        "monte_carlo_validation_gate_met": bool(monte_carlo_gate_met),
        "monte_carlo_validation_action_counts": monte_carlo_validation_action_counts,
        "monte_carlo_validation_all_noop_action_collapse": (monte_carlo_validation_action_collapse),
        "monte_carlo_validation_action_collapse_gate_met": bool(
            monte_carlo_validation_action_gate_met
        ),
        "transfer_source_gate_met": bool(
            fixed_gate_met
            and fixed_action_gate_met
            and semantic_gate_met
            and monte_carlo_gate_met
            and monte_carlo_validation_action_gate_met
        ),
        "failure_reasons": failure_reasons,
    }


def _action_counts_from_metric_prefix(
    metrics: Mapping[str, Any],
    prefix: str,
    *,
    action_count: int = 6,
) -> dict[str, float]:
    counts: dict[str, float] = {}
    for action_index in range(action_count):
        value = _optional_float(metrics.get(f"{prefix}_action_count_{action_index}"))
        if value is not None:
            counts[str(action_index)] = value
    return counts


def _all_noop_action_collapse_from_metrics(
    metrics: Mapping[str, Any],
    prefix: str,
    action_counts: Mapping[str, Any],
) -> Optional[bool]:
    total = 0.0
    noop = 0.0
    saw_count = False
    for raw_action, raw_count in action_counts.items():
        try:
            action = int(raw_action)
            count = float(raw_count)
        except (TypeError, ValueError):
            continue
        if count <= 0.0:
            continue
        saw_count = True
        total += count
        if action == 0:
            noop += count
    if saw_count:
        return total > 0.0 and noop == total
    marker = _optional_float(metrics.get(f"{prefix}_all_noop_action_collapse"))
    if marker is None:
        return None
    return marker >= 1.0


def _validate_policy_checkpoint_architecture(
    checkpoint: Mapping[str, Any],
    *,
    expected_stage: Any,
    checkpoint_kind: str,
    context: str,
) -> Any:
    architecture = _checkpoint_architecture_spec(checkpoint)
    validate_checkpoint_compatibility(
        checkpoint,
        stage=expected_stage,
        architecture=architecture,
        checkpoint_kind=checkpoint_kind,
        required_states=("model",),
        context=context,
    )
    if architecture.output_contract not in POLICY_TUPLE_OUTPUT_CONTRACTS:
        raise ValueError(
            f"{context} requires a trainer-compatible architecture output contract "
            f"in {POLICY_TUPLE_OUTPUT_CONTRACTS!r}, got {architecture.output_contract!r}"
        )
    return architecture


def _checkpoint_architecture_spec(checkpoint: Mapping[str, Any]) -> Any:
    architecture = checkpoint.get("architecture", {})
    if isinstance(architecture, Mapping) and architecture.get("name"):
        return get_architecture(str(architecture["name"]))
    config = checkpoint.get("config", {})
    if isinstance(config, Mapping) and config.get("architecture_name"):
        return get_architecture(str(config["architecture_name"]))
    return BASELINE_ARCHITECTURE_SPEC


def _validate_transfer_dimensions(checkpoint: Mapping[str, Any]) -> None:
    specs = checkpoint.get("specs", {})
    stage = specs.get("stage", {}) if isinstance(specs, Mapping) else {}
    if not isinstance(stage, Mapping) or not stage:
        return
    expected = {
        "seq_len_a": FULL_SMB_SPEC.seq_len_a,
        "seq_len_b": FULL_SMB_SPEC.seq_len_b,
        "seq_len_c": FULL_SMB_SPEC.seq_len_c,
        "ratio_bc": FULL_SMB_SPEC.ratio_bc,
        "vocab_size": FULL_SMB_SPEC.vocab_size,
        "action_count": FULL_SMB_SPEC.action_count,
    }
    for key, value in expected.items():
        if int(stage.get(key, value)) != value:
            raise ValueError(
                "Block SMB policy checkpoint cannot transfer to Full SMB: "
                f"{key}={stage.get(key)!r} does not match {value!r}"
            )


def policy_architecture_from_checkpoint(
    checkpoint: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Resolve policy architecture identity from new or legacy checkpoint config."""

    architecture = checkpoint.get("architecture", {})
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        config = {}
    if isinstance(architecture, Mapping) and architecture.get("name"):
        architecture_name = str(architecture["name"])
        architecture_config = architecture.get("config", config.get("architecture_config"))
    else:
        architecture_name = str(config.get("architecture_name", BASELINE_ARCHITECTURE_NAME))
        architecture_config = config.get("architecture_config")
    if isinstance(architecture_config, Mapping):
        resolved = dict(architecture_config)
        if "hidden_dim" in resolved:
            resolved["hidden_dim"] = int(resolved["hidden_dim"])
            state = checkpoint.get("states", {}).get("model", {})
            if isinstance(state, Mapping) and "agent.embedding.weight" in state:
                state_hidden_dim = _source_hidden_dim(checkpoint)
                if int(resolved["hidden_dim"]) != state_hidden_dim:
                    raise ValueError(
                        "source policy architecture hidden_dim does not match model state: "
                        f"{resolved['hidden_dim']!r} != {state_hidden_dim!r}"
                    )
        elif architecture_name == BASELINE_ARCHITECTURE_NAME:
            resolved["hidden_dim"] = _source_hidden_dim(checkpoint)
        if "controller_schedule" in resolved:
            resolved["controller_schedule"] = str(resolved["controller_schedule"])
        elif architecture_name == BASELINE_ARCHITECTURE_NAME:
            resolved["controller_schedule"] = _source_controller_schedule(checkpoint)
    else:
        model_config = config.get("model", {})
        if not isinstance(model_config, Mapping):
            model_config = {}
        resolved = {}
        if "hidden_dim" in model_config:
            resolved["hidden_dim"] = int(model_config["hidden_dim"])
        elif "hidden_dim" in config:
            resolved["hidden_dim"] = int(config["hidden_dim"])
        else:
            resolved["hidden_dim"] = _source_hidden_dim(checkpoint)
        resolved["controller_schedule"] = str(
            model_config.get(
                "controller_schedule",
                config.get("controller_schedule", "constant"),
            )
        )
    return architecture_name, resolved


def policy_architecture_metadata(
    architecture_name: str,
    architecture_config: Mapping[str, Any],
) -> dict[str, Any]:
    architecture = get_architecture(architecture_name)
    return {
        "name": architecture.name,
        "config": dict(architecture_config),
        "spec": architecture.metadata(),
    }


def _source_hidden_dim(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint["states"]["model"]
    if "agent.embedding.weight" not in state:
        raise ValueError("source policy state is missing agent.embedding.weight")
    state_hidden_dim = int(state["agent.embedding.weight"].shape[1])
    config = checkpoint.get("config", {})
    config_hidden_dim = config.get("hidden_dim") if isinstance(config, Mapping) else None
    if config_hidden_dim is not None and int(config_hidden_dim) != state_hidden_dim:
        raise ValueError(
            "source policy hidden_dim does not match model state: "
            f"{config_hidden_dim!r} != {state_hidden_dim!r}"
        )
    return state_hidden_dim


def _source_controller_schedule(checkpoint: Mapping[str, Any]) -> str:
    config = checkpoint.get("config", {})
    if not isinstance(config, Mapping):
        return "constant"
    return str(config.get("controller_schedule", "constant"))


def _validate_policy_load_result(load_result: Any) -> tuple[str, ...]:
    allowed_missing_prefixes = (
        "transition_representation_head.",
        "reward_head.",
        "value_head.",
        *ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    )
    unexpected = tuple(load_result.unexpected_keys)
    unsupported_missing = tuple(
        key for key in load_result.missing_keys if not key.startswith(allowed_missing_prefixes)
    )
    if unexpected or unsupported_missing:
        raise ValueError(
            "transferred policy state is incompatible with Full SMB policy model; "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    return tuple(load_result.missing_keys)


def _build_transfer_checkpoint(
    model: torch.nn.Module,
    vision: FullSMBSegmentationVision,
    *,
    source_checkpoint: Mapping[str, Any],
    source_policy_path: Path,
    source_vision_path: Optional[Path],
    full_smb_vision_path: Optional[Path],
    architecture_name: str,
    architecture_config: Mapping[str, Any],
    missing_keys: tuple[str, ...],
    source_transfer_gate: Mapping[str, Any],
) -> dict[str, Any]:
    hidden_dim = architecture_config.get("hidden_dim")
    controller_schedule = architecture_config.get("controller_schedule")
    architecture = policy_architecture_metadata(architecture_name, architecture_config)
    model_config = {
        "seq_len_a": FULL_SMB_SPEC.seq_len_a,
        "seq_len_b": FULL_SMB_SPEC.seq_len_b,
        "seq_len_c": FULL_SMB_SPEC.seq_len_c,
        "ratio_bc": FULL_SMB_SPEC.ratio_bc,
        "vocab_size": FULL_SMB_SPEC.vocab_size,
        "action_count": FULL_SMB_SPEC.action_count,
    }
    if hidden_dim is not None:
        model_config["hidden_dim"] = int(hidden_dim)
    if controller_schedule is not None:
        model_config["controller_schedule"] = str(controller_schedule)
    return build_checkpoint(
        stage=FULL_SMB_SPEC.name,
        model_name=FULL_SMB_TRANSFER_MODEL_NAME,
        checkpoint_kind=FULL_SMB_TRANSFER_CHECKPOINT_KIND,
        epoch=int(source_checkpoint.get("epoch", 0)),
        global_step=int(source_checkpoint.get("global_step", 0)),
        metrics={},
        config={
            "architecture_name": architecture_name,
            "architecture_config": dict(architecture_config),
            "model": model_config,
            "source": source_checkpoint.get("config", {}),
        },
        specs={
            "stage": {
                "name": FULL_SMB_SPEC.name,
                "seq_len_a": FULL_SMB_SPEC.seq_len_a,
                "seq_len_b": FULL_SMB_SPEC.seq_len_b,
                "seq_len_c": FULL_SMB_SPEC.seq_len_c,
                "ratio_bc": FULL_SMB_SPEC.ratio_bc,
                "vocab_size": FULL_SMB_SPEC.vocab_size,
                "action_count": FULL_SMB_SPEC.action_count,
            },
            "vision": asdict(vision.spec),
            "architecture": architecture["spec"],
            "architecture_config": architecture["config"],
        },
        states={"model": model.state_dict()},
        metadata={
            "architecture": architecture,
            "source": {
                "policy_checkpoint": str(source_policy_path),
                "policy_stage": source_checkpoint["stage"],
                "policy_model_name": source_checkpoint["model_name"],
                "policy_checkpoint_kind": source_checkpoint["checkpoint_kind"],
                "block_vision_checkpoint": (
                    str(source_vision_path) if source_vision_path is not None else None
                ),
                "full_smb_vision_checkpoint": (
                    str(full_smb_vision_path) if full_smb_vision_path is not None else None
                ),
            },
            "source_metrics": source_checkpoint.get("metrics", {}),
            "source_transfer_gate": dict(source_transfer_gate),
            "missing_model_keys": missing_keys,
            "transfer_note": (
                "Actor/world-model/critic weights are reused because the Full SMB "
                "and Block SMB hierarchy specs match. Block ViT perception weights "
                "are not reused directly; Full SMB uses the Full SMB ViT classes."
            ),
        },
    )


def _validate_transfer_checkpoint(checkpoint: Mapping[str, Any], path: Path) -> None:
    if checkpoint["stage"] != FULL_SMB_SPEC.name:
        raise ValueError(f"{path} is not a Full SMB checkpoint")
    if checkpoint["model_name"] != FULL_SMB_TRANSFER_MODEL_NAME:
        raise ValueError(f"{path} is not a transferred Full SMB policy checkpoint")
    if checkpoint["checkpoint_kind"] != FULL_SMB_TRANSFER_CHECKPOINT_KIND:
        raise ValueError(f"{path} has unsupported checkpoint kind")
    _validate_policy_checkpoint_architecture(
        checkpoint,
        expected_stage=FULL_SMB_SPEC,
        checkpoint_kind=FULL_SMB_TRANSFER_CHECKPOINT_KIND,
        context=str(path),
    )


def _optional_path(value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(str(value))


def _optional_float(value: Any) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not torch.isfinite(torch.tensor(parsed)).item():
        return None
    return parsed


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-policy-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument(
        "--block-vision-checkpoint",
        type=Path,
        default=DEFAULT_BLOCK_VIT_CHECKPOINT,
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fine-tune-vision", action="store_true")
    parser.add_argument(
        "--allow-ungated-block-source",
        action="store_true",
        help="bypass fixed plus Monte Carlo Block SMB transfer-source gates",
    )
    args = parser.parse_args(argv)

    result = transfer_block_smb_checkpoint_to_full_smb(
        args.block_policy_checkpoint,
        output_checkpoint=args.output_checkpoint,
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        block_vision_checkpoint=args.block_vision_checkpoint,
        device=args.device,
        freeze_vision=not args.fine_tune_vision,
        require_transfer_source_gate=not args.allow_ungated_block_source,
    )
    print(
        "Transferred Block SMB policy to Full SMB: "
        f"{result.source_policy_path} -> {result.output_path}"
    )


if __name__ == "__main__":
    main()
