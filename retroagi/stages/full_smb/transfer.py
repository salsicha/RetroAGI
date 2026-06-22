"""Transfer Block SMB checkpoints into the Full SMB stage contract."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from retroagi.core import (
    AgentWorldModelCritic,
    SMB_ACTIONS,
    StageBatch,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from retroagi.stages.block_smb.adapter import BLOCK_SMB_SPEC
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


@dataclass(frozen=True)
class FullSMBTransferResult:
    """Loaded Full SMB policy, perception, and transfer checkpoint metadata."""

    model: AgentWorldModelCritic
    vision: FullSMBSegmentationVision
    checkpoint: dict[str, Any]
    source_checkpoint: dict[str, Any]
    source_policy_path: Path
    source_vision_path: Optional[Path]
    full_smb_vision_path: Optional[Path]
    output_path: Optional[Path]
    missing_model_keys: tuple[str, ...]


@dataclass(frozen=True)
class FullSMBActionSelection:
    action: int
    action_name: str
    logits: torch.Tensor


def make_full_smb_policy_model(
    *,
    hidden_dim: int,
    controller_schedule: str = "constant",
) -> AgentWorldModelCritic:
    """Build the shared actor/world-model/critic under the Full SMB spec."""

    return AgentWorldModelCritic(
        FULL_SMB_SPEC.vocab_size,
        FULL_SMB_SPEC.seq_len_a,
        FULL_SMB_SPEC.seq_len_c,
        FULL_SMB_SPEC.ratio_bc,
        d_model=hidden_dim,
        controller_schedule=controller_schedule,
    )


def transfer_block_smb_checkpoint_to_full_smb(
    block_policy_checkpoint: Path,
    *,
    output_checkpoint: Optional[Path] = None,
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    block_vision_checkpoint: Optional[Path] = DEFAULT_BLOCK_VIT_CHECKPOINT,
    device: str | torch.device = "cpu",
    freeze_vision: bool = True,
) -> FullSMBTransferResult:
    """Load Block SMB policy weights and save a Full SMB transfer checkpoint.

    The actor/world-model/critic weights transfer because Block SMB and Full SMB
    share hierarchy lengths and the `SMBAction` vocabulary. Block ViT weights
    are validated for provenance but are not reused directly because Full SMB
    uses a different semantic vocabulary; the returned stage vision is the Full
    SMB ViT checkpoint.
    """

    source_path = Path(block_policy_checkpoint)
    source_checkpoint = _load_block_policy_source(source_path, map_location=device)
    hidden_dim = _source_hidden_dim(source_checkpoint)
    controller_schedule = _source_controller_schedule(source_checkpoint)
    model = make_full_smb_policy_model(
        hidden_dim=hidden_dim,
        controller_schedule=controller_schedule,
    ).to(device)
    load_result = model.load_state_dict(
        source_checkpoint["states"]["model"], strict=False
    )
    missing_keys = _validate_policy_load_result(load_result)
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
        hidden_dim=hidden_dim,
        controller_schedule=controller_schedule,
        missing_keys=missing_keys,
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
    model_config = checkpoint["config"]["model"]
    model = make_full_smb_policy_model(
        hidden_dim=int(model_config["hidden_dim"]),
        controller_schedule=str(model_config.get("controller_schedule", "constant")),
    ).to(device)
    load_result = model.load_state_dict(checkpoint["states"]["model"], strict=False)
    missing_keys = _validate_policy_load_result(load_result)
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
    )


@torch.no_grad()
def select_transferred_full_smb_action(
    model: AgentWorldModelCritic,
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
        episode_mask = torch.as_tensor(
            episode_mask, dtype=src_c.dtype, device=src_c.device
        )
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
) -> dict[str, Any]:
    checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["stage"] != BLOCK_SMB_SPEC.name:
        raise ValueError(
            f"source policy stage must be {BLOCK_SMB_SPEC.name!r}, "
            f"got {checkpoint['stage']!r}"
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
    if "model" not in checkpoint["states"]:
        raise ValueError("source policy checkpoint is missing states.model")
    _validate_transfer_dimensions(checkpoint)
    return checkpoint


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
    }
    for key, value in expected.items():
        if int(stage.get(key, value)) != value:
            raise ValueError(
                "Block SMB policy checkpoint cannot transfer to Full SMB: "
                f"{key}={stage.get(key)!r} does not match {value!r}"
            )


def _source_hidden_dim(checkpoint: Mapping[str, Any]) -> int:
    state = checkpoint["states"]["model"]
    if "agent.embedding.weight" not in state:
        raise ValueError("source policy state is missing agent.embedding.weight")
    state_hidden_dim = int(state["agent.embedding.weight"].shape[1])
    config = checkpoint.get("config", {})
    config_hidden_dim = (
        config.get("hidden_dim") if isinstance(config, Mapping) else None
    )
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
    )
    unexpected = tuple(load_result.unexpected_keys)
    unsupported_missing = tuple(
        key
        for key in load_result.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    )
    if unexpected or unsupported_missing:
        raise ValueError(
            "transferred policy state is incompatible with Full SMB policy model; "
            f"missing={unsupported_missing}, unexpected={unexpected}"
        )
    return tuple(load_result.missing_keys)


def _build_transfer_checkpoint(
    model: AgentWorldModelCritic,
    vision: FullSMBSegmentationVision,
    *,
    source_checkpoint: Mapping[str, Any],
    source_policy_path: Path,
    source_vision_path: Optional[Path],
    full_smb_vision_path: Optional[Path],
    hidden_dim: int,
    controller_schedule: str,
    missing_keys: tuple[str, ...],
) -> dict[str, Any]:
    return build_checkpoint(
        stage=FULL_SMB_SPEC.name,
        model_name=FULL_SMB_TRANSFER_MODEL_NAME,
        checkpoint_kind=FULL_SMB_TRANSFER_CHECKPOINT_KIND,
        epoch=int(source_checkpoint.get("epoch", 0)),
        global_step=int(source_checkpoint.get("global_step", 0)),
        metrics={},
        config={
            "model": {
                "hidden_dim": hidden_dim,
                "controller_schedule": controller_schedule,
                "seq_len_a": FULL_SMB_SPEC.seq_len_a,
                "seq_len_b": FULL_SMB_SPEC.seq_len_b,
                "seq_len_c": FULL_SMB_SPEC.seq_len_c,
                "ratio_bc": FULL_SMB_SPEC.ratio_bc,
                "vocab_size": FULL_SMB_SPEC.vocab_size,
            },
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
            },
            "vision": asdict(vision.spec),
        },
        states={"model": model.state_dict()},
        metadata={
            "source": {
                "policy_checkpoint": str(source_policy_path),
                "policy_stage": source_checkpoint["stage"],
                "policy_model_name": source_checkpoint["model_name"],
                "policy_checkpoint_kind": source_checkpoint["checkpoint_kind"],
                "block_vision_checkpoint": str(source_vision_path)
                if source_vision_path is not None
                else None,
                "full_smb_vision_checkpoint": str(full_smb_vision_path)
                if full_smb_vision_path is not None
                else None,
            },
            "source_metrics": source_checkpoint.get("metrics", {}),
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
    if "model" not in checkpoint["states"]:
        raise ValueError(f"{path} is missing states.model")


def _optional_path(value: Any) -> Optional[Path]:
    if value is None or value == "":
        return None
    return Path(str(value))


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
    args = parser.parse_args(argv)

    result = transfer_block_smb_checkpoint_to_full_smb(
        args.block_policy_checkpoint,
        output_checkpoint=args.output_checkpoint,
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        block_vision_checkpoint=args.block_vision_checkpoint,
        device=args.device,
        freeze_vision=not args.fine_tune_vision,
    )
    print(
        "Transferred Block SMB policy to Full SMB: "
        f"{result.source_policy_path} -> {result.output_path}"
    )


if __name__ == "__main__":
    main()
