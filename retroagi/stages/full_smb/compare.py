"""Compare transferred Full SMB policies against scratch baselines."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

from retroagi.core import SMB_ACTIONS, StageBatch, load_checkpoint
from retroagi.stages.full_smb.adapter import FullSMBStage
from retroagi.stages.full_smb.transfer import (
    load_transferred_full_smb_policy,
    make_full_smb_policy_model,
    policy_architecture_from_checkpoint,
)
from retroagi.stages.full_smb.vision import DEFAULT_FULL_SMB_VIT_CHECKPOINT


@dataclass(frozen=True)
class FullSMBPolicyComparisonConfig:
    steps: int = 128
    seed: int = 0
    scratch_seed: int = 0
    device: str = "cpu"
    reset_on_done: bool = True
    render: bool = False

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")


@dataclass(frozen=True)
class FullSMBPolicyComparisonResult:
    seed: int
    scratch_seed: int
    requested_steps: int
    evaluated_steps: int
    action_agreement: float
    transfer_action_histogram: Mapping[str, int]
    scratch_action_histogram: Mapping[str, int]
    mean_transfer_entropy: float
    mean_scratch_entropy: float
    mean_transfer_margin: float
    mean_scratch_margin: float
    collection_reward: float
    resets: int
    completed_episodes: int
    terminated_count: int
    truncated_count: int
    transfer_checkpoint: str
    scratch_checkpoint: Optional[str]
    scratch_source: str
    full_smb_vision_checkpoint: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def compare_transferred_checkpoint_with_scratch(
    transfer_checkpoint: Path,
    *,
    make_stage: Callable[[Any], FullSMBStage],
    scratch_checkpoint: Optional[Path] = None,
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    config: FullSMBPolicyComparisonConfig = FullSMBPolicyComparisonConfig(),
) -> FullSMBPolicyComparisonResult:
    """Compare a transferred Full SMB policy with a scratch policy baseline.

    The environment trajectory is driven by seeded random actions so both
    policies are evaluated on exactly the same `FullSMBStage` batches.
    """

    transfer = load_transferred_full_smb_policy(
        transfer_checkpoint,
        full_smb_vision_checkpoint=full_smb_vision_checkpoint,
        device=config.device,
    )
    scratch_model, scratch_source = _scratch_model_for_transfer(
        transfer.checkpoint,
        scratch_checkpoint=scratch_checkpoint,
        scratch_seed=config.scratch_seed,
        device=config.device,
    )
    stage = make_stage(transfer.vision)
    try:
        return compare_full_smb_policies_on_stage(
            transfer.model,
            scratch_model,
            stage,
            transfer_checkpoint=Path(transfer_checkpoint),
            scratch_checkpoint=Path(scratch_checkpoint) if scratch_checkpoint is not None else None,
            scratch_source=scratch_source,
            full_smb_vision_checkpoint=transfer.full_smb_vision_path,
            config=config,
        )
    finally:
        stage.close()


def compare_full_smb_policies_on_stage(
    transfer_model: torch.nn.Module,
    scratch_model: torch.nn.Module,
    stage: FullSMBStage,
    *,
    transfer_checkpoint: Path,
    scratch_checkpoint: Optional[Path],
    scratch_source: str,
    full_smb_vision_checkpoint: Optional[Path],
    config: FullSMBPolicyComparisonConfig = FullSMBPolicyComparisonConfig(),
) -> FullSMBPolicyComparisonResult:
    """Evaluate two Full SMB policies on the same seeded observation stream."""

    rng = random.Random(config.seed)
    observation = stage.reset(seed=config.seed)
    resets = 1
    completed_episodes = 0
    terminated_count = 0
    truncated_count = 0
    collection_reward = 0.0
    transfer_actions: list[int] = []
    scratch_actions: list[int] = []
    transfer_entropies: list[float] = []
    scratch_entropies: list[float] = []
    transfer_margins: list[float] = []
    scratch_margins: list[float] = []

    for step_index in range(config.steps):
        batch = stage.encode_observation(observation)
        transfer_logits = _policy_action_logits(
            transfer_model,
            batch,
            device=config.device,
            torch_seed=config.seed * 1_000_003 + step_index,
        )
        scratch_logits = _policy_action_logits(
            scratch_model,
            batch,
            device=config.device,
            torch_seed=config.seed * 1_000_003 + step_index,
        )
        transfer_actions.append(int(transfer_logits.argmax(dim=-1).item()))
        scratch_actions.append(int(scratch_logits.argmax(dim=-1).item()))
        transfer_entropies.append(_entropy(transfer_logits))
        scratch_entropies.append(_entropy(scratch_logits))
        transfer_margins.append(_top_margin(transfer_logits))
        scratch_margins.append(_top_margin(scratch_logits))

        action = rng.choice(SMB_ACTIONS)
        observation, reward, terminated, truncated, _info = stage.step(action)
        collection_reward += float(reward)
        if config.render:
            render = getattr(stage.env, "render", None)
            if render is not None:
                render()
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if terminated or truncated:
            completed_episodes += 1
            if config.reset_on_done and step_index + 1 < config.steps:
                observation = stage.reset(seed=config.seed + resets)
                resets += 1

    evaluated_steps = len(transfer_actions)
    agreement = (
        sum(
            1
            for transfer_action, scratch_action in zip(transfer_actions, scratch_actions)
            if transfer_action == scratch_action
        )
        / evaluated_steps
        if evaluated_steps
        else 0.0
    )
    return FullSMBPolicyComparisonResult(
        seed=config.seed,
        scratch_seed=config.scratch_seed,
        requested_steps=config.steps,
        evaluated_steps=evaluated_steps,
        action_agreement=float(agreement),
        transfer_action_histogram=_action_histogram(transfer_actions),
        scratch_action_histogram=_action_histogram(scratch_actions),
        mean_transfer_entropy=_mean(transfer_entropies),
        mean_scratch_entropy=_mean(scratch_entropies),
        mean_transfer_margin=_mean(transfer_margins),
        mean_scratch_margin=_mean(scratch_margins),
        collection_reward=float(collection_reward),
        resets=resets,
        completed_episodes=completed_episodes,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
        transfer_checkpoint=str(transfer_checkpoint),
        scratch_checkpoint=str(scratch_checkpoint) if scratch_checkpoint else None,
        scratch_source=scratch_source,
        full_smb_vision_checkpoint=(
            str(full_smb_vision_checkpoint) if full_smb_vision_checkpoint else None
        ),
    )


def save_full_smb_policy_comparison(path: Path, result: FullSMBPolicyComparisonResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _scratch_model_for_transfer(
    transfer_checkpoint: Mapping[str, Any],
    *,
    scratch_checkpoint: Optional[Path],
    scratch_seed: int,
    device: str | torch.device,
) -> tuple[torch.nn.Module, str]:
    if scratch_checkpoint is not None:
        checkpoint = load_checkpoint(Path(scratch_checkpoint), map_location=device)
        architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
        model = make_full_smb_policy_model(
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        ).to(device)
        load_result = model.load_state_dict(checkpoint["states"]["model"], strict=False)
        if load_result.unexpected_keys or load_result.missing_keys:
            raise ValueError(
                "scratch checkpoint is incompatible with Full SMB policy model; "
                f"missing={tuple(load_result.missing_keys)}, "
                f"unexpected={tuple(load_result.unexpected_keys)}"
            )
        model.eval()
        return model, "checkpoint"

    architecture_name, architecture_config = policy_architecture_from_checkpoint(
        transfer_checkpoint
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(scratch_seed)
        model = make_full_smb_policy_model(
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        ).to(device)
    model.eval()
    return model, "scratch_initialization"


@torch.no_grad()
def _policy_action_logits(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    device: str | torch.device,
    torch_seed: int,
) -> torch.Tensor:
    src_a = batch.src_a.to(device)
    src_b = batch.src_b.to(device)
    src_c = batch.src_c.to(device)
    episode = (batch.metadata or {}).get("episode", {})
    episode_mask = episode.get("mask") if isinstance(episode, Mapping) else None
    if episode_mask is not None:
        episode_mask = torch.as_tensor(episode_mask, dtype=src_c.dtype, device=src_c.device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(torch_seed)
        outputs = model(
            src_a,
            src_b,
            src_c,
            tau=1.0,
            episode_mask=episode_mask,
        )
    logits_a = outputs[4]
    return logits_a[:, -1, : len(SMB_ACTIONS)].detach().cpu()


def _entropy(logits: torch.Tensor) -> float:
    distribution = torch.distributions.Categorical(logits=logits)
    return float(distribution.entropy().mean().item())


def _top_margin(logits: torch.Tensor) -> float:
    top = logits.topk(k=2, dim=-1).values
    return float((top[:, 0] - top[:, 1]).mean().item())


def _action_histogram(actions: list[int]) -> dict[str, int]:
    counts = {action.name: 0 for action in SMB_ACTIONS}
    for action_id in actions:
        counts[SMB_ACTIONS[action_id].name] += 1
    return counts


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--scratch-checkpoint", type=Path)
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scratch-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = FullSMBPolicyComparisonConfig(
        steps=args.steps,
        seed=args.seed,
        scratch_seed=args.scratch_seed,
        device=args.device,
        render=args.render,
    )

    result = compare_transferred_checkpoint_with_scratch(
        args.transfer_checkpoint,
        make_stage=lambda vision: FullSMBStage(vision=vision),
        scratch_checkpoint=args.scratch_checkpoint,
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        config=config,
    )
    if args.output is not None:
        save_full_smb_policy_comparison(args.output, result)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
