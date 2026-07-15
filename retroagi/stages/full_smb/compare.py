"""Compare transferred Full SMB policies against scratch baselines."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import torch

from retroagi.core import (
    ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES,
    SMB_ACTIONS,
    StageBatch,
    action_level_world_model_state_dict,
    load_checkpoint,
)
from retroagi.stages.full_smb.adapter import FullSMBEnvConfig, FullSMBStage
from retroagi.stages.full_smb.save_states import load_full_smb_save_state_payload
from retroagi.stages.full_smb.tasks import (
    FULL_SMB_TASK_SET_NAMES,
    FullSMBTaskSpec,
    full_smb_task_catalog,
)
from retroagi.stages.full_smb.train import _apply_full_smb_motor_primitive_bias
from retroagi.stages.full_smb.transfer import (
    load_transferred_full_smb_policy,
    make_full_smb_policy_model,
    policy_architecture_from_checkpoint,
)
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    FullSMBSegmentationVision,
)

PolicySuiteStageFactory = Callable[[Any, Optional[FullSMBTaskSpec]], FullSMBStage]


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
class FullSMBPolicySuiteComparisonConfig:
    steps: int = 128
    seeds: tuple[int, ...] = (0,)
    scratch_seed: int = 0
    device: str = "cpu"
    reset_on_done: bool = True
    render: bool = False

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise ValueError("steps must be non-negative")
        if not self.seeds:
            raise ValueError("seeds must contain at least one seed")
        object.__setattr__(self, "steps", int(self.steps))
        object.__setattr__(self, "seeds", tuple(int(seed) for seed in self.seeds))
        object.__setattr__(self, "scratch_seed", int(self.scratch_seed))


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


@dataclass(frozen=True)
class FullSMBPolicySuiteComparisonResult:
    """Comparison report for named Full SMB policies on identical task streams."""

    seeds: tuple[int, ...]
    task_names: tuple[Optional[str], ...]
    requested_steps_per_stream: int
    evaluated_steps: int
    policies: Mapping[str, Mapping[str, Any]]
    streams: tuple[Mapping[str, Any], ...]
    aggregate_pairwise: Mapping[str, Mapping[str, Any]]
    full_smb_vision_checkpoint: Optional[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class _LoadedComparisonPolicy:
    name: str
    model: torch.nn.Module
    checkpoint_path: Optional[Path]
    source: str
    architecture_name: str
    architecture_config: Mapping[str, Any]
    model_name: Optional[str]


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


def compare_full_smb_policy_suite(
    transfer_checkpoint: Path,
    *,
    make_stage: PolicySuiteStageFactory,
    scratch_checkpoint: Optional[Path] = None,
    fine_tuned_checkpoint: Optional[Path] = None,
    known_good_checkpoint: Optional[Path] = None,
    extra_policy_checkpoints: Optional[Mapping[str, Path]] = None,
    full_smb_vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    config: FullSMBPolicySuiteComparisonConfig = FullSMBPolicySuiteComparisonConfig(),
    task_names: Iterable[str] = (),
    task_set: Optional[str] = None,
) -> FullSMBPolicySuiteComparisonResult:
    """Compare named Full SMB policies on identical seeded task streams."""

    reference_checkpoint = load_checkpoint(Path(transfer_checkpoint), map_location=config.device)
    policies = [
        _load_comparison_policy_from_checkpoint(
            "transferred",
            Path(transfer_checkpoint),
            source="transfer_checkpoint",
            device=config.device,
        )
    ]
    policies.append(
        _scratch_policy_for_reference(
            reference_checkpoint,
            scratch_checkpoint=scratch_checkpoint,
            scratch_seed=config.scratch_seed,
            device=config.device,
        )
    )
    if fine_tuned_checkpoint is not None:
        policies.append(
            _load_comparison_policy_from_checkpoint(
                "fine_tuned",
                Path(fine_tuned_checkpoint),
                source="fine_tuned_checkpoint",
                device=config.device,
            )
        )
    if known_good_checkpoint is not None:
        policies.append(
            _load_comparison_policy_from_checkpoint(
                "known_good",
                Path(known_good_checkpoint),
                source="known_good_checkpoint",
                device=config.device,
            )
        )
    for name, checkpoint in (extra_policy_checkpoints or {}).items():
        if name in {policy.name for policy in policies}:
            raise ValueError(f"duplicate Full SMB comparison policy name {name!r}")
        policies.append(
            _load_comparison_policy_from_checkpoint(
                name,
                Path(checkpoint),
                source="named_checkpoint",
                device=config.device,
            )
        )

    vision = FullSMBSegmentationVision(
        checkpoint=full_smb_vision_checkpoint,
        device=config.device,
        freeze=True,
    )
    tasks = _comparison_tasks(task_names=tuple(task_names), task_set=task_set)
    policy_accumulators = _empty_policy_accumulators(policies)
    stream_results: list[Mapping[str, Any]] = []

    for task in tasks:
        start_state = _comparison_task_start_state(task)
        for seed in config.seeds:
            stage = make_stage(vision, task)
            try:
                stream_results.append(
                    _compare_loaded_policies_on_stage(
                        policies,
                        stage,
                        seed=seed,
                        task=task,
                        start_state=start_state,
                        config=config,
                        accumulators=policy_accumulators,
                    )
                )
            finally:
                stage.close()

    return FullSMBPolicySuiteComparisonResult(
        seeds=config.seeds,
        task_names=tuple(task.name if task is not None else None for task in tasks),
        requested_steps_per_stream=config.steps,
        evaluated_steps=sum(int(stream["evaluated_steps"]) for stream in stream_results),
        policies=_policy_suite_metadata(policies, policy_accumulators),
        streams=tuple(stream_results),
        aggregate_pairwise=_aggregate_pairwise_agreement(policy_accumulators),
        full_smb_vision_checkpoint=(
            str(full_smb_vision_checkpoint) if full_smb_vision_checkpoint else None
        ),
    )


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


def _compare_loaded_policies_on_stage(
    policies: list[_LoadedComparisonPolicy],
    stage: FullSMBStage,
    *,
    seed: int,
    task: Optional[FullSMBTaskSpec],
    start_state: Any = None,
    config: FullSMBPolicySuiteComparisonConfig,
    accumulators: dict[str, dict[str, list[Any]]],
) -> dict[str, Any]:
    rng = random.Random(seed)
    observation = _reset_comparison_stage(stage, seed=seed, start_state=start_state)
    resets = 1
    completed_episodes = 0
    terminated_count = 0
    truncated_count = 0
    collection_reward = 0.0
    stream_actions = {policy.name: [] for policy in policies}
    stream_entropies = {policy.name: [] for policy in policies}
    stream_margins = {policy.name: [] for policy in policies}

    for step_index in range(config.steps):
        batch = stage.encode_observation(observation)
        torch_seed = seed * 1_000_003 + step_index
        for policy in policies:
            logits = _policy_action_logits(
                policy.model,
                batch,
                device=config.device,
                torch_seed=torch_seed,
            )
            action = int(logits.argmax(dim=-1).item())
            entropy = _entropy(logits)
            margin = _top_margin(logits)
            stream_actions[policy.name].append(action)
            stream_entropies[policy.name].append(entropy)
            stream_margins[policy.name].append(margin)
            accumulators[policy.name]["actions"].append(action)
            accumulators[policy.name]["entropies"].append(entropy)
            accumulators[policy.name]["margins"].append(margin)

        driver_action = rng.choice(SMB_ACTIONS)
        observation, reward, terminated, truncated, _info = stage.step(driver_action)
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
                observation = _reset_comparison_stage(
                    stage,
                    seed=seed + resets,
                    start_state=start_state,
                )
                resets += 1

    return {
        "seed": int(seed),
        "task": _comparison_task_manifest(task),
        "requested_steps": int(config.steps),
        "evaluated_steps": len(next(iter(stream_actions.values()), ())),
        "collection_reward": float(collection_reward),
        "resets": int(resets),
        "completed_episodes": int(completed_episodes),
        "terminated_count": int(terminated_count),
        "truncated_count": int(truncated_count),
        "policies": {
            policy.name: {
                "action_histogram": _action_histogram(stream_actions[policy.name]),
                "mean_entropy": _mean(stream_entropies[policy.name]),
                "mean_margin": _mean(stream_margins[policy.name]),
            }
            for policy in policies
        },
        "pairwise": _stream_pairwise_agreement(stream_actions),
    }


def _comparison_tasks(
    *,
    task_names: tuple[str, ...],
    task_set: Optional[str],
) -> tuple[Optional[FullSMBTaskSpec], ...]:
    catalog = full_smb_task_catalog()
    if task_names:
        return tuple(catalog.task(name) for name in task_names)
    if task_set is not None:
        return catalog.tasks_for_set(task_set)
    return (None,)


def _comparison_task_start_state(task: Optional[FullSMBTaskSpec]) -> Any:
    if task is None or task.start.mode != "save_state_artifact":
        return None
    path = task.start.save_state_path
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Full SMB comparison task {task.name!r} requires the local save-state "
            f"artifact {path}; create it with "
            "`python -m retroagi.stages.full_smb.save_states create` before comparing."
        )
    return load_full_smb_save_state_payload(Path(path))["state"]


def _reset_comparison_stage(
    stage: FullSMBStage,
    *,
    seed: int,
    start_state: Any,
) -> Any:
    observation = stage.reset(seed=seed)
    if start_state is None:
        return observation
    observation = stage.load_emulator_state(start_state)
    reset_frame_stack = getattr(stage, "_reset_frame_stack", None)
    if callable(reset_frame_stack):
        reset_frame_stack(observation)
    return observation


def _comparison_task_manifest(task: Optional[FullSMBTaskSpec]) -> dict[str, Any]:
    if task is None:
        return {
            "name": None,
            "task_set": None,
            "split": None,
            "state": None,
            "start_mode": None,
            "save_state_path": None,
        }
    return {
        "name": task.name,
        "task_set": task.task_set,
        "split": task.split,
        "state": task.start.state,
        "start_mode": task.start.mode,
        "save_state_path": (
            str(task.start.save_state_path) if task.start.save_state_path is not None else None
        ),
        "reset_seed": task.reset_seed,
        "max_steps": task.max_steps,
    }


def _empty_policy_accumulators(
    policies: list[_LoadedComparisonPolicy],
) -> dict[str, dict[str, list[Any]]]:
    return {
        policy.name: {
            "actions": [],
            "entropies": [],
            "margins": [],
        }
        for policy in policies
    }


def _policy_suite_metadata(
    policies: list[_LoadedComparisonPolicy],
    accumulators: Mapping[str, Mapping[str, list[Any]]],
) -> dict[str, Any]:
    return {
        policy.name: {
            "checkpoint": str(policy.checkpoint_path) if policy.checkpoint_path else None,
            "source": policy.source,
            "architecture_name": policy.architecture_name,
            "architecture_config": dict(policy.architecture_config),
            "model_name": policy.model_name,
            "action_histogram": _action_histogram(
                [int(action) for action in accumulators[policy.name]["actions"]]
            ),
            "mean_entropy": _mean(
                [float(value) for value in accumulators[policy.name]["entropies"]]
            ),
            "mean_margin": _mean([float(value) for value in accumulators[policy.name]["margins"]]),
        }
        for policy in policies
    }


def _stream_pairwise_agreement(
    actions_by_policy: Mapping[str, list[int]],
) -> dict[str, dict[str, Any]]:
    pairwise: dict[str, dict[str, Any]] = {}
    for left, right in combinations(actions_by_policy.keys(), 2):
        left_actions = actions_by_policy[left]
        right_actions = actions_by_policy[right]
        compared_steps = min(len(left_actions), len(right_actions))
        matches = sum(
            1
            for left_action, right_action in zip(left_actions, right_actions)
            if left_action == right_action
        )
        key = _pairwise_key(left, right)
        pairwise[key] = {
            "left": left,
            "right": right,
            "compared_steps": int(compared_steps),
            "matching_actions": int(matches),
            "action_agreement": float(matches / compared_steps) if compared_steps else 0.0,
        }
    return pairwise


def _aggregate_pairwise_agreement(
    accumulators: Mapping[str, Mapping[str, list[Any]]],
) -> dict[str, dict[str, Any]]:
    actions_by_policy = {
        name: [int(action) for action in values["actions"]] for name, values in accumulators.items()
    }
    return _stream_pairwise_agreement(actions_by_policy)


def _pairwise_key(left: str, right: str) -> str:
    return f"{left}_vs_{right}"


def save_full_smb_policy_comparison(
    path: Path,
    result: FullSMBPolicyComparisonResult | FullSMBPolicySuiteComparisonResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_comparison_policy_from_checkpoint(
    name: str,
    checkpoint_path: Path,
    *,
    source: str,
    device: str | torch.device,
) -> _LoadedComparisonPolicy:
    checkpoint = load_checkpoint(Path(checkpoint_path), map_location=device)
    architecture_name, architecture_config = policy_architecture_from_checkpoint(checkpoint)
    model = make_full_smb_policy_model(
        architecture_name=architecture_name,
        architecture_config=architecture_config,
    ).to(device)
    model_state, _skipped_world_model_keys = action_level_world_model_state_dict(
        model,
        checkpoint["states"]["model"],
    )
    load_result = model.load_state_dict(model_state, strict=False)
    unsupported_missing = tuple(
        key
        for key in load_result.missing_keys
        if not key.startswith(ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES)
    )
    if load_result.unexpected_keys or unsupported_missing:
        raise ValueError(
            f"{name} checkpoint is incompatible with Full SMB policy model; "
            f"missing={unsupported_missing}, "
            f"unexpected={tuple(load_result.unexpected_keys)}"
        )
    model.eval()
    return _LoadedComparisonPolicy(
        name=name,
        model=model,
        checkpoint_path=Path(checkpoint_path),
        source=source,
        architecture_name=architecture_name,
        architecture_config=architecture_config,
        model_name=checkpoint.get("model_name"),
    )


def _scratch_policy_for_reference(
    reference_checkpoint: Mapping[str, Any],
    *,
    scratch_checkpoint: Optional[Path],
    scratch_seed: int,
    device: str | torch.device,
) -> _LoadedComparisonPolicy:
    if scratch_checkpoint is not None:
        return _load_comparison_policy_from_checkpoint(
            "scratch_trained",
            Path(scratch_checkpoint),
            source="scratch_checkpoint",
            device=device,
        )
    architecture_name, architecture_config = policy_architecture_from_checkpoint(
        reference_checkpoint
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(scratch_seed)
        model = make_full_smb_policy_model(
            architecture_name=architecture_name,
            architecture_config=architecture_config,
        ).to(device)
    model.eval()
    return _LoadedComparisonPolicy(
        name="scratch_initialized",
        model=model,
        checkpoint_path=None,
        source="scratch_initialization",
        architecture_name=architecture_name,
        architecture_config=architecture_config,
        model_name=None,
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
        model_state, _skipped_world_model_keys = action_level_world_model_state_dict(
            model,
            checkpoint["states"]["model"],
        )
        load_result = model.load_state_dict(model_state, strict=False)
        unsupported_missing = tuple(
            key
            for key in load_result.missing_keys
            if not key.startswith(ACTION_EVALUATION_ALLOWED_MISSING_PREFIXES)
        )
        if load_result.unexpected_keys or unsupported_missing:
            raise ValueError(
                "scratch checkpoint is incompatible with Full SMB policy model; "
                f"missing={unsupported_missing}, "
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
    logits = logits_a[:, -1, : len(SMB_ACTIONS)]
    logits = _apply_full_smb_motor_primitive_bias(
        logits,
        getattr(model, "last_motor_primitives", None),
    )
    return logits.detach().cpu()


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


def _policy_spec_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("policy specs must use NAME=CHECKPOINT")
    name, checkpoint = value.split("=", 1)
    name = name.strip()
    checkpoint = checkpoint.strip()
    if not name:
        raise argparse.ArgumentTypeError("policy spec name must be non-empty")
    if not checkpoint:
        raise argparse.ArgumentTypeError("policy spec checkpoint must be non-empty")
    reserved = {
        "transferred",
        "scratch_initialized",
        "scratch_trained",
        "fine_tuned",
        "known_good",
    }
    if name in reserved:
        raise argparse.ArgumentTypeError(f"policy spec name {name!r} is reserved")
    return name, Path(checkpoint)


def _make_full_smb_comparison_stage(
    vision: Any,
    task: Optional[FullSMBTaskSpec],
) -> FullSMBStage:
    return FullSMBStage(
        env_config=FullSMBEnvConfig(state=task.start.state if task is not None else None),
        vision=vision,
    )


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-checkpoint", type=Path, required=True)
    parser.add_argument("--scratch-checkpoint", "--scratch-trained-checkpoint", type=Path)
    parser.add_argument("--fine-tuned-checkpoint", "--finetuned-checkpoint", type=Path)
    parser.add_argument("--known-good-checkpoint", type=Path)
    parser.add_argument(
        "--policy",
        action="append",
        type=_policy_spec_arg,
        default=[],
        help="additional named policy checkpoint as NAME=PATH; may be repeated",
    )
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--seed", dest="seeds", type=int, action="append")
    parser.add_argument("--scratch-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--task-set", choices=FULL_SMB_TASK_SET_NAMES)
    parser.add_argument("--task", dest="task_names", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds or (0,))
    suite_requested = bool(
        args.fine_tuned_checkpoint is not None
        or args.known_good_checkpoint is not None
        or args.policy
        or args.task_names
        or args.task_set is not None
        or len(seeds) > 1
    )
    if suite_requested:
        result = compare_full_smb_policy_suite(
            args.transfer_checkpoint,
            make_stage=_make_full_smb_comparison_stage,
            scratch_checkpoint=args.scratch_checkpoint,
            fine_tuned_checkpoint=args.fine_tuned_checkpoint,
            known_good_checkpoint=args.known_good_checkpoint,
            extra_policy_checkpoints=dict(args.policy),
            full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
            config=FullSMBPolicySuiteComparisonConfig(
                steps=args.steps,
                seeds=seeds,
                scratch_seed=args.scratch_seed,
                device=args.device,
                render=args.render,
            ),
            task_names=tuple(args.task_names),
            task_set=args.task_set,
        )
    else:
        config = FullSMBPolicyComparisonConfig(
            steps=args.steps,
            seed=seeds[0],
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
