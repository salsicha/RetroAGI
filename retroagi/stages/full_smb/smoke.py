"""Headless smoke checks for the Full SMB stage adapter."""

from dataclasses import asdict, dataclass
import hashlib
import random
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from retroagi.core import SMB_ACTIONS, SMBAction, coerce_smb_action
from retroagi.stages.full_smb.adapter import FullSMBStage


@dataclass(frozen=True)
class FullSMBSmokeConfig:
    steps: int = 128
    seed: int = 0
    encode_observations: bool = False
    reset_on_done: bool = True
    render: bool = False


@dataclass(frozen=True)
class FullSMBRandomAgentSmokeResult:
    seed: int
    requested_steps: int
    executed_steps: int
    resets: int
    completed_episodes: int
    terminated_count: int
    truncated_count: int
    total_reward: float
    encoded_observations: int
    initial_observation_checksum: str
    final_observation_checksum: str
    final_signals: Mapping[str, Any]
    action_ids: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FullSMBRolloutTrace:
    seed: int
    observation_checksums: tuple[str, ...]
    rewards: tuple[float, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    signals: tuple[Mapping[str, Any], ...]
    encoded_observations: int
    action_ids: tuple[int, ...]


@dataclass(frozen=True)
class FullSMBDeterministicResetSmokeResult:
    seed: int
    deterministic: bool
    mismatch: Optional[str]
    first: FullSMBRolloutTrace
    second: FullSMBRolloutTrace

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_headless_random_agent_smoke(
    stage: FullSMBStage,
    config: FullSMBSmokeConfig = FullSMBSmokeConfig(),
) -> FullSMBRandomAgentSmokeResult:
    """Run a seeded random policy without rendering unless explicitly requested."""

    if config.steps < 0:
        raise ValueError("steps must be non-negative")

    rng = random.Random(config.seed)
    observation = stage.reset(seed=config.seed)
    initial_checksum = _observation_checksum(observation)
    final_checksum = initial_checksum
    encoded_observations = _maybe_encode(stage, observation, config.encode_observations)
    action_ids: list[int] = []
    total_reward = 0.0
    resets = 1
    completed_episodes = 0
    terminated_count = 0
    truncated_count = 0
    last_info: Mapping[str, Any] = stage.last_info

    for step_index in range(config.steps):
        action = rng.choice(SMB_ACTIONS)
        observation, reward, terminated, truncated, info = stage.step(action)
        total_reward += float(reward)
        action_ids.append(int(action))
        final_checksum = _observation_checksum(observation)
        last_info = info
        encoded_observations += _maybe_encode(
            stage, observation, config.encode_observations, info
        )
        _maybe_render(stage, config.render)

        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if terminated or truncated:
            completed_episodes += 1
            if config.reset_on_done and step_index + 1 < config.steps:
                observation = stage.reset(seed=config.seed + resets)
                resets += 1
                final_checksum = _observation_checksum(observation)
                last_info = stage.last_info
                encoded_observations += _maybe_encode(
                    stage, observation, config.encode_observations
                )

    return FullSMBRandomAgentSmokeResult(
        seed=config.seed,
        requested_steps=config.steps,
        executed_steps=len(action_ids),
        resets=resets,
        completed_episodes=completed_episodes,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
        total_reward=total_reward,
        encoded_observations=encoded_observations,
        initial_observation_checksum=initial_checksum,
        final_observation_checksum=final_checksum,
        final_signals=dict(last_info.get("full_smb_signals", {})),
        action_ids=tuple(action_ids),
    )


def run_deterministic_reset_smoke(
    make_stage: Callable[[], FullSMBStage],
    *,
    seed: int = 0,
    steps: int = 16,
    actions: Optional[Sequence[SMBAction | int]] = None,
    encode_observations: bool = False,
) -> FullSMBDeterministicResetSmokeResult:
    """Compare two fresh stages reset with the same seed and action sequence."""

    if steps < 0:
        raise ValueError("steps must be non-negative")
    action_sequence = (
        tuple(actions) if actions is not None else _seeded_actions(seed, steps)
    )

    first_stage = make_stage()
    try:
        first = _rollout_trace(
            first_stage,
            seed=seed,
            actions=action_sequence,
            encode_observations=encode_observations,
        )
    finally:
        first_stage.close()

    second_stage = make_stage()
    try:
        second = _rollout_trace(
            second_stage,
            seed=seed,
            actions=action_sequence,
            encode_observations=encode_observations,
        )
    finally:
        second_stage.close()

    mismatch = _trace_mismatch(first, second)
    return FullSMBDeterministicResetSmokeResult(
        seed=seed,
        deterministic=mismatch is None,
        mismatch=mismatch,
        first=first,
        second=second,
    )


def _rollout_trace(
    stage: FullSMBStage,
    *,
    seed: int,
    actions: Sequence[SMBAction | int],
    encode_observations: bool,
) -> FullSMBRolloutTrace:
    observation = stage.reset(seed=seed)
    observation_checksums = [_observation_checksum(observation)]
    encoded_observations = _maybe_encode(stage, observation, encode_observations)
    rewards: list[float] = []
    terminated_values: list[bool] = []
    truncated_values: list[bool] = []
    signals: list[Mapping[str, Any]] = []
    action_ids: list[int] = []

    for action in actions:
        shared_action = coerce_smb_action(action)
        observation, reward, terminated, truncated, info = stage.step(shared_action)
        observation_checksums.append(_observation_checksum(observation))
        rewards.append(float(reward))
        terminated_values.append(bool(terminated))
        truncated_values.append(bool(truncated))
        signals.append(dict(info.get("full_smb_signals", {})))
        action_ids.append(int(shared_action))
        encoded_observations += _maybe_encode(
            stage, observation, encode_observations, info
        )
        if terminated or truncated:
            break

    return FullSMBRolloutTrace(
        seed=seed,
        observation_checksums=tuple(observation_checksums),
        rewards=tuple(rewards),
        terminated=tuple(terminated_values),
        truncated=tuple(truncated_values),
        signals=tuple(signals),
        encoded_observations=encoded_observations,
        action_ids=tuple(action_ids),
    )


def _seeded_actions(seed: int, steps: int) -> tuple[SMBAction, ...]:
    rng = random.Random(seed)
    return tuple(rng.choice(SMB_ACTIONS) for _ in range(steps))


def _trace_mismatch(
    first: FullSMBRolloutTrace, second: FullSMBRolloutTrace
) -> Optional[str]:
    for field in (
        "observation_checksums",
        "rewards",
        "terminated",
        "truncated",
        "signals",
        "encoded_observations",
        "action_ids",
    ):
        if getattr(first, field) != getattr(second, field):
            return field
    return None


def _maybe_encode(
    stage: FullSMBStage,
    observation: np.ndarray,
    enabled: bool,
    info: Optional[Mapping[str, Any]] = None,
) -> int:
    if not enabled:
        return 0
    stage.encode_observation(observation, info)
    return 1


def _maybe_render(stage: FullSMBStage, enabled: bool) -> None:
    if not enabled:
        return
    render = getattr(stage.env, "render", None)
    if render is not None:
        render()


def _observation_checksum(observation: Any) -> str:
    array = np.ascontiguousarray(np.asarray(observation))
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()
