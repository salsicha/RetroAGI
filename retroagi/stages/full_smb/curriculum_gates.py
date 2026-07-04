"""Short Full SMB curriculum gates before fixed benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from retroagi.core import (
    SMB_ACTIONS,
    SMBAction,
    SMBJumpActionTerminator,
    select_device,
    to_plain_data,
)
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.vision import DEFAULT_FULL_SMB_VIT_CHECKPOINT

FULL_SMB_CURRICULUM_GATE_SCHEMA_VERSION = 1
DEFAULT_FULL_SMB_CURRICULUM_GATE_PASS_RATE = 1.0
DEFAULT_FULL_SMB_GATE_BENCHMARK_TASK = "benchmark_1_1_start"
DEFAULT_FULL_SMB_GATE_BENCHMARK_MAX_STEPS = 2400


@dataclass(frozen=True)
class FullSMBCurriculumGateThreshold:
    """Pass criteria for one short Level 1-1 curriculum gate."""

    name: str
    min_progress: float
    max_steps: int
    episodes: int = 1
    min_episode_pass_rate: float = 1.0
    min_survival_rate: float = 1.0
    min_mean_score: float = 0.0
    min_right_jump_fraction: float = 0.0
    rationale: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("gate name must be non-empty")
        if self.min_progress < 0.0:
            raise ValueError("min_progress must be non-negative")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.episodes <= 0:
            raise ValueError("episodes must be positive")
        for name in ("min_episode_pass_rate", "min_survival_rate", "min_right_jump_fraction"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_mean_score < 0.0:
            raise ValueError("min_mean_score must be non-negative")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError("gate tags must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_FULL_SMB_CURRICULUM_GATES: tuple[FullSMBCurriculumGateThreshold, ...] = (
    FullSMBCurriculumGateThreshold(
        name="opening_movement",
        min_progress=256.0,
        max_steps=260,
        rationale="Policy must leave the spawn area before full Level 1-1 evaluation.",
        tags=("level_1_1", "opening", "movement"),
    ),
    FullSMBCurriculumGateThreshold(
        name="first_pipe",
        min_progress=512.0,
        max_steps=500,
        min_right_jump_fraction=0.01,
        rationale="Policy must pass the first pipe region with at least some jump timing.",
        tags=("level_1_1", "pipe", "obstacle"),
    ),
    FullSMBCurriculumGateThreshold(
        name="first_enemy",
        min_progress=640.0,
        max_steps=650,
        min_right_jump_fraction=0.01,
        rationale="Policy must survive and progress beyond the first enemy encounter.",
        tags=("level_1_1", "enemy", "avoidance"),
    ),
    FullSMBCurriculumGateThreshold(
        name="first_gap_or_stair",
        min_progress=1024.0,
        max_steps=1000,
        min_right_jump_fraction=0.01,
        rationale="Policy must reach the first gap or stair transition before full benchmark.",
        tags=("level_1_1", "gap", "stairs"),
    ),
)


@torch.no_grad()
def run_full_smb_curriculum_gate_evaluation(
    model: torch.nn.Module,
    stage: FullSMBStage,
    *,
    device: torch.device,
    gates: Sequence[FullSMBCurriculumGateThreshold] = DEFAULT_FULL_SMB_CURRICULUM_GATES,
    seed: int = 0,
    min_gate_pass_rate: float = DEFAULT_FULL_SMB_CURRICULUM_GATE_PASS_RATE,
    episodes: Optional[int] = None,
    benchmark_task: str = DEFAULT_FULL_SMB_GATE_BENCHMARK_TASK,
    benchmark_max_steps: int = DEFAULT_FULL_SMB_GATE_BENCHMARK_MAX_STEPS,
) -> dict[str, Any]:
    """Run short deterministic Full SMB gates and return a benchmark gate report."""

    if not gates:
        raise ValueError("gates must be non-empty")
    if not 0.0 <= min_gate_pass_rate <= 1.0:
        raise ValueError("min_gate_pass_rate must be in [0, 1]")
    if episodes is not None and episodes <= 0:
        raise ValueError("episodes must be positive")
    if benchmark_max_steps <= 0:
        raise ValueError("benchmark_max_steps must be positive")

    model.eval()
    gate_results: dict[str, Any] = {}
    for gate_index, gate in enumerate(gates):
        gate_results[gate.name] = _run_full_smb_curriculum_gate(
            model,
            stage,
            gate,
            device=device,
            seed=seed + gate_index * 1_000,
            episodes=episodes if episodes is not None else gate.episodes,
        )
    gate_count = len(gate_results)
    gates_passed = sum(
        1 for result in gate_results.values() if bool(result["threshold_met"])
    )
    gate_pass_rate = gates_passed / gate_count if gate_count else 0.0
    blocking_gates = [
        name for name, result in gate_results.items() if not bool(result["threshold_met"])
    ]
    full_benchmark_allowed = gate_pass_rate >= min_gate_pass_rate
    return {
        "schema_version": FULL_SMB_CURRICULUM_GATE_SCHEMA_VERSION,
        "config": {
            "seed": int(seed),
            "episodes_override": int(episodes) if episodes is not None else None,
            "min_gate_pass_rate": float(min_gate_pass_rate),
            "benchmark_task": str(benchmark_task),
            "benchmark_max_steps": int(benchmark_max_steps),
        },
        "gates": gate_results,
        "summary": {
            "gate_count": int(gate_count),
            "gates_passed": int(gates_passed),
            "gate_pass_rate": float(gate_pass_rate),
            "blocking_gates": blocking_gates,
            "full_benchmark_allowed": bool(full_benchmark_allowed),
            "full_benchmark_blocked": bool(not full_benchmark_allowed),
            "full_benchmark_block_reason": (
                ""
                if full_benchmark_allowed
                else (
                    f"gate_pass_rate {gate_pass_rate:.3f} is below required "
                    f"{min_gate_pass_rate:.3f}"
                )
            ),
        },
    }


def default_full_smb_curriculum_gates() -> tuple[FullSMBCurriculumGateThreshold, ...]:
    """Return the ordered Level 1-1 short-gate thresholds."""

    return DEFAULT_FULL_SMB_CURRICULUM_GATES


def evaluate_full_smb_curriculum_gate_threshold(
    gate: FullSMBCurriculumGateThreshold,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Return threshold diagnostics for one curriculum-gate aggregate."""

    episodes = int(result.get("episodes", 0))
    episode_pass_rate = _metric(result, "episode_pass_rate")
    max_progress = _metric(result, "max_progress")
    mean_score = _metric(result, "mean_score")
    survival_rate = _metric(result, "survival_rate")
    right_jump_fraction = _metric(
        result.get("action_fractions", {}),
        SMBAction.RIGHT_JUMP.name,
    )
    enough_episodes = episodes >= gate.episodes
    within_step_budget = int(result.get("max_steps_per_episode", 0)) <= gate.max_steps
    meets_episode_pass_rate = episode_pass_rate >= gate.min_episode_pass_rate
    meets_progress = max_progress >= gate.min_progress
    meets_survival = survival_rate >= gate.min_survival_rate
    meets_score = mean_score >= gate.min_mean_score
    meets_right_jump_fraction = right_jump_fraction >= gate.min_right_jump_fraction
    threshold_met = (
        enough_episodes
        and within_step_budget
        and meets_episode_pass_rate
        and meets_progress
        and meets_survival
        and meets_score
        and meets_right_jump_fraction
    )
    return {
        "threshold": gate.to_dict(),
        "observed": {
            "episodes": episodes,
            "episode_pass_rate": float(episode_pass_rate),
            "max_progress": float(max_progress),
            "survival_rate": float(survival_rate),
            "mean_score": float(mean_score),
            "right_jump_fraction": float(right_jump_fraction),
        },
        "enough_episodes": bool(enough_episodes),
        "within_step_budget": bool(within_step_budget),
        "meets_episode_pass_rate": bool(meets_episode_pass_rate),
        "meets_progress": bool(meets_progress),
        "meets_survival": bool(meets_survival),
        "meets_score": bool(meets_score),
        "meets_right_jump_fraction": bool(meets_right_jump_fraction),
        "threshold_met": bool(threshold_met),
    }


def _run_full_smb_curriculum_gate(
    model: torch.nn.Module,
    stage: FullSMBStage,
    gate: FullSMBCurriculumGateThreshold,
    *,
    device: torch.device,
    seed: int,
    episodes: int,
) -> dict[str, Any]:
    from retroagi.stages.full_smb.train import (
        _full_smb_walk_action_limiter,
        _policy_action_logits_and_state,
    )

    episode_results = []
    all_actions: list[int] = []
    for episode_index in range(episodes):
        observation = stage.reset(seed=seed + episode_index)
        world_model_state = None
        jump_terminator = SMBJumpActionTerminator()
        walk_limiter = _full_smb_walk_action_limiter(stage)
        progress_values: list[float] = []
        score_values: list[float] = []
        coin_values: list[float] = []
        episode_actions: list[int] = []
        episode_return = 0.0
        death = False
        completion = False
        terminated = False
        truncated = False
        steps = 0
        for step_index in range(gate.max_steps):
            batch = stage.encode_observation(observation)
            forward = _policy_action_logits_and_state(
                model,
                batch,
                device=device,
                world_model_state=world_model_state,
            )
            action = int(forward.logits.argmax(dim=-1).item())
            action = jump_terminator.filter_action(action, batch=batch)
            action = walk_limiter.filter_action(action)
            observation, reward, terminated, truncated, info = stage.step(action)
            source = _signal_source(info)
            progress = _signal_progress(source)
            if progress is not None:
                progress_values.append(progress)
            score = _optional_float(source.get("score"))
            if score is not None:
                score_values.append(score)
            coins = _coins_from_source(source)
            if coins is not None:
                coin_values.append(coins)
            death = bool(death or source.get("death", False) or source.get("game_over", False))
            completion = bool(completion or source.get("completion", False))
            episode_return += float(reward)
            episode_actions.append(action)
            all_actions.append(action)
            steps = step_index + 1
            if terminated or truncated:
                break
            world_model_state = (
                forward.next_world_model_state.detach()
                if forward.next_world_model_state is not None
                else None
            )
        action_summary = _action_count_summary(episode_actions)
        episode_result = {
            "episode_index": int(episode_index),
            "seed": int(seed + episode_index),
            "steps": int(steps),
            "return": float(episode_return),
            "max_progress": float(max(progress_values, default=0.0)),
            "last_progress": float(progress_values[-1]) if progress_values else 0.0,
            "mean_progress": _mean(progress_values),
            "score": float(score_values[-1]) if score_values else 0.0,
            "coins": float(coin_values[-1]) if coin_values else 0.0,
            "survival": 0.0 if death else 1.0,
            "completion": 1.0 if completion else 0.0,
            "death": 1.0 if death else 0.0,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action_counts": action_summary["counts"],
            "action_fractions": action_summary["fractions"],
        }
        episode_result["threshold_met"] = bool(
            episode_result["max_progress"] >= gate.min_progress
            and episode_result["survival"] >= gate.min_survival_rate
            and episode_result["score"] >= gate.min_mean_score
            and episode_result["action_fractions"].get(SMBAction.RIGHT_JUMP.name, 0.0)
            >= gate.min_right_jump_fraction
        )
        episode_results.append(episode_result)

    aggregate = _aggregate_gate_episodes(episode_results, all_actions, gate=gate)
    diagnostics = evaluate_full_smb_curriculum_gate_threshold(gate, aggregate)
    return {
        **aggregate,
        "threshold": diagnostics["threshold"],
        "threshold_met": bool(diagnostics["threshold_met"]),
        "threshold_diagnostics": {
            key: value
            for key, value in diagnostics.items()
            if key not in {"threshold", "threshold_met"}
        },
        "episodes_detail": episode_results,
    }


def _aggregate_gate_episodes(
    episodes: Sequence[Mapping[str, Any]],
    actions: Sequence[int],
    *,
    gate: FullSMBCurriculumGateThreshold,
) -> dict[str, Any]:
    action_summary = _action_count_summary(actions)
    max_progress = [float(episode.get("max_progress", 0.0)) for episode in episodes]
    returns = [float(episode.get("return", 0.0)) for episode in episodes]
    scores = [float(episode.get("score", 0.0)) for episode in episodes]
    coins = [float(episode.get("coins", 0.0)) for episode in episodes]
    survival = [float(episode.get("survival", 0.0)) for episode in episodes]
    completion = [float(episode.get("completion", 0.0)) for episode in episodes]
    deaths = [float(episode.get("death", 0.0)) for episode in episodes]
    passed = [1.0 if bool(episode.get("threshold_met", False)) else 0.0 for episode in episodes]
    return {
        "gate": gate.name,
        "episodes": int(len(episodes)),
        "max_steps_per_episode": int(gate.max_steps),
        "steps": int(sum(int(episode.get("steps", 0)) for episode in episodes)),
        "max_progress": float(max(max_progress, default=0.0)),
        "mean_max_progress": _mean(max_progress),
        "mean_return": _mean(returns),
        "survival_rate": _mean(survival),
        "completion_rate": _mean(completion),
        "death_count": float(sum(deaths)),
        "mean_score": _mean(scores),
        "mean_coins": _mean(coins),
        "episode_pass_rate": _mean(passed),
        "action_counts": action_summary["counts"],
        "action_fractions": action_summary["fractions"],
    }


def _action_count_summary(actions: Sequence[int]) -> dict[str, Any]:
    counts = Counter(SMB_ACTIONS[int(action)].name for action in actions)
    total = max(sum(counts.values()), 1)
    return {
        "total": int(sum(counts.values())),
        "counts": {action.name: int(counts.get(action.name, 0)) for action in SMB_ACTIONS},
        "fractions": {
            action.name: float(counts.get(action.name, 0) / total) for action in SMB_ACTIONS
        },
    }


def _signal_source(info: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = info.get("full_smb_signals")
    return signals if isinstance(signals, Mapping) else info


def _signal_progress(source: Mapping[str, Any]) -> Optional[float]:
    progress = _optional_float(source.get("progress"))
    if progress is not None:
        return progress
    position = source.get("position")
    if position is None:
        return None
    try:
        values = tuple(position)
    except TypeError:
        return None
    return _optional_float(values[0]) if values else None


def _coins_from_source(source: Mapping[str, Any]) -> Optional[float]:
    coins = _optional_float(source.get("coins"))
    if coins is not None:
        return coins
    collectibles = source.get("collectibles")
    if isinstance(collectibles, Mapping):
        return _optional_float(collectibles.get("coins"))
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metric(result: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in result:
            value = _optional_float(result[name])
            if value is not None:
                return value
    return 0.0


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi gate --stage full")
    parser.add_argument("--policy-checkpoint", "--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        "--vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int)
    parser.add_argument(
        "--min-gate-pass-rate",
        type=float,
        default=DEFAULT_FULL_SMB_CURRICULUM_GATE_PASS_RATE,
    )
    parser.add_argument("--game-id", default=DEFAULT_FULL_SMB_CONTENT.game)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--benchmark-task", default=DEFAULT_FULL_SMB_GATE_BENCHMARK_TASK)
    parser.add_argument(
        "--benchmark-max-steps",
        type=int,
        default=DEFAULT_FULL_SMB_GATE_BENCHMARK_MAX_STEPS,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--fail-on-block",
        action="store_true",
        help="exit nonzero when gates block the configured full benchmark",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    device = select_device(args.device)
    from retroagi.stages.full_smb.train import (
        FullSMBTrainingConfig,
        _build_full_smb_perception,
        load_full_smb_policy_checkpoint,
    )

    model, _optimizer, _checkpoint = load_full_smb_policy_checkpoint(
        args.policy_checkpoint,
        device=device,
    )
    config = FullSMBTrainingConfig(
        device=str(device),
        full_smb_vision_checkpoint=args.full_smb_vision_checkpoint,
        game_id=args.game_id,
        emulator_state=args.state,
        frame_skip=args.frame_skip,
        evaluation_episodes=0,
        evaluation_max_steps=0,
    )
    vision = _build_full_smb_perception(config, device)
    stage = FullSMBStage(
        env_config=FullSMBEnvConfig(game=args.game_id, state=args.state),
        vision=vision,
        observation_config=FullSMBObservationConfig(frame_skip=args.frame_skip),
    )
    try:
        result = run_full_smb_curriculum_gate_evaluation(
            model,
            stage,
            device=device,
            seed=args.seed,
            min_gate_pass_rate=args.min_gate_pass_rate,
            episodes=args.episodes,
            benchmark_task=args.benchmark_task,
            benchmark_max_steps=args.benchmark_max_steps,
        )
    finally:
        stage.close()

    output = json.dumps(to_plain_data(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 1 if args.fail_on_block and result["summary"]["full_benchmark_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
