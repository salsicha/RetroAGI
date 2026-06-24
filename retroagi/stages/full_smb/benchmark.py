"""Throughput benchmarks for local Full SMB emulator runs."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import torch

from retroagi.core import SMB_ACTIONS, select_device, to_plain_data
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FullSMBContentSpec,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.vision import (
    DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    FullSMBSegmentationVision,
)


@dataclass(frozen=True)
class FullSMBThroughputBenchmarkConfig:
    """Configuration for measuring Full SMB emulator step throughput."""

    steps: int = 500
    warmup_steps: int = 25
    seed: int = 0
    frame_skip: int = 2
    render: bool = False
    encode_observations: bool = False
    device: str = "cpu"
    output: Optional[Path] = None
    env_config: FullSMBEnvConfig = field(default_factory=FullSMBEnvConfig)
    content_spec: FullSMBContentSpec = DEFAULT_FULL_SMB_CONTENT
    vision_checkpoint: Optional[Path] = DEFAULT_FULL_SMB_VIT_CHECKPOINT

    def __post_init__(self) -> None:
        for name in ("steps", "warmup_steps"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(getattr(self, name)))
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        object.__setattr__(self, "frame_skip", int(self.frame_skip))
        for path_name in ("output", "vision_checkpoint"):
            value = getattr(self, path_name)
            if value is not None and not isinstance(value, Path):
                object.__setattr__(self, path_name, Path(value))


@dataclass(frozen=True)
class FullSMBThroughputBenchmarkResult:
    """JSON-serializable Full SMB throughput report."""

    seed: int
    selected_device: str
    requested_steps: int
    warmup_steps: int
    measured_steps: int
    emulator_frames: int
    elapsed_seconds: float
    steps_per_second: float
    emulator_frames_per_second: float
    average_emulator_frames_per_step: float
    frame_skip: int
    render: bool
    encode_observations: bool
    resets: int
    terminated_count: int
    truncated_count: int
    total_reward: float
    final_signals: Mapping[str, Any]
    recommended_settings: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return to_plain_data(asdict(self))


StageFactory = Callable[[Any], FullSMBStage]


def benchmark_full_smb_throughput(
    stage: FullSMBStage,
    config: FullSMBThroughputBenchmarkConfig = FullSMBThroughputBenchmarkConfig(),
    *,
    selected_device: str | torch.device = "cpu",
) -> FullSMBThroughputBenchmarkResult:
    """Measure stage step throughput after a short warmup period."""

    rng = random.Random(config.seed)
    observation = stage.reset(seed=config.seed)
    resets = 1
    last_info: Mapping[str, Any] = stage.last_info

    for _ in range(config.warmup_steps):
        observation, _reward, terminated, truncated, last_info = _benchmark_step(
            stage,
            rng=rng,
            config=config,
        )
        if terminated or truncated:
            observation = stage.reset(seed=config.seed + resets)
            resets += 1
            last_info = stage.last_info

    measured_steps = 0
    emulator_frames = 0
    total_reward = 0.0
    terminated_count = 0
    truncated_count = 0
    started_at = time.perf_counter()
    for _ in range(config.steps):
        action = rng.choice(SMB_ACTIONS)
        observation, reward, terminated, truncated, info = stage.step(action)
        measured_steps += 1
        total_reward += float(reward)
        last_info = info
        emulator_frames += _executed_emulator_frames(info)
        if config.encode_observations:
            stage.encode_observation(observation, info)
        _maybe_render(stage, config.render)
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if terminated or truncated:
            if measured_steps < config.steps:
                observation = stage.reset(seed=config.seed + resets)
                resets += 1
                last_info = stage.last_info
    elapsed_seconds = max(time.perf_counter() - started_at, 1e-9)
    steps_per_second = measured_steps / elapsed_seconds
    frames_per_second = emulator_frames / elapsed_seconds
    return FullSMBThroughputBenchmarkResult(
        seed=config.seed,
        selected_device=str(selected_device),
        requested_steps=config.steps,
        warmup_steps=config.warmup_steps,
        measured_steps=measured_steps,
        emulator_frames=emulator_frames,
        elapsed_seconds=elapsed_seconds,
        steps_per_second=steps_per_second,
        emulator_frames_per_second=frames_per_second,
        average_emulator_frames_per_step=(
            float(emulator_frames / measured_steps) if measured_steps else 0.0
        ),
        frame_skip=config.frame_skip,
        render=config.render,
        encode_observations=config.encode_observations,
        resets=resets,
        terminated_count=terminated_count,
        truncated_count=truncated_count,
        total_reward=total_reward,
        final_signals=dict(last_info.get("full_smb_signals", {})),
        recommended_settings=recommended_full_smb_runtime_settings(),
    )


def run_full_smb_throughput_benchmark(
    config: FullSMBThroughputBenchmarkConfig = FullSMBThroughputBenchmarkConfig(),
    *,
    stage_factory: Optional[StageFactory] = None,
) -> FullSMBThroughputBenchmarkResult:
    """Create a Full SMB stage, run a throughput benchmark, and write output."""

    device = select_device(config.device)
    vision = _benchmark_vision(config, device)
    stage = (
        stage_factory(vision)
        if stage_factory is not None
        else FullSMBStage(
            env_config=config.env_config,
            content_spec=config.content_spec,
            observation_config=FullSMBObservationConfig(
                frame_skip=config.frame_skip,
                frame_stack=2,
                resize_shape=None,
            ),
            vision=vision,
        )
    )
    try:
        result = benchmark_full_smb_throughput(
            stage,
            config,
            selected_device=device,
        )
    finally:
        stage.close()
    if config.output is not None:
        save_full_smb_throughput_benchmark(config.output, result)
    return result


def save_full_smb_throughput_benchmark(
    path: Path,
    result: FullSMBThroughputBenchmarkResult,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def recommended_full_smb_runtime_settings() -> dict[str, Any]:
    """Return conservative device settings for Full SMB training and play."""

    return {
        "cpu": {
            "device_flag": "--device cpu",
            "training": (
                "Use for CI, content checks, short smoke training, and machines "
                "without an accelerator. Prefer headless runs, frozen perception, "
                "small rollout windows, and explicit benchmark reports before long jobs."
            ),
            "play": (
                "Use --render-mode human --fps 30 for manual play, or "
                "--render-mode none for deterministic evaluation and recording."
            ),
            "notes": (
                "stable-retro emulator stepping is CPU-bound, so CPU throughput "
                "sets the lower bound for every Full SMB run."
            ),
        },
        "cuda": {
            "device_flag": "--device cuda",
            "training": (
                "Use on Linux/NVIDIA systems for policy, world-model, critic, and "
                "Full SMB ViT inference or fine-tuning. Keep emulator workers "
                "headless and benchmark first because the emulator still runs on CPU."
            ),
            "play": (
                "Useful for heavy policy or ViT inference during rendered playback; "
                "disable recording/video export when measuring interaction latency."
            ),
            "notes": (
                "Expect gains only when model compute dominates emulator stepping. "
                "Small one-environment rollouts may remain emulator-limited."
            ),
        },
        "mps": {
            "device_flag": "--device mps",
            "training": (
                "Use on Apple Silicon for ViT and policy compute after comparing "
                "against CPU on the same benchmark. Keep perception frozen for "
                "short policy fine-tunes unless the asset-mock gate requires updates."
            ),
            "play": (
                "Good default for macOS rendered policy playback when the policy or "
                "ViT is larger than the baseline smoke configuration."
            ),
            "notes": (
                "MPS transfer overhead can outweigh acceleration for tiny batches, "
                "so benchmark CPU and MPS with the same frame_skip and rollout size."
            ),
        },
    }


def _benchmark_step(
    stage: FullSMBStage,
    *,
    rng: random.Random,
    config: FullSMBThroughputBenchmarkConfig,
) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
    action = rng.choice(SMB_ACTIONS)
    observation, reward, terminated, truncated, info = stage.step(action)
    if config.encode_observations:
        stage.encode_observation(observation, info)
    _maybe_render(stage, config.render)
    return observation, reward, terminated, truncated, info


def _executed_emulator_frames(info: Mapping[str, Any]) -> int:
    action = info.get("action", {})
    if isinstance(action, Mapping):
        try:
            frames = int(action.get("frames_executed", 1))
        except (TypeError, ValueError):
            frames = 1
        return max(frames, 0)
    return 1


def _maybe_render(stage: FullSMBStage, enabled: bool) -> None:
    if not enabled:
        return
    render = getattr(stage.env, "render", None)
    if render is not None:
        render()


def _benchmark_vision(
    config: FullSMBThroughputBenchmarkConfig,
    device: torch.device,
) -> FullSMBSegmentationVision | "_NoopVision":
    if not config.encode_observations:
        return _NoopVision()
    return FullSMBSegmentationVision(
        checkpoint=config.vision_checkpoint,
        device=device,
        freeze=True,
    )


class _NoopVision:
    def encode(self, _observation: Any) -> Any:
        raise RuntimeError("Full SMB throughput benchmark did not request observation encoding")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retroagi.stages.full_smb.benchmark",
        description="Benchmark local Full SMB stable-retro throughput.",
    )
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--state")
    parser.add_argument("--scenario")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--encode-observations", action="store_true")
    parser.add_argument(
        "--vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument(
        "--no-vision-checkpoint",
        action="store_true",
        help="use an untrained Full SMB ViT when --encode-observations is set",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = FullSMBThroughputBenchmarkConfig(
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
        frame_skip=args.frame_skip,
        render=args.render,
        encode_observations=args.encode_observations,
        device=args.device,
        output=args.output,
        env_config=FullSMBEnvConfig(state=args.state, scenario=args.scenario),
        vision_checkpoint=None if args.no_vision_checkpoint else args.vision_checkpoint,
    )
    result = run_full_smb_throughput_benchmark(config)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
