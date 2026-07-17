"""Action-logit probes for canonical Block SMB hazard states."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from retroagi.core import (
    SMB_ACTIONS,
    SMBAction,
    StageBatch,
    VisionEncoder,
    WorldModelState,
)

from .adapter import BLOCK_SMB_SPEC, BlockSMBStage
from .env import MarioScenarioEnv
from .scripted_policy import fixed_scenario_action_scripts
from .train import (
    BlockSMBAblationConfig,
    apply_block_smb_ablations,
    load_fixed_scenarios,
)
from .vision import BlockVisionTransformer

BLOCK_SMB_ACTION_PROBE_SCHEMA_VERSION = 1
DEFAULT_BLOCK_SMB_ACTION_PROBE_SCENARIOS = (
    "level_2_gap.json",
    "level_3_stairs.json",
    "level_8_enemy_gap.json",
)
DEFAULT_BLOCK_SMB_ACTION_PROBE_MAX_STEPS = 80
DEFAULT_BLOCK_SMB_ACTION_PROBE_POINTS_PER_SCENARIO = 2


@torch.no_grad()
def run_block_smb_action_probe(
    model: torch.nn.Module,
    *,
    device: torch.device,
    vision_factory: Callable[[], VisionEncoder] = BlockVisionTransformer,
    scenarios: Sequence[str] = DEFAULT_BLOCK_SMB_ACTION_PROBE_SCENARIOS,
    seed: int = 0,
    max_steps: int = DEFAULT_BLOCK_SMB_ACTION_PROBE_MAX_STEPS,
    points_per_scenario: int = DEFAULT_BLOCK_SMB_ACTION_PROBE_POINTS_PER_SCENARIO,
    ablation: BlockSMBAblationConfig | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Probe policy logits at scripted pre-jump hazard states.

    The selected states are the scripted policy's jump-transition points in
    canonical gap/stair scenarios. Each sample logs actor logits, an explicit
    motor-primitive combo-biased view, RIGHT-vs-RIGHT_JUMP margins, predicted
    C-stream motion, and critic-vector magnitude.
    """

    if not scenarios:
        raise ValueError("scenarios must not be empty")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if points_per_scenario <= 0:
        raise ValueError("points_per_scenario must be positive")

    ablation_config = _ablation_config(ablation)
    scripts = fixed_scenario_action_scripts(max_steps=max_steps)
    scenario_names = tuple(str(name) for name in scenarios)
    fixed = dict(load_fixed_scenarios(scenario_names))

    model.eval()
    samples: list[dict[str, Any]] = []
    missing_probe_points: list[dict[str, Any]] = []

    for scenario_index, scenario_name in enumerate(scenario_names):
        if scenario_name not in scripts:
            raise ValueError(f"no scripted action sequence for {scenario_name!r}")
        candidate_steps = _canonical_jump_transition_steps(
            scripts[scenario_name],
            limit=points_per_scenario,
        )
        if not candidate_steps:
            missing_probe_points.append(
                {
                    "scenario_name": scenario_name,
                    "reason": "no_jump_transition_in_script",
                }
            )
            continue

        stage = BlockSMBStage(
            env=MarioScenarioEnv(),
            scenario=fixed[scenario_name],
            vision=vision_factory(),
        )
        world_model_state: WorldModelState | None = None
        try:
            observation = stage.reset(seed=seed + scenario_index)
            for step_index in range(max_steps):
                batch = _batch_to_device(
                    apply_block_smb_ablations(
                        stage.encode_observation(observation),
                        ablation_config,
                    ),
                    device,
                )
                forward = _policy_probe_forward(
                    model,
                    batch,
                    world_model_state=(
                        world_model_state if ablation_config.recurrent_state_enabled else None
                    ),
                    ablation=ablation_config,
                )
                if step_index in candidate_steps:
                    samples.append(
                        _probe_sample(
                            scenario_name=scenario_name,
                            step_index=step_index,
                            expected_action=int(scripts[scenario_name][step_index]),
                            batch=batch,
                            info=stage.last_info,
                            forward=forward,
                            label=_probe_label(scenario_name, step_index, candidate_steps),
                        )
                    )

                action = int(scripts[scenario_name][step_index])
                observation, _reward, terminated, truncated, _info = stage.step(action)
                if terminated or truncated:
                    break
                world_model_state = (
                    forward["next_world_model_state"].detach()
                    if ablation_config.recurrent_state_enabled
                    and forward["next_world_model_state"] is not None
                    else None
                )
        finally:
            stage.env.close()

        collected_steps = {
            sample["step_index"] for sample in samples if sample["scenario_name"] == scenario_name
        }
        for step_index in candidate_steps:
            if step_index not in collected_steps:
                missing_probe_points.append(
                    {
                        "scenario_name": scenario_name,
                        "step_index": int(step_index),
                        "reason": "trajectory_ended_before_probe",
                    }
                )

    return {
        "schema_version": BLOCK_SMB_ACTION_PROBE_SCHEMA_VERSION,
        "config": {
            "scenarios": list(scenario_names),
            "seed": int(seed),
            "max_steps": int(max_steps),
            "points_per_scenario": int(points_per_scenario),
            "ablation": {
                "vision_enabled": bool(ablation_config.vision_enabled),
                "world_model_enabled": bool(ablation_config.world_model_enabled),
                "critic_feedback_enabled": bool(ablation_config.critic_feedback_enabled),
                "hierarchy_enabled": bool(ablation_config.hierarchy_enabled),
                "recurrent_state_enabled": bool(ablation_config.recurrent_state_enabled),
                "checkpoint_transfer_enabled": bool(ablation_config.checkpoint_transfer_enabled),
            },
        },
        "samples": samples,
        "summary": _probe_summary(samples),
        "missing_probe_points": missing_probe_points,
    }


def _ablation_config(
    ablation: BlockSMBAblationConfig | Mapping[str, Any] | None,
) -> BlockSMBAblationConfig:
    if ablation is None:
        return BlockSMBAblationConfig()
    if isinstance(ablation, BlockSMBAblationConfig):
        return ablation
    return BlockSMBAblationConfig(**dict(ablation))


def _canonical_jump_transition_steps(
    actions: Sequence[int],
    *,
    limit: int,
) -> tuple[int, ...]:
    jump_actions = {
        int(SMBAction.RIGHT_JUMP),
        int(SMBAction.LEFT_JUMP),
        int(SMBAction.JUMP),
    }
    steps: list[int] = []
    previous: int | None = None
    for step_index, action in enumerate(actions):
        action = int(action)
        if action in jump_actions and action != previous:
            steps.append(step_index)
            if len(steps) >= limit:
                break
        previous = action
    return tuple(steps)


def _batch_to_device(batch: StageBatch, device: torch.device) -> StageBatch:
    batch.src_a = batch.src_a.to(device)
    batch.src_b = batch.src_b.to(device)
    batch.src_c = batch.src_c.to(device)
    return batch


def _episode_mask(batch: StageBatch) -> torch.Tensor | None:
    episode = (batch.metadata or {}).get("episode", {})
    if not isinstance(episode, Mapping):
        return None
    mask = episode.get("mask")
    if mask is None:
        return None
    return torch.as_tensor(mask, dtype=batch.src_c.dtype, device=batch.src_c.device)


def _policy_probe_forward(
    model: torch.nn.Module,
    batch: StageBatch,
    *,
    world_model_state: WorldModelState | None,
    ablation: BlockSMBAblationConfig,
) -> dict[str, Any]:
    outputs = model(
        batch.src_a,
        batch.src_b,
        batch.src_c,
        tau=1.0,
        world_model_state=world_model_state,
        episode_mask=_episode_mask(batch),
        return_world_model_state=True,
        critic_feedback_enabled=ablation.critic_feedback_enabled,
        world_model_enabled=ablation.world_model_enabled,
    )
    if len(outputs) != 8:
        raise ValueError(
            "Block SMB action probe requires trainer-compatible policy tuple "
            f"with 8 outputs, got {len(outputs)}"
        )
    (
        _actions1,
        next_state_pred,
        criticism,
        _actions2,
        logits_a,
        _w,
        _b,
        next_world_model_state,
    ) = outputs
    raw_logits = logits_a[:, -1, : len(SMB_ACTIONS)].detach().float()
    motor_primitives = getattr(model, "last_motor_primitives", None)
    motor_biased_logits = _apply_smb_motor_primitive_bias(
        raw_logits,
        motor_primitives,
    ).detach()
    return {
        "raw_logits": raw_logits,
        "motor_biased_logits": motor_biased_logits,
        "next_state_pred": next_state_pred.detach().float(),
        "criticism": criticism.detach().float(),
        "motor_primitives": (
            motor_primitives.detach() if hasattr(motor_primitives, "detach") else motor_primitives
        ),
        "next_world_model_state": next_world_model_state,
    }


def _apply_smb_motor_primitive_bias(
    logits: torch.Tensor,
    motor_primitives: Any,
) -> torch.Tensor:
    if motor_primitives is None or logits.size(-1) < len(SMB_ACTIONS):
        return logits
    try:
        confidence = motor_primitives.confidence[:, -1]
        replan_probability = motor_primitives.replan_probability[:, -1]
    except (AttributeError, IndexError, TypeError):
        return logits

    combo_strength = (confidence * replan_probability).to(
        device=logits.device,
        dtype=logits.dtype,
    )
    if combo_strength.ndim != 1 or combo_strength.size(0) != logits.size(0):
        return logits

    max_boost = 5.0
    base_boost = (0.5 * combo_strength).clamp(min=0.0, max=max_boost)
    bias = torch.zeros_like(logits)
    bias = bias + _combined_action_bias(
        logits,
        primary=int(SMBAction.RIGHT),
        combo=int(SMBAction.RIGHT_JUMP),
        base_boost=base_boost,
        max_boost=max_boost,
    )
    bias = bias + _combined_action_bias(
        logits,
        primary=int(SMBAction.LEFT),
        combo=int(SMBAction.LEFT_JUMP),
        base_boost=base_boost,
        max_boost=max_boost,
    )
    return logits + bias


def _combined_action_bias(
    logits: torch.Tensor,
    *,
    primary: int,
    combo: int,
    base_boost: torch.Tensor,
    max_boost: float,
) -> torch.Tensor:
    combo_gap = (logits[:, primary] - logits[:, combo]).clamp(min=0.0, max=max_boost)
    active = (base_boost > 0.0).to(dtype=logits.dtype, device=logits.device)
    boost = (base_boost + combo_gap) * active
    bias = torch.zeros_like(logits)
    bias[:, combo] = boost
    return bias


def _probe_sample(
    *,
    scenario_name: str,
    step_index: int,
    expected_action: int,
    batch: StageBatch,
    info: Mapping[str, Any],
    forward: Mapping[str, Any],
    label: str,
) -> dict[str, Any]:
    raw_logits = forward["raw_logits"][0]
    motor_biased_logits = forward["motor_biased_logits"][0]
    raw_action = int(raw_logits.argmax(dim=-1).item())
    motor_biased_action = int(motor_biased_logits.argmax(dim=-1).item())
    return {
        "label": label,
        "scenario_name": scenario_name,
        "step_index": int(step_index),
        "expected_action": int(expected_action),
        "expected_action_name": SMB_ACTIONS[int(expected_action)].name,
        "raw_action": raw_action,
        "raw_action_name": SMB_ACTIONS[raw_action].name,
        "motor_biased_action": motor_biased_action,
        "motor_biased_action_name": SMB_ACTIONS[motor_biased_action].name,
        "raw_logits": _logits_by_action(raw_logits),
        "motor_biased_logits": _logits_by_action(motor_biased_logits),
        "motor_bias": _logits_by_action(motor_biased_logits - raw_logits),
        "margins": _right_jump_margins(raw_logits, motor_biased_logits),
        "predicted_motion": _predicted_motion_summary(
            forward["next_state_pred"],
            batch.src_c,
        ),
        "critic": _critic_summary(forward["criticism"]),
        "motor_primitives": _motor_primitive_summary(forward["motor_primitives"]),
        "state": _state_summary(info),
    }


def _probe_label(
    scenario_name: str,
    step_index: int,
    candidate_steps: Sequence[int],
) -> str:
    prefix = "pre_stair" if "stair" in scenario_name else "pre_gap"
    ordinal = tuple(candidate_steps).index(step_index) + 1
    return f"{prefix}_jump_{ordinal}"


def _logits_by_action(logits: torch.Tensor) -> dict[str, float]:
    values = logits.detach().float().cpu()
    return {action.name: float(values[int(action)].item()) for action in SMB_ACTIONS}


def _right_jump_margins(
    raw_logits: torch.Tensor,
    motor_biased_logits: torch.Tensor,
) -> dict[str, float]:
    right = int(SMBAction.RIGHT)
    right_jump = int(SMBAction.RIGHT_JUMP)

    def margin(values: torch.Tensor) -> float:
        return float((values[right] - values[right_jump]).detach().cpu().item())

    raw = margin(raw_logits)
    motor = margin(motor_biased_logits)
    return {
        "raw_right_minus_right_jump": raw,
        "raw_right_jump_minus_right": -raw,
        "motor_biased_right_minus_right_jump": motor,
        "motor_biased_right_jump_minus_right": -motor,
        "motor_bias_delta_right_minus_right_jump": motor - raw,
    }


def _predicted_motion_summary(
    next_state_pred: torch.Tensor,
    current_state: torch.Tensor,
) -> dict[str, float]:
    if tuple(next_state_pred.shape) != tuple(current_state.shape):
        return {
            "available": 0.0,
            "shape_mismatch": 1.0,
        }
    if next_state_pred.ndim != 2:
        return {
            "available": 0.0,
            "shape_mismatch": 0.0,
        }
    seq_len_b = current_state.size(1) // BLOCK_SMB_SPEC.ratio_bc
    usable = seq_len_b * BLOCK_SMB_SPEC.ratio_bc
    if usable <= 0:
        return {
            "available": 0.0,
            "shape_mismatch": 0.0,
        }
    delta = next_state_pred[:, :usable] - current_state[:, :usable]
    signed = delta.reshape(delta.size(0), seq_len_b, BLOCK_SMB_SPEC.ratio_bc).mean(dim=-1)
    absolute = (
        delta.abs()
        .reshape(
            delta.size(0),
            seq_len_b,
            BLOCK_SMB_SPEC.ratio_bc,
        )
        .mean(dim=-1)
    )
    return {
        "available": 1.0,
        "absolute_last": float(absolute[:, -1].mean().cpu().item()),
        "absolute_mean": float(absolute.mean().cpu().item()),
        "absolute_max": float(absolute.max().cpu().item()),
        "signed_last": float(signed[:, -1].mean().cpu().item()),
        "signed_mean": float(signed.mean().cpu().item()),
    }


def _critic_summary(criticism: torch.Tensor) -> dict[str, float]:
    values = criticism.detach().float()
    return {
        "norm": float(values.norm().cpu().item()),
        "mean_abs": float(values.abs().mean().cpu().item()),
        "max_abs": float(values.abs().max().cpu().item()),
    }


def _motor_primitive_summary(motor_primitives: Any) -> dict[str, float]:
    if motor_primitives is None:
        return {"available": 0.0}
    summary = {"available": 1.0}
    for field_name in (
        "confidence",
        "replan_probability",
        "hold_duration",
        "interrupt_logit",
    ):
        try:
            value = getattr(motor_primitives, field_name)[:, -1][0]
        except (AttributeError, IndexError, TypeError):
            continue
        summary[field_name] = float(value.detach().float().cpu().item())
    return summary


def _state_summary(info: Mapping[str, Any]) -> dict[str, Any]:
    mario = info.get("mario", {})
    if not isinstance(mario, Mapping):
        mario = {}
    return {
        "mario": {
            name: _finite_float(mario.get(name)) for name in ("x", "y", "vx", "vy", "on_ground")
        },
        "support_right_dx": _finite_float(info.get("support_right_dx")),
        "next_platform_delta": _finite_mapping(info.get("next_platform_delta")),
        "ground_ahead": _finite_mapping(info.get("ground_ahead")),
        "nearest_enemy": _finite_mapping(info.get("nearest_enemy")),
        "goal_delta": _finite_mapping(info.get("goal_delta")),
    }


def _finite_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): number for key, raw in value.items() if (number := _finite_float(raw)) is not None
    }


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _probe_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    raw_actions = [str(sample["raw_action_name"]) for sample in samples]
    motor_actions = [str(sample["motor_biased_action_name"]) for sample in samples]
    margins = [float(sample["margins"]["raw_right_minus_right_jump"]) for sample in samples]
    motor_margins = [
        float(sample["margins"]["motor_biased_right_minus_right_jump"]) for sample in samples
    ]
    critic_norms = [float(sample["critic"]["norm"]) for sample in samples]
    return {
        "sample_count": len(samples),
        "raw_action_counts": _count_names(raw_actions),
        "motor_biased_action_counts": _count_names(motor_actions),
        "mean_raw_right_minus_right_jump": _mean(margins),
        "mean_motor_biased_right_minus_right_jump": _mean(motor_margins),
        "mean_critic_norm": _mean(critic_norms),
    }


def _count_names(names: Sequence[str]) -> dict[str, int]:
    return {
        action.name: int(sum(1 for name in names if name == action.name)) for action in SMB_ACTIONS
    }


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def save_block_smb_action_probe(path: Path, result: Mapping[str, Any]) -> None:
    """Write a Block SMB action probe JSON artifact."""

    import json

    from retroagi.core import to_plain_data

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_plain_data(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
