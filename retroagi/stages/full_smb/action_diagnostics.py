"""Action-contract diagnostics for transferred Full SMB policies."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch

from retroagi.core import SMB_ACTIONS, SMBAction, select_device, to_plain_data
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.vision import DEFAULT_FULL_SMB_VIT_CHECKPOINT

FULL_SMB_ACTION_DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_FULL_SMB_ACTION_PROGRESS_GATE = 512.0


@torch.no_grad()
def run_full_smb_action_contract_diagnostic(
    model: torch.nn.Module,
    stage: FullSMBStage,
    *,
    device: torch.device,
    samples: int = 16,
    seed: int = 0,
    sample_repeats: int = 8,
    temperature: float = 1.0,
    progress_gate: float = DEFAULT_FULL_SMB_ACTION_PROGRESS_GATE,
    recording_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    """Inspect policy action selection and saved rollout action contracts."""

    if samples <= 0:
        raise ValueError("samples must be positive")
    if sample_repeats <= 0:
        raise ValueError("sample_repeats must be positive")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if progress_gate < 0.0:
        raise ValueError("progress_gate must be non-negative")

    from retroagi.stages.full_smb.train import _policy_action_logits_and_state

    model.eval()
    observation = stage.reset(seed=seed)
    world_model_state = None
    deterministic_actions: list[int] = []
    sampled_actions: list[int] = []
    progress_values: list[float] = []
    motor_snapshots: list[dict[str, Any]] = []

    for index in range(samples):
        batch = stage.encode_observation(observation)
        forward = _policy_action_logits_and_state(
            model,
            batch,
            device=device,
            world_model_state=world_model_state,
        )
        logits = forward.logits.detach().float()
        action = int(logits.argmax(dim=-1).item())
        deterministic_actions.append(action)
        distribution = torch.distributions.Categorical(logits=logits / float(temperature))
        sampled = distribution.sample((sample_repeats,)).reshape(-1).detach().cpu().tolist()
        sampled_actions.extend(int(item) for item in sampled)
        motor_snapshots.append(
            _motor_snapshot(
                index=index,
                action=action,
                logits=logits,
                motor_primitives=forward.motor_primitives,
            )
        )
        observation, _reward, terminated, truncated, info = stage.step(action)
        progress = _progress_from_info(info)
        if progress is not None:
            progress_values.append(progress)
        if terminated or truncated:
            observation = stage.reset(seed=seed + index + 1)
            world_model_state = None
        else:
            world_model_state = (
                forward.next_world_model_state.detach()
                if forward.next_world_model_state is not None
                else None
            )

    deterministic_summary = _action_count_summary(deterministic_actions)
    sampled_summary = _action_count_summary(sampled_actions)
    recording_summary = summarize_full_smb_action_recordings(
        recording_paths,
        progress_gate=progress_gate,
    )
    block_reference = scripted_block_smb_action_reference(max_steps=max(samples, 1))
    comparison = compare_full_smb_action_distribution(
        block_reference["action_counts"],
        recording_summary["action_counts"] or deterministic_summary["counts"],
    )
    canonical_max_progress = max(progress_values, default=0.0)
    flags = _action_contract_flags(
        deterministic_summary=deterministic_summary,
        recording_summary=recording_summary,
        canonical_max_progress=canonical_max_progress,
        progress_gate=progress_gate,
    )
    return {
        "schema_version": FULL_SMB_ACTION_DIAGNOSTIC_SCHEMA_VERSION,
        "config": {
            "samples": int(samples),
            "seed": int(seed),
            "sample_repeats": int(sample_repeats),
            "temperature": float(temperature),
            "progress_gate": float(progress_gate),
            "recording_paths": [str(path) for path in recording_paths],
        },
        "canonical_policy": {
            "deterministic": deterministic_summary,
            "sampled": sampled_summary,
            "progress": {
                "samples": len(progress_values),
                "max": float(canonical_max_progress),
                "last": float(progress_values[-1]) if progress_values else 0.0,
            },
        },
        "motor_primitives": _motor_summary(motor_snapshots),
        "recordings": recording_summary,
        "block_smb_scripted_reference": block_reference,
        "transfer_action_comparison": comparison,
        "flags": flags,
        "bottleneck_reasons": [
            name for name, value in flags.items() if bool(value)
        ],
    }


def summarize_full_smb_action_recordings(
    recording_paths: Sequence[Path],
    *,
    progress_gate: float = DEFAULT_FULL_SMB_ACTION_PROGRESS_GATE,
) -> dict[str, Any]:
    """Summarize Full SMB recording npz action names and progress traces."""

    artifacts = []
    all_actions: list[int] = []
    for path in _iter_recording_npz_paths(recording_paths):
        try:
            with np.load(path, allow_pickle=True) as data:
                actions = _recording_actions(data)
                if not actions:
                    continue
                progress_values = _recording_progress(data)
        except (OSError, ValueError, KeyError):
            continue
        all_actions.extend(actions)
        action_summary = _action_count_summary(actions)
        max_progress = max(progress_values, default=0.0)
        artifacts.append(
            {
                "path": str(path),
                "steps": len(actions),
                "action_counts": action_summary["counts"],
                "action_fractions": action_summary["fractions"],
                "max_progress": float(max_progress),
                "last_progress": float(progress_values[-1]) if progress_values else 0.0,
                "missing_right_jump_when_stalled": bool(
                    max_progress < progress_gate
                    and action_summary["counts"].get(SMBAction.RIGHT_JUMP.name, 0) == 0
                ),
            }
        )
    summary = _action_count_summary(all_actions)
    max_progress = max((item["max_progress"] for item in artifacts), default=0.0)
    return {
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "steps": len(all_actions),
        "action_counts": summary["counts"],
        "action_fractions": summary["fractions"],
        "max_progress": float(max_progress),
        "missing_right_jump_when_stalled": any(
            item["missing_right_jump_when_stalled"] for item in artifacts
        ),
    }


def scripted_block_smb_action_reference(*, max_steps: int = 200) -> dict[str, Any]:
    """Return action frequencies from the scripted Block SMB fixed scenarios."""

    from retroagi.stages.block_smb.scripted_policy import fixed_scenario_action_scripts

    actions: list[int] = []
    scripts = fixed_scenario_action_scripts(max_steps=max_steps)
    for script in scripts.values():
        actions.extend(int(action) for action in script)
    summary = _action_count_summary(actions)
    return {
        "source": "fixed_scenario_action_scripts",
        "scenario_count": len(scripts),
        "max_steps": int(max_steps),
        "action_counts": summary["counts"],
        "action_fractions": summary["fractions"],
    }


def compare_full_smb_action_distribution(
    reference_counts: Mapping[str, int],
    observed_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Compare a reference action distribution against a Full SMB rollout."""

    reference_total = max(sum(int(value) for value in reference_counts.values()), 1)
    observed_total = max(sum(int(value) for value in observed_counts.values()), 1)
    actions = [action.name for action in SMB_ACTIONS]
    deltas = {}
    for name in actions:
        reference_fraction = int(reference_counts.get(name, 0)) / reference_total
        observed_fraction = int(observed_counts.get(name, 0)) / observed_total
        deltas[name] = float(observed_fraction - reference_fraction)
    return {
        "reference_total": int(reference_total),
        "observed_total": int(observed_total),
        "fraction_delta": deltas,
        "right_jump_delta": deltas[SMBAction.RIGHT_JUMP.name],
    }


def _action_contract_flags(
    *,
    deterministic_summary: Mapping[str, Any],
    recording_summary: Mapping[str, Any],
    canonical_max_progress: float,
    progress_gate: float,
) -> dict[str, bool]:
    deterministic_counts = deterministic_summary["counts"]
    deterministic_fractions = deterministic_summary["fractions"]
    canonical_stalled = canonical_max_progress < progress_gate
    missing_canonical = (
        canonical_stalled
        and deterministic_counts.get(SMBAction.RIGHT_JUMP.name, 0) == 0
    )
    overactive_canonical = (
        canonical_stalled
        and deterministic_fractions.get(SMBAction.RIGHT_JUMP.name, 0.0) >= 0.95
    )
    return {
        "missing_right_jump_when_stalled": bool(
            missing_canonical or recording_summary["missing_right_jump_when_stalled"]
        ),
        "overactive_right_jump_when_stalled": bool(overactive_canonical),
    }


def _motor_snapshot(
    *,
    index: int,
    action: int,
    logits: torch.Tensor,
    motor_primitives: Any,
) -> dict[str, Any]:
    logits_1d = logits[0] if logits.ndim > 1 else logits
    raw_logits = logits_1d
    if motor_primitives is not None:
        try:
            raw_logits = motor_primitives.button_combo_logits[:, -1, : len(SMB_ACTIONS)][0]
        except (AttributeError, IndexError, TypeError):
            raw_logits = logits_1d
    bias = (logits_1d - raw_logits.to(device=logits_1d.device, dtype=logits_1d.dtype)).detach()
    snapshot = {
        "index": int(index),
        "action": int(action),
        "action_name": SMB_ACTIONS[int(action)].name,
        "right_jump_bias": float(bias[int(SMBAction.RIGHT_JUMP)].cpu().item()),
        "left_jump_bias": float(bias[int(SMBAction.LEFT_JUMP)].cpu().item()),
    }
    if motor_primitives is not None:
        for field_name in ("confidence", "replan_probability", "hold_duration", "interrupt_logit"):
            try:
                value = getattr(motor_primitives, field_name)[:, -1][0]
            except (AttributeError, IndexError, TypeError):
                continue
            snapshot[field_name] = float(value.detach().cpu().item())
    return snapshot


def _motor_summary(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "confidence",
        "replan_probability",
        "hold_duration",
        "interrupt_logit",
        "right_jump_bias",
        "left_jump_bias",
    )
    means = {}
    for field_name in numeric_fields:
        values = [float(item[field_name]) for item in snapshots if field_name in item]
        means[f"mean_{field_name}"] = float(sum(values) / len(values)) if values else 0.0
    return {
        "samples": len(snapshots),
        **means,
        "snapshots": list(snapshots[:8]),
    }


def _recording_actions(data: Mapping[str, Any]) -> list[int]:
    if "actions" in data:
        return [int(action) for action in np.asarray(data["actions"]).flatten().tolist()]
    if "action_names" not in data:
        return []
    by_name = {action.name: int(action) for action in SMB_ACTIONS}
    return [
        by_name[str(name)]
        for name in np.asarray(data["action_names"]).flatten().tolist()
        if str(name) in by_name
    ]


def _recording_progress(data: Mapping[str, Any]) -> list[float]:
    if "signals_json" not in data:
        return []
    values = []
    for item in np.asarray(data["signals_json"]).flatten().tolist():
        try:
            payload = json.loads(str(item))
        except json.JSONDecodeError:
            continue
        progress = payload.get("progress")
        try:
            number = float(progress)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            values.append(number)
    return values


def _iter_recording_npz_paths(paths: Sequence[Path]) -> Iterable[Path]:
    for path in paths:
        path = Path(path)
        if path.is_dir():
            yield from sorted(path.glob("**/*.npz"))
        elif path.suffix == ".npz":
            yield path


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


def _progress_from_info(info: Mapping[str, Any]) -> Optional[float]:
    signals = info.get("full_smb_signals")
    source = signals if isinstance(signals, Mapping) else info
    value = source.get("progress") if isinstance(source, Mapping) else None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="retroagi diagnose-actions --stage full")
    parser.add_argument("--policy-checkpoint", "--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--full-smb-vision-checkpoint",
        "--vision-checkpoint",
        type=Path,
        default=DEFAULT_FULL_SMB_VIT_CHECKPOINT,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--sample-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--progress-gate", type=float, default=DEFAULT_FULL_SMB_ACTION_PROGRESS_GATE)
    parser.add_argument("--game-id", default=DEFAULT_FULL_SMB_CONTENT.game)
    parser.add_argument("--state", default="Level1-1")
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--recording", dest="recordings", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
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
        result = run_full_smb_action_contract_diagnostic(
            model,
            stage,
            device=device,
            samples=args.samples,
            seed=args.seed,
            sample_repeats=args.sample_repeats,
            temperature=args.temperature,
            progress_gate=args.progress_gate,
            recording_paths=tuple(args.recordings or ()),
        )
    finally:
        stage.close()

    output = json.dumps(to_plain_data(result), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
