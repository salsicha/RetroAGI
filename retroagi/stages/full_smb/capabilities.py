"""Full SMB environment capability checks."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np

from retroagi.core import (
    BackendCapabilityProbeConfig,
    full_smb_action,
    probe_backend_capabilities,
    to_plain_data,
)
from retroagi.core.actions import SMBAction
from retroagi.stages.full_smb.adapter import (
    DEFAULT_FULL_SMB_CONTENT,
    FullSMBContentSpec,
    FullSMBEnvConfig,
    FullSMBObservationConfig,
    FullSMBStage,
)
from retroagi.stages.full_smb.smoke import run_deterministic_reset_smoke


@dataclass(frozen=True)
class FullSMBEnvironmentCheckConfig:
    """Configuration for the local Full SMB backend/content preflight."""

    seed: int = 0
    steps: int = 4
    frame_skip: int = 2
    action: SMBAction = SMBAction.RIGHT
    output: Optional[Path] = None
    env_config: FullSMBEnvConfig = field(default_factory=FullSMBEnvConfig)
    content_spec: FullSMBContentSpec = DEFAULT_FULL_SMB_CONTENT

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")


@dataclass(frozen=True)
class FullSMBEnvironmentCapabilityResult:
    """JSON-serializable Full SMB environment preflight result."""

    content: Mapping[str, Any]
    checks: Mapping[str, Mapping[str, Any]]
    backend_probe: Mapping[str, Any] | None = None
    deterministic_reset: Mapping[str, Any] | None = None

    @property
    def passed(self) -> bool:
        return all(bool(check.get("passed")) for check in self.checks.values())

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "content": dict(self.content),
            "checks": {key: dict(value) for key, value in self.checks.items()},
            "backend_probe": self.backend_probe,
            "deterministic_reset": self.deterministic_reset,
        }


def run_full_smb_environment_check(
    config: FullSMBEnvironmentCheckConfig = FullSMBEnvironmentCheckConfig(),
    *,
    retro_module_factory: Callable[[], Any] | None = None,
    stage_factory: Callable[[], FullSMBStage] | None = None,
) -> FullSMBEnvironmentCapabilityResult:
    """Verify local Full SMB backend, content, and deterministic lifecycle behavior."""

    checks: dict[str, dict[str, Any]] = {}
    backend_probe: Mapping[str, Any] | None = None
    deterministic_reset: Mapping[str, Any] | None = None
    retro_module = None

    try:
        retro_module = (
            retro_module_factory() if retro_module_factory is not None else _import_retro_module()
        )
        checks["backend_import"] = _check(
            True,
            details={"module": getattr(retro_module, "__name__", "retro")},
        )
    except Exception as exc:
        checks["backend_import"] = _check(
            False,
            reason=config.content_spec.setup_failure_message(
                f"{type(exc).__name__}: {exc}",
                game=config.env_config.game,
            ),
        )
        _mark_prerequisite_failures(checks, "backend_import")
        return FullSMBEnvironmentCapabilityResult(
            content=config.content_spec.to_manifest(),
            checks=checks,
        )

    registered = _registered_games(retro_module)
    if registered is not None:
        checks["game_registration"] = _check(
            config.env_config.game in registered,
            reason=(
                None
                if config.env_config.game in registered
                else f"{config.env_config.game!r} is not registered with stable-retro"
            ),
            details={"registered_games": sorted(registered)},
        )

    make_stage = stage_factory or (lambda: _make_full_smb_check_stage(config))
    stage = _try_make_stage(make_stage, config)
    if isinstance(stage, _FailedStage):
        checks.setdefault(
            "game_registration",
            _check(False, reason="cannot verify game registration before ROM availability"),
        )
        checks["rom_availability"] = _check(False, reason=stage.reason)
        _mark_prerequisite_failures(checks, "rom_availability")
        return FullSMBEnvironmentCapabilityResult(
            content=config.content_spec.to_manifest(),
            checks=checks,
        )

    checks["rom_availability"] = _check(
        True,
        details={"game": config.env_config.game},
    )
    checks["game_registration"] = _check(
        True,
        details={"verification": "stable-retro environment created successfully"},
    )
    try:
        native_action = full_smb_action(config.action, stage.buttons)
        backend_report = probe_backend_capabilities(
            stage.backend,
            BackendCapabilityProbeConfig(
                seed=config.seed,
                action=native_action,
                action_repeat=config.frame_skip,
            ),
        )
        backend_probe = backend_report.to_manifest()
        checks["save_load_state"] = _check_from_backend_probe(backend_probe, "save_load_state")
    finally:
        stage.close()

    checks["headless_reset"] = _run_stage_check(make_stage, config, _check_headless_reset)
    checks["render_reset"] = _run_stage_check(make_stage, config, _check_render_reset)
    checks["action_step"] = _run_stage_check(make_stage, config, _check_action_step)
    checks["frame_skip"] = _run_stage_check(make_stage, config, _check_frame_skip)

    try:
        deterministic = run_deterministic_reset_smoke(
            make_stage,
            seed=config.seed,
            steps=config.steps,
            encode_observations=False,
        )
        deterministic_reset = deterministic.as_dict()
        checks["deterministic_seeding"] = _check(
            deterministic.deterministic,
            reason=deterministic.mismatch,
            details={"steps": config.steps},
        )
    except Exception as exc:
        checks["deterministic_seeding"] = _check(False, reason=_error_message(exc))

    return FullSMBEnvironmentCapabilityResult(
        content=config.content_spec.to_manifest(),
        checks=checks,
        backend_probe=backend_probe,
        deterministic_reset=deterministic_reset,
    )


def _import_retro_module() -> Any:
    return importlib.import_module("retro")


def _make_full_smb_check_stage(config: FullSMBEnvironmentCheckConfig) -> FullSMBStage:
    return FullSMBStage(
        env_config=config.env_config,
        content_spec=config.content_spec,
        observation_config=FullSMBObservationConfig(
            frame_skip=config.frame_skip,
            frame_stack=2,
            resize_shape=None,
        ),
        vision=_NoopVision(),
    )


class _NoopVision:
    def encode(self, _observation: Any) -> Any:
        raise RuntimeError("Full SMB environment checks do not encode observations")


def _try_make_stage(
    make_stage: Callable[[], FullSMBStage],
    config: FullSMBEnvironmentCheckConfig,
) -> FullSMBStage | _FailedStage:
    try:
        return make_stage()
    except Exception as exc:
        return _FailedStage(
            config.content_spec.setup_failure_message(
                f"{type(exc).__name__}: {exc}",
                game=config.env_config.game,
            )
        )


class _FailedStage:
    def __init__(self, reason: str):
        self.reason = reason


def _mark_prerequisite_failures(checks: dict[str, dict[str, Any]], reason: str) -> None:
    for name in (
        "game_registration",
        "rom_availability",
        "headless_reset",
        "render_reset",
        "save_load_state",
        "action_step",
        "frame_skip",
        "deterministic_seeding",
    ):
        checks.setdefault(name, _check(False, reason=f"requires passing {reason}"))


def _run_stage_check(
    make_stage: Callable[[], FullSMBStage],
    config: FullSMBEnvironmentCheckConfig,
    check: Callable[[FullSMBStage, FullSMBEnvironmentCheckConfig], dict[str, Any]],
) -> dict[str, Any]:
    stage = make_stage()
    try:
        return check(stage, config)
    except Exception as exc:
        return _check(False, reason=_error_message(exc))
    finally:
        stage.close()


def _check_headless_reset(
    stage: FullSMBStage,
    config: FullSMBEnvironmentCheckConfig,
) -> dict[str, Any]:
    observation = stage.reset(seed=config.seed)
    array = np.asarray(observation)
    return _check(
        array.ndim == 3 and array.shape[-1] == 3,
        reason="reset did not return an RGB observation" if array.ndim != 3 else None,
        details={"shape": tuple(int(item) for item in array.shape)},
    )


def _check_action_step(
    stage: FullSMBStage,
    config: FullSMBEnvironmentCheckConfig,
) -> dict[str, Any]:
    stage.reset(seed=config.seed)
    observation, reward, terminated, truncated, info = stage.step(config.action)
    array = np.asarray(observation)
    return _check(
        array.ndim == 3 and array.shape[-1] == 3 and "action" in info,
        reason="step did not return RGB observation plus action metadata",
        details={
            "shape": tuple(int(item) for item in array.shape),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "action": dict(info.get("action", {})),
        },
    )


def _check_render_reset(
    stage: FullSMBStage,
    config: FullSMBEnvironmentCheckConfig,
) -> dict[str, Any]:
    stage.reset(seed=config.seed)
    rendered = stage.backend.render()
    details = {"rendered": rendered is not None}
    if rendered is not None:
        details["shape"] = tuple(int(item) for item in np.asarray(rendered).shape)
    return _check(True, details=details)


def _check_frame_skip(
    stage: FullSMBStage,
    config: FullSMBEnvironmentCheckConfig,
) -> dict[str, Any]:
    stage.reset(seed=config.seed)
    _observation, _reward, _terminated, _truncated, info = stage.step(config.action)
    action = info.get("action", {})
    frames_executed = int(action.get("frames_executed", 0))
    return _check(
        int(action.get("frame_skip", 0)) == config.frame_skip
        and 1 <= frames_executed <= config.frame_skip,
        reason="frame-skip metadata did not match configured frame_skip",
        details={
            "frame_skip": action.get("frame_skip"),
            "frames_executed": frames_executed,
        },
    )


def _check_from_backend_probe(
    backend_probe: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    failures = backend_probe.get("failures", {})
    reason = failures.get(name) if isinstance(failures, Mapping) else None
    return _check(bool(backend_probe.get(name)), reason=reason)


def _registered_games(retro_module: Any) -> set[str] | None:
    games: set[str] = set()
    for owner in (getattr(retro_module, "data", None), retro_module):
        if owner is None:
            continue
        for name in ("list_games", "list_integrations"):
            listing = getattr(owner, name, None)
            if not callable(listing):
                continue
            for values in _call_listing_variants(owner, listing):
                games.update(str(value) for value in values)
    return games or None


def _call_listing_variants(owner: Any, listing: Callable[..., Any]) -> list[list[Any]]:
    calls = [
        (),
    ]
    integrations = getattr(owner, "Integrations", None)
    if integrations is not None and getattr(integrations, "ALL", None) is not None:
        calls.append((getattr(integrations, "ALL"),))
    results = []
    for args in calls:
        try:
            value = listing(*args)
        except TypeError:
            if args:
                continue
            try:
                if _accepts_keyword(listing, "inttype") and len(calls) > 1:
                    value = listing(inttype=calls[1][0])
                else:
                    continue
            except Exception:
                continue
        except Exception:
            continue
        if isinstance(value, Mapping):
            results.append(list(value.keys()))
        elif isinstance(value, (str, bytes)):
            results.append([value.decode() if isinstance(value, bytes) else value])
        else:
            try:
                results.append(list(value))
            except TypeError:
                continue
    return results


def _accepts_keyword(callable_obj: Callable[..., Any], keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        or (
            parameter.name == keyword
            and parameter.kind
            in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        )
        for parameter in signature.parameters.values()
    )


def _check(
    passed: bool,
    *,
    reason: str | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "reason": None if passed else reason,
        "details": dict(details or {}),
    }


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="retroagi check-env --stage full-smb",
        description="Check the local Full SMB stable-retro environment and content setup.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--state")
    parser.add_argument("--scenario")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = FullSMBEnvironmentCheckConfig(
        seed=args.seed,
        steps=args.steps,
        frame_skip=args.frame_skip,
        output=args.output,
        env_config=FullSMBEnvConfig(state=args.state, scenario=args.scenario),
    )
    result = run_full_smb_environment_check(config)
    output = json.dumps(to_plain_data(result.as_dict()), indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
