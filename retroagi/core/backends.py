"""Backend provider contracts for game stage adapters."""

from __future__ import annotations

import copy
import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

import numpy as np

BACKEND_PROVIDER_KINDS = (
    "stable_retro",
    "native_python",
    "gymnasium",
    "custom",
)


@dataclass(frozen=True)
class BackendCapabilitySpec:
    """Declared backend capabilities used by smoke tests and manifests."""

    reset_seed: bool
    save_load_state: bool
    frame_step: bool
    action_repeat: bool
    render: bool
    headless: bool
    gymnasium_step_api: bool = True
    legacy_gym_step_api: bool = False

    def to_manifest(self) -> dict[str, bool]:
        return {
            "reset_seed": self.reset_seed,
            "save_load_state": self.save_load_state,
            "frame_step": self.frame_step,
            "action_repeat": self.action_repeat,
            "render": self.render,
            "headless": self.headless,
            "gymnasium_step_api": self.gymnasium_step_api,
            "legacy_gym_step_api": self.legacy_gym_step_api,
        }


@dataclass(frozen=True)
class GameBackendSpec:
    """Game-owned backend provider declaration."""

    name: str
    provider_kind: str
    entrypoint: str
    observation_api: str
    action_api: str
    reset_api: str = "reset(seed=None)"
    step_api: str = "step(action)"
    state_api: str = ""
    capabilities: BackendCapabilitySpec = field(
        default_factory=lambda: BackendCapabilitySpec(
            reset_seed=False,
            save_load_state=False,
            frame_step=True,
            action_repeat=False,
            render=False,
            headless=True,
        )
    )
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("backend spec name must be non-empty")
        if self.provider_kind not in BACKEND_PROVIDER_KINDS:
            raise ValueError(
                f"backend spec {self.name!r} provider_kind must be one of "
                f"{BACKEND_PROVIDER_KINDS}"
            )
        for field_name in (
            "entrypoint",
            "observation_api",
            "action_api",
            "reset_api",
            "step_api",
        ):
            if not getattr(self, field_name):
                raise ValueError(f"backend spec {self.name!r} must define {field_name}")

    def to_manifest(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider_kind": self.provider_kind,
            "entrypoint": self.entrypoint,
            "observation_api": self.observation_api,
            "action_api": self.action_api,
            "reset_api": self.reset_api,
            "step_api": self.step_api,
            "state_api": self.state_api,
            "capabilities": self.capabilities.to_manifest(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class BackendResetResult:
    observation: Any
    info: Mapping[str, Any]


@dataclass(frozen=True)
class BackendStepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


@dataclass(frozen=True)
class BackendCapabilityProbeConfig:
    """Inputs for deterministic backend capability probes."""

    seed: int = 123
    action: Any = 0
    action_repeat: int = 2

    def __post_init__(self) -> None:
        if self.action_repeat <= 0:
            raise ValueError("action_repeat must be positive")


@dataclass(frozen=True)
class BackendCapabilityReport:
    """Pass/fail result for backend lifecycle capabilities."""

    reset_seed: bool
    save_load_state: bool
    frame_step: bool
    action_repeat: bool
    render: bool
    headless: bool
    failures: Mapping[str, str] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(
            (
                self.reset_seed,
                self.save_load_state,
                self.frame_step,
                self.action_repeat,
                self.render,
                self.headless,
            )
        )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "reset_seed": self.reset_seed,
            "save_load_state": self.save_load_state,
            "frame_step": self.frame_step,
            "action_repeat": self.action_repeat,
            "render": self.render,
            "headless": self.headless,
            "passed": self.passed,
            "failures": dict(self.failures),
        }


class BackendAdapter(Protocol):
    """Normalized lifecycle API for stage-native backends."""

    @property
    def buttons(self) -> tuple[str, ...]:
        """Return backend-native button names when the action API uses buttons."""

    def reset(self, seed: int | None = None) -> BackendResetResult:
        """Reset the backend and normalize the return value."""

    def step(self, action: Any) -> BackendStepResult:
        """Step the backend and normalize legacy Gym or Gymnasium outputs."""

    def render(self) -> Any:
        """Render the backend if supported."""

    def close(self) -> None:
        """Close backend resources if supported."""

    def get_state(self) -> Any:
        """Return a backend save-state payload."""

    def set_state(self, state: Any) -> None:
        """Restore a backend save-state payload."""


class GymnasiumBackendAdapter:
    """Adapter for Gymnasium, legacy Gym, stable-retro, and similar envs."""

    def __init__(
        self,
        env: Any,
        *,
        button_names: tuple[str, ...] | None = None,
        context: str = "backend",
    ):
        self.env = env
        self._button_names = button_names
        self.context = context

    @property
    def buttons(self) -> tuple[str, ...]:
        buttons = self._button_names
        if buttons is None:
            buttons = getattr(self.env, "buttons", None)
        if buttons is None:
            raise ValueError(f"{self.context} does not expose button names")
        return tuple(str(button) for button in buttons)

    def reset(self, seed: int | None = None) -> BackendResetResult:
        if seed is None:
            return self._unpack_reset(self.env.reset())
        reset = self.env.reset
        if _call_accepts_keyword(reset, "seed"):
            return self._unpack_reset(reset(seed=seed))
        seed_fn = getattr(self.env, "seed", None)
        if seed_fn is not None:
            seed_fn(seed)
        return self._unpack_reset(reset())

    def step(self, action: Any) -> BackendStepResult:
        result = self.env.step(action)
        if not isinstance(result, tuple):
            raise ValueError(f"{self.context} step must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return BackendStepResult(
                observation=observation,
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=_info(info, context=self.context),
            )
        if len(result) == 4:
            observation, reward, done, info = result
            info = _info(info, context=self.context)
            truncated = bool(info.get("truncated", False) or info.get("TimeLimit.truncated", False))
            return BackendStepResult(
                observation=observation,
                reward=float(reward),
                terminated=bool(done and not truncated),
                truncated=truncated,
                info=info,
            )
        raise ValueError(
            f"{self.context} step must return 4 values (Gym) or 5 values " "(Gymnasium)"
        )

    def render(self) -> Any:
        render = getattr(self.env, "render", None)
        if render is None:
            raise RuntimeError(f"{self.context} does not expose render")
        return render()

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def get_state(self) -> Any:
        return self._state_owner().get_state()

    def set_state(self, state: Any) -> None:
        self._state_owner().set_state(state)

    def _state_owner(self) -> Any:
        if _has_state_api(self.env):
            return self.env
        emulator = getattr(self.env, "em", None)
        if _has_state_api(emulator):
            return emulator
        raise RuntimeError(f"{self.context} must expose get_state/set_state on env or env.em")

    @staticmethod
    def _unpack_reset(result: Any) -> BackendResetResult:
        if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], Mapping):
            return BackendResetResult(result[0], dict(result[1]))
        return BackendResetResult(result, {})


def probe_backend_capabilities(
    backend: BackendAdapter,
    config: BackendCapabilityProbeConfig = BackendCapabilityProbeConfig(),
) -> BackendCapabilityReport:
    """Run deterministic lifecycle probes against a backend adapter."""

    failures: dict[str, str] = {}

    def run(name: str, check) -> bool:
        try:
            return bool(check())
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            return False

    reset_seed = run(
        "reset_seed",
        lambda: _fingerprint(backend.reset(config.seed))
        == _fingerprint(backend.reset(config.seed)),
    )
    frame_step = run(
        "frame_step",
        lambda: _valid_step_result(
            _step_from_seed(backend, seed=config.seed, action=config.action)
        ),
    )
    save_load_state = run(
        "save_load_state",
        lambda: _save_load_replays_step(backend, seed=config.seed, action=config.action),
    )
    action_repeat = run(
        "action_repeat",
        lambda: _action_repeat_replays(
            backend,
            seed=config.seed,
            action=config.action,
            action_repeat=config.action_repeat,
        ),
    )
    render = run("render", lambda: _render_available(backend))
    headless = run(
        "headless",
        lambda: _valid_step_result(
            _step_from_seed(
                backend,
                seed=config.seed + 1,
                action=config.action,
            )
        ),
    )

    return BackendCapabilityReport(
        reset_seed=reset_seed,
        save_load_state=save_load_state,
        frame_step=frame_step,
        action_repeat=action_repeat,
        render=render,
        headless=headless,
        failures=failures,
    )


def _step_from_seed(
    backend: BackendAdapter,
    *,
    seed: int,
    action: Any,
) -> BackendStepResult:
    backend.reset(seed)
    return backend.step(action)


def _save_load_replays_step(
    backend: BackendAdapter,
    *,
    seed: int,
    action: Any,
) -> bool:
    backend.reset(seed)
    state = copy.deepcopy(backend.get_state())
    first = backend.step(action)
    backend.set_state(copy.deepcopy(state))
    replay = backend.step(action)
    return _fingerprint(first) == _fingerprint(replay)


def _action_repeat_replays(
    backend: BackendAdapter,
    *,
    seed: int,
    action: Any,
    action_repeat: int,
) -> bool:
    backend.reset(seed)
    state = copy.deepcopy(backend.get_state())
    first = tuple(_fingerprint(backend.step(action)) for _ in range(action_repeat))
    backend.set_state(copy.deepcopy(state))
    replay = tuple(_fingerprint(backend.step(action)) for _ in range(action_repeat))
    return first == replay


def _valid_step_result(result: BackendStepResult) -> bool:
    return (
        isinstance(result, BackendStepResult)
        and isinstance(result.reward, float)
        and isinstance(result.terminated, bool)
        and isinstance(result.truncated, bool)
        and isinstance(result.info, Mapping)
    )


def _render_available(backend: BackendAdapter) -> bool:
    backend.render()
    return True


def _info(info: Any, *, context: str) -> dict[str, Any]:
    if info is None:
        return {}
    if not isinstance(info, Mapping):
        raise ValueError(f"{context} info must be a mapping")
    return dict(info)


def _has_state_api(candidate: Any) -> bool:
    return callable(getattr(candidate, "get_state", None)) and callable(
        getattr(candidate, "set_state", None)
    )


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword and parameter.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


def _fingerprint(value: Any) -> Any:
    if isinstance(value, BackendResetResult):
        return (
            "reset",
            _fingerprint(value.observation),
            _fingerprint(dict(value.info)),
        )
    if isinstance(value, BackendStepResult):
        return (
            "step",
            _fingerprint(value.observation),
            float(value.reward),
            bool(value.terminated),
            bool(value.truncated),
            _fingerprint(dict(value.info)),
        )
    if isinstance(value, Mapping):
        return tuple(
            (repr(key), _fingerprint(item))
            for key, item in sorted(value.items(), key=lambda pair: repr(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_fingerprint(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_fingerprint(item) for item in value))
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, bytes, int, float, bool, type(None))):
        return value
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return repr(value)
    if array.dtype == object:
        return repr(value)
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
    return ("array", tuple(contiguous.shape), str(contiguous.dtype), digest)
