"""Backend provider contracts for game stage adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Mapping, Protocol

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
                raise ValueError(
                    f"backend spec {self.name!r} must define {field_name}"
                )

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
            truncated = bool(
                info.get("truncated", False)
                or info.get("TimeLimit.truncated", False)
            )
            return BackendStepResult(
                observation=observation,
                reward=float(reward),
                terminated=bool(done and not truncated),
                truncated=truncated,
                info=info,
            )
        raise ValueError(
            f"{self.context} step must return 4 values (Gym) or 5 values "
            "(Gymnasium)"
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
        raise RuntimeError(
            f"{self.context} must expose get_state/set_state on env or env.em"
        )

    @staticmethod
    def _unpack_reset(result: Any) -> BackendResetResult:
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], Mapping)
        ):
            return BackendResetResult(result[0], dict(result[1]))
        return BackendResetResult(result, {})


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
