"""Full-SMB stable-retro adapter for the shared stage contract."""

from collections import deque
import copy
from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core import (
    SMB_GAME_SPEC,
    SMBAction,
    StageBatch,
    StageSpec,
    VisionEncoder,
    VisionHierarchyProjector,
    coerce_smb_action,
    full_smb_action,
)
from retroagi.stages.full_smb.vision import FullSMBSegmentationVision

FULL_SMB_GAME = "SuperMarioBros-Nes"

FULL_SMB_SPEC = StageSpec(
    name="full_smb",
    observation_kind="stable-retro SuperMarioBros-Nes RGB frame",
    action_kind="shared SMBAction vocabulary mapped to stable-retro buttons",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
    action_space_name=SMB_GAME_SPEC.name,
    action_count=SMB_GAME_SPEC.action_count,
    action_names=tuple(action.name for action in SMB_GAME_SPEC.action_space),
)


@dataclass(frozen=True)
class FullSMBEnvConfig:
    """Backend selection for the stable-retro Full SMB environment."""

    game: str = FULL_SMB_GAME
    state: Optional[str] = None
    scenario: Optional[str] = None


@dataclass(frozen=True)
class FullSMBSignalConfig:
    """Normalization scales for raw stable-retro game-variable signals."""

    position_x_max: float = 4096.0
    position_y_max: float = 240.0
    score_max: float = 999_999.0
    coins_max: float = 99.0
    lives_max: float = 99.0

    def __post_init__(self) -> None:
        for name in (
            "position_x_max",
            "position_y_max",
            "score_max",
            "coins_max",
            "lives_max",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class FullSMBObservationConfig:
    """Preprocessing contract for Full SMB policy observations."""

    frame_skip: int = 1
    frame_stack: int = 4
    resize_shape: Optional[tuple[int, int]] = (224, 256)

    def __post_init__(self) -> None:
        if self.frame_skip <= 0:
            raise ValueError("frame_skip must be positive")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")
        if self.resize_shape is not None:
            if len(self.resize_shape) != 2:
                raise ValueError("resize_shape must contain (height, width)")
            height, width = self.resize_shape
            if height <= 0 or width <= 0:
                raise ValueError("resize_shape dimensions must be positive")


@dataclass(frozen=True)
class FullSMBEmulatorState:
    """Snapshot of backend emulator state plus adapter-owned rollout state."""

    backend_state: Any
    observation: np.ndarray
    last_info: dict[str, Any]
    episode_mask: float
    terminated: bool
    truncated: bool
    frame_stack: tuple[torch.Tensor, ...]
    frame_mask: tuple[bool, ...]


@dataclass(frozen=True)
class FullSMBSignals:
    """Normalized view of game variables emitted by the Full SMB backend."""

    position: Optional[tuple[float, float]]
    score: Optional[int]
    coins: Optional[int]
    lives: Optional[int]
    completion: bool
    death: bool
    terminated: bool
    truncated: bool
    termination_reason: Optional[str] = None

    def to_state_vec(
        self, config: FullSMBSignalConfig = FullSMBSignalConfig()
    ) -> np.ndarray:
        x, y = self.position if self.position is not None else (0.0, 0.0)
        return np.asarray(
            [
                _normalize_feature(x, config.position_x_max),
                _normalize_feature(y, config.position_y_max),
                _normalize_feature(self.score, config.score_max),
                _normalize_feature(self.coins, config.coins_max),
                _normalize_feature(self.lives, config.lives_max),
                float(self.completion),
                float(self.death),
                float(self.terminated),
                float(self.truncated),
            ],
            dtype=np.float32,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": self.position,
            "score": self.score,
            "coins": self.coins,
            "lives": self.lives,
            "completion": self.completion,
            "death": self.death,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "termination_reason": self.termination_reason,
        }


POSITION_X_KEYS = (
    "x",
    "x_pos",
    "x_position",
    "xpos",
    "mario_x",
    "player_x",
)
POSITION_Y_KEYS = (
    "y",
    "y_pos",
    "y_position",
    "ypos",
    "mario_y",
    "player_y",
)
SCREEN_X_KEYS = ("screen_x", "screenX", "x_screen")
SCROLL_X_KEYS = ("xscroll", "x_scroll", "scroll_x", "camera_x")
SCROLL_X_LO_KEYS = ("xscrollLo", "x_scroll_lo", "scroll_x_lo")
SCROLL_X_HI_KEYS = ("xscrollHi", "x_scroll_hi", "scroll_x_hi")
SCORE_KEYS = ("score", "Score")
COIN_KEYS = ("coins", "coin", "coin_count", "coins_collected")
LIFE_KEYS = ("lives", "life", "lives_left")
COMPLETION_KEYS = ("completion", "complete", "level_complete", "flag_get", "goal_reached")
DEATH_KEYS = ("death", "dead", "died", "player_dead", "is_dead", "game_over")
REASON_KEYS = ("terminal_reason", "termination_reason", "done_reason", "end_reason", "reason")


def make_stable_retro_env(
    config: FullSMBEnvConfig = FullSMBEnvConfig(),
    **kwargs: Any,
):
    """Create the stable-retro backend lazily so tests do not require ROM setup."""

    try:
        import retro
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Full SMB requires stable-retro and an imported SuperMarioBros-Nes ROM"
        ) from exc

    make_kwargs: dict[str, Any] = {"game": config.game}
    if config.state is not None:
        make_kwargs["state"] = config.state
    if config.scenario is not None:
        make_kwargs["scenario"] = config.scenario
    make_kwargs.update(kwargs)
    return retro.make(**make_kwargs)


class FullSMBStage:
    """Stage adapter for the full Super Mario Bros stable-retro backend."""

    spec = FULL_SMB_SPEC

    def __init__(
        self,
        env: Any = None,
        env_config: FullSMBEnvConfig = FullSMBEnvConfig(),
        signal_config: FullSMBSignalConfig = FullSMBSignalConfig(),
        observation_config: FullSMBObservationConfig = FullSMBObservationConfig(),
        vision: Optional[VisionEncoder] = None,
        env_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.env_config = env_config
        self.signal_config = signal_config
        self.observation_config = observation_config
        self.env = (
            env
            if env is not None
            else make_stable_retro_env(env_config, **dict(env_kwargs or {}))
        )
        self.vision = vision or FullSMBSegmentationVision()
        if isinstance(self.vision, torch.nn.Module):
            self.vision.eval()
        self.vision_projector = VisionHierarchyProjector(self.spec)
        self.last_info: Mapping[str, Any] = {}
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        self._frame_stack: deque[torch.Tensor] = deque(
            maxlen=self.observation_config.frame_stack
        )
        self._frame_mask: deque[bool] = deque(
            maxlen=self.observation_config.frame_stack
        )
        self._last_observation: Optional[np.ndarray] = None

    @property
    def buttons(self) -> tuple[str, ...]:
        buttons = getattr(self.env, "buttons", None)
        if buttons is None:
            raise ValueError("stable-retro environment does not expose button names")
        return tuple(str(button) for button in buttons)

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        result = self._reset_backend(seed=seed)
        observation, info = self._unpack_reset(result)
        observation = self._rgb_observation(observation)
        self.last_info = self._annotated_info(
            info, terminated=False, truncated=False
        )
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        self._reset_frame_stack(observation)
        self._last_observation = observation.copy()
        return observation

    def step(
        self, action: SMBAction | int
    ) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
        shared_action = coerce_smb_action(action)
        button_action = full_smb_action(shared_action, self.buttons)
        total_reward = 0.0
        frame_rewards: list[float] = []
        observation = None
        info: Mapping[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(self.observation_config.frame_skip):
            result = self.env.step(button_action)
            observation, reward, terminated, truncated, info = self._unpack_step(result)
            observation = self._rgb_observation(observation)
            total_reward += reward
            frame_rewards.append(float(reward))
            self._append_frame(observation, valid=True)
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("Full SMB frame_skip must execute at least one frame")
        info = self._annotated_info(
            info, terminated=terminated, truncated=truncated
        )
        info["action"] = {
            "shared_id": int(shared_action),
            "shared_name": shared_action.name,
            "buttons": self.buttons,
            "button_vector": button_action.tolist(),
            "frame_skip": self.observation_config.frame_skip,
            "frames_executed": len(frame_rewards),
            "frame_rewards": frame_rewards,
        }
        self.last_info = info
        self._last_episode_mask = 0.0 if terminated or truncated else 1.0
        self._last_terminal = terminated
        self._last_truncated = truncated
        self._last_observation = observation.copy()
        return observation, total_reward, terminated, truncated, info

    def save_emulator_state(self) -> FullSMBEmulatorState:
        """Snapshot backend emulator state and adapter metadata."""

        if self._last_observation is None:
            raise RuntimeError("cannot save Full SMB emulator state before reset")
        owner = self._emulator_state_owner()
        return FullSMBEmulatorState(
            backend_state=copy.deepcopy(owner.get_state()),
            observation=self._last_observation.copy(),
            last_info=copy.deepcopy(dict(self.last_info)),
            episode_mask=float(self._last_episode_mask),
            terminated=bool(self._last_terminal),
            truncated=bool(self._last_truncated),
            frame_stack=tuple(frame.clone() for frame in self._frame_stack),
            frame_mask=tuple(bool(item) for item in self._frame_mask),
        )

    def load_emulator_state(self, state: FullSMBEmulatorState) -> np.ndarray:
        """Restore a snapshot and return its current RGB observation."""

        if not isinstance(state, FullSMBEmulatorState):
            raise TypeError("state must be a FullSMBEmulatorState")
        if len(state.frame_stack) > self.observation_config.frame_stack:
            raise ValueError("saved frame stack is larger than this stage config")
        owner = self._emulator_state_owner()
        owner.set_state(copy.deepcopy(state.backend_state))
        observation = self._rgb_observation(state.observation)
        self._last_observation = observation.copy()
        self.last_info = copy.deepcopy(state.last_info)
        self._last_episode_mask = float(state.episode_mask)
        self._last_terminal = bool(state.terminated)
        self._last_truncated = bool(state.truncated)
        self._frame_stack.clear()
        self._frame_mask.clear()
        for frame in state.frame_stack:
            self._frame_stack.append(frame.clone())
        for valid in state.frame_mask:
            self._frame_mask.append(bool(valid))
        return observation

    def encode_observation(
        self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None
    ) -> StageBatch:
        info = info or self.last_info
        observation = self._rgb_observation(observation)
        processed_observation = self._preprocess_observation(observation)
        if not self._frame_stack:
            self._reset_frame_stack(observation)
        elif not torch.equal(self._frame_stack[-1], processed_observation):
            self._append_frame(observation, valid=True)
        with torch.no_grad():
            vision = self.vision.encode(processed_observation)

        return self.vision_projector.project(
            vision,
            state=self._state_vec(info),
            metadata={
                "raw_observation_shape": observation.shape,
                "observation": self._observation_metadata(vision.position.device),
                "episode": {
                    "mask": torch.tensor(
                        [self._last_episode_mask],
                        dtype=torch.float32,
                        device=vision.position.device,
                    ),
                    "terminated": self._last_terminal,
                    "truncated": self._last_truncated,
                },
                "info": info,
            },
        )

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if close is not None:
            close()

    def _reset_backend(self, seed: Optional[int]):
        if seed is None:
            return self.env.reset()
        reset = self.env.reset
        if _call_accepts_keyword(reset, "seed"):
            return reset(seed=seed)
        seed_fn = getattr(self.env, "seed", None)
        if seed_fn is not None:
            seed_fn(seed)
        return reset()

    def _emulator_state_owner(self):
        if _has_state_api(self.env):
            return self.env
        emulator = getattr(self.env, "em", None)
        if _has_state_api(emulator):
            return emulator
        raise RuntimeError(
            "Full SMB backend must expose get_state/set_state on env or env.em"
        )

    @staticmethod
    def _rgb_observation(observation: Any) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(
                "Full SMB observations must have shape [H, W, C] with RGB or RGBA channels"
            )
        array = array[..., :3]
        if array.dtype != np.uint8:
            array = array.astype(np.float32)
            if bool(array.size) and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
            array = np.clip(array, 0.0, 255.0).round().astype(np.uint8)
        return np.ascontiguousarray(array)

    def _reset_frame_stack(self, observation: np.ndarray) -> None:
        self._frame_stack.clear()
        self._frame_mask.clear()
        processed = self._preprocess_observation(observation)
        padding = self.observation_config.frame_stack - 1
        for _ in range(padding):
            self._frame_stack.append(processed.clone())
            self._frame_mask.append(False)
        self._frame_stack.append(processed)
        self._frame_mask.append(True)

    def _append_frame(self, observation: np.ndarray, *, valid: bool) -> None:
        self._frame_stack.append(self._preprocess_observation(observation))
        self._frame_mask.append(valid)

    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError("Full SMB RGB observations must have shape [H, W, 3]")
        if bool(tensor.numel()) and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        tensor = tensor.clamp(0.0, 1.0)
        if self.observation_config.resize_shape is None:
            return tensor.contiguous()
        target_shape = self.observation_config.resize_shape
        if tuple(tensor.shape[:2]) == target_shape:
            return tensor.contiguous()
        chw = tensor.permute(2, 0, 1).unsqueeze(0)
        resized = F.interpolate(
            chw, size=target_shape, mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).permute(1, 2, 0).contiguous()

    def _observation_metadata(self, device: torch.device) -> dict[str, Any]:
        frame_stack = torch.stack(tuple(self._frame_stack), dim=0).permute(
            0, 3, 1, 2
        )
        return {
            "frame_stack": frame_stack.unsqueeze(0).to(device),
            "frame_mask": torch.tensor(
                tuple(self._frame_mask), dtype=torch.bool, device=device
            ).unsqueeze(0),
            "frame_stack_size": self.observation_config.frame_stack,
            "frame_skip": self.observation_config.frame_skip,
            "resize_shape": self.observation_config.resize_shape,
            "normalized_range": (0.0, 1.0),
        }

    @staticmethod
    def _unpack_reset(result: Any) -> tuple[Any, Mapping[str, Any]]:
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[1], Mapping)
        ):
            return result[0], result[1]
        return result, {}

    @staticmethod
    def _unpack_step(result: Any) -> tuple[Any, float, bool, bool, Mapping[str, Any]]:
        if not isinstance(result, tuple):
            raise ValueError("stable-retro step must return a tuple")
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
            return observation, float(reward), bool(terminated), bool(truncated), info
        if len(result) == 4:
            observation, reward, done, info = result
            info = FullSMBStage._info(info)
            truncated = bool(
                info.get("truncated", False)
                or info.get("TimeLimit.truncated", False)
            )
            return (
                observation,
                float(reward),
                bool(done and not truncated),
                truncated,
                info,
            )
        raise ValueError(
            "stable-retro step must return 4 values (Gym) or 5 values (Gymnasium)"
        )

    @staticmethod
    def _info(info: Any) -> dict[str, Any]:
        if info is None:
            return {}
        if not isinstance(info, Mapping):
            raise ValueError("stable-retro info must be a mapping")
        return dict(info)

    def _annotated_info(
        self, info: Any, *, terminated: bool, truncated: bool
    ) -> dict[str, Any]:
        annotated = self._info(info)
        signals = extract_full_smb_signals(
            annotated, terminated=terminated, truncated=truncated
        )
        state_vec = signals.to_state_vec(self.signal_config)
        annotated["full_smb_signals"] = signals.as_dict()
        annotated["state_vec"] = state_vec
        return annotated

    @staticmethod
    def _state_vec(info: Mapping[str, Any]) -> Optional[np.ndarray]:
        if "state_vec" not in info:
            return None
        state = np.asarray(info["state_vec"], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError("Full SMB info['state_vec'] must be a 1D vector")
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


def extract_full_smb_signals(
    info: Mapping[str, Any], *, terminated: bool, truncated: bool
) -> FullSMBSignals:
    """Extract common stable-retro SMB variables into a stable schema."""

    reason = _string_value(info, REASON_KEYS)
    completion = _bool_value(info, COMPLETION_KEYS, default=False) or _reason_matches(
        reason, ("complete", "completed", "clear", "cleared", "goal", "flag")
    )
    death = _bool_value(info, DEATH_KEYS, default=False) or _reason_matches(
        reason, ("death", "dead", "died", "game_over", "game over")
    )
    return FullSMBSignals(
        position=_position_value(info),
        score=_int_value(info, SCORE_KEYS),
        coins=_int_value(info, COIN_KEYS),
        lives=_int_value(info, LIFE_KEYS),
        completion=completion,
        death=death,
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=reason,
    )


def _call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    for parameter in signature.parameters.values():
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.name == keyword:
            return True
    return False


def _has_state_api(owner: Any) -> bool:
    return callable(getattr(owner, "get_state", None)) and callable(
        getattr(owner, "set_state", None)
    )


def _position_value(info: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    direct = info.get("position")
    if isinstance(direct, Mapping):
        x = _numeric_value(direct, POSITION_X_KEYS)
        y = _numeric_value(direct, POSITION_Y_KEYS)
        if x is not None and y is not None:
            return (x, y)
    elif direct is not None:
        array = np.asarray(direct, dtype=np.float32).flatten()
        if array.size >= 2:
            return (float(array[0]), float(array[1]))

    x = _numeric_value(info, POSITION_X_KEYS)
    y = _numeric_value(info, POSITION_Y_KEYS)
    if x is None:
        x = _scroll_position_x(info)
    if x is not None and y is not None:
        return (x, y)
    return None


def _scroll_position_x(info: Mapping[str, Any]) -> Optional[float]:
    scroll_x = _numeric_value(info, SCROLL_X_KEYS)
    if scroll_x is None:
        scroll_lo = _numeric_value(info, SCROLL_X_LO_KEYS)
        scroll_hi = _numeric_value(info, SCROLL_X_HI_KEYS)
        if scroll_lo is not None and scroll_hi is not None:
            scroll_x = scroll_hi * 256.0 + scroll_lo
    if scroll_x is None:
        return None
    screen_x = _numeric_value(info, SCREEN_X_KEYS)
    return scroll_x + (screen_x or 0.0)


def _int_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    value = _numeric_value(info, keys)
    if value is None:
        return None
    return int(round(value))


def _numeric_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        try:
            array = np.asarray(value, dtype=np.float32).flatten()
        except (TypeError, ValueError):
            continue
        if array.size == 1 and np.isfinite(array[0]):
            return float(array[0])
    return None


def _bool_value(
    info: Mapping[str, Any], keys: tuple[str, ...], *, default: bool
) -> bool:
    for key in keys:
        if key not in info:
            continue
        value = info[key]
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "complete", "dead"}:
                return True
            if normalized in {"0", "false", "no", "n", ""}:
                return False
        try:
            array = np.asarray(value).flatten()
        except (TypeError, ValueError):
            continue
        if array.size == 1:
            return bool(array[0])
    return default


def _string_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = info.get(key)
        if value is not None:
            return str(value)
    return None


def _reason_matches(reason: Optional[str], needles: tuple[str, ...]) -> bool:
    if reason is None:
        return False
    normalized = reason.strip().lower()
    return any(needle in normalized for needle in needles)


def _normalize_feature(value: Optional[float], scale: float) -> float:
    if value is None:
        return 0.0
    return float(np.clip(float(value) / scale, 0.0, 1.0))
