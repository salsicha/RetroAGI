"""Full-SMB stable-retro adapter for the shared stage contract."""

from dataclasses import dataclass
import inspect
from typing import Any, Mapping, Optional

import numpy as np
import torch

from retroagi.core import (
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
)


@dataclass(frozen=True)
class FullSMBEnvConfig:
    """Backend selection for the stable-retro Full SMB environment."""

    game: str = FULL_SMB_GAME
    state: Optional[str] = None
    scenario: Optional[str] = None


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
        vision: Optional[VisionEncoder] = None,
        env_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.env_config = env_config
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
        self.last_info = self._info(info)
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        return observation

    def step(
        self, action: SMBAction | int
    ) -> tuple[np.ndarray, float, bool, bool, Mapping[str, Any]]:
        shared_action = coerce_smb_action(action)
        button_action = full_smb_action(shared_action, self.buttons)
        result = self.env.step(button_action)
        observation, reward, terminated, truncated, info = self._unpack_step(result)
        observation = self._rgb_observation(observation)
        info = self._info(info)
        info["action"] = {
            "shared_id": int(shared_action),
            "shared_name": shared_action.name,
            "buttons": self.buttons,
            "button_vector": button_action.tolist(),
        }
        self.last_info = info
        self._last_episode_mask = 0.0 if terminated or truncated else 1.0
        self._last_terminal = terminated
        self._last_truncated = truncated
        return observation, float(reward), terminated, truncated, info

    def encode_observation(
        self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None
    ) -> StageBatch:
        info = info or self.last_info
        observation = self._rgb_observation(observation)
        with torch.no_grad():
            vision = self.vision.encode(observation)

        return self.vision_projector.project(
            vision,
            state=self._state_vec(info),
            metadata={
                "raw_observation_shape": observation.shape,
                "observation": {"normalized_range": (0.0, 1.0)},
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

    @staticmethod
    def _state_vec(info: Mapping[str, Any]) -> Optional[np.ndarray]:
        if "state_vec" not in info:
            return None
        state = np.asarray(info["state_vec"], dtype=np.float32)
        if state.ndim != 1:
            raise ValueError("Full SMB info['state_vec'] must be a 1D vector")
        return np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0)


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
