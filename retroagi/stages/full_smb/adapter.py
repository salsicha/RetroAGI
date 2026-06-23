"""Full-SMB stable-retro adapter for the shared stage contract."""

from collections import deque
import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.nn.functional as F

from retroagi.core import (
    GameSignalExtractor,
    GameSignals,
    GymnasiumBackendAdapter,
    RewardConfigSchema,
    RewardTermSpec,
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
class FullSMBContentSpec:
    """Supported local content contract for real Full SMB emulator runs."""

    game: str = FULL_SMB_GAME
    rom_label: str = "Super Mario Bros NES ROM"
    stable_retro_import_dir: str = "local/full_smb/roms"
    checksum_dir: str = "local/full_smb/checksums"
    checksum_algorithm: str = "sha256"
    install_command: str = "python -m pip install -e '.[full-smb]'"
    legal_notice: str = (
        "Use a legally obtained, user-provided ROM. Do not commit, bundle, or "
        "redistribute ROM content with RetroAGI artifacts."
    )

    @property
    def stable_retro_import_command(self) -> str:
        return f"python -m retro.import {self.stable_retro_import_dir}"

    @property
    def checksum_record_path(self) -> str:
        return f"{self.checksum_dir}/{self.game}.{self.checksum_algorithm}"

    def to_manifest(self) -> dict[str, Any]:
        return {
            "game": self.game,
            "rom_label": self.rom_label,
            "stable_retro_import_dir": self.stable_retro_import_dir,
            "stable_retro_import_command": self.stable_retro_import_command,
            "checksum_algorithm": self.checksum_algorithm,
            "checksum_record_path": self.checksum_record_path,
            "install_command": self.install_command,
            "legal_notice": self.legal_notice,
        }

    def setup_failure_message(self, reason: str, *, game: Optional[str] = None) -> str:
        target_game = game or self.game
        return (
            f"Full SMB stable-retro setup failed for game {target_game!r}: {reason}\n"
            f"Install backend: {self.install_command}\n"
            f"Required content: {self.rom_label} imported as stable-retro game "
            f"{target_game!r}.\n"
            f"Local ROM staging directory: {self.stable_retro_import_dir}/ "
            "(ignored by git).\n"
            f"Import command: {self.stable_retro_import_command}\n"
            f"Checksum record: write the {self.checksum_algorithm} hash to "
            f"{self.checksum_record_path} and preserve it with the local run notes.\n"
            f"Legal/provenance: {self.legal_notice}"
        )


DEFAULT_FULL_SMB_CONTENT = FullSMBContentSpec()


FULL_SMB_REWARD_SCHEMA = RewardConfigSchema(
    game_name="full_smb_adapter",
    terms=(
        RewardTermSpec(
            name="emulator_progress",
            default=1.0,
            direction="positive",
            signal="stable_retro_reward",
            description="Scale applied to the stable-retro emulator progress reward",
        ),
        RewardTermSpec(
            name="completion",
            default=0.0,
            direction="positive",
            signal="full_smb_signals.completion",
            description="Bonus for clearing a Full SMB task or level section",
        ),
        RewardTermSpec(
            name="survival",
            default=0.0,
            direction="positive",
            signal="full_smb_signals.terminated",
            description="Per-transition or terminal bonus for staying alive",
        ),
        RewardTermSpec(
            name="score",
            default=0.0,
            direction="positive",
            signal="full_smb_signals.score",
            description="Reward weight for score deltas exposed by the emulator",
        ),
        RewardTermSpec(
            name="coin",
            default=0.0,
            direction="positive",
            signal="full_smb_signals.coins",
            description="Reward weight for coin-count deltas",
        ),
        RewardTermSpec(
            name="enemy",
            default=0.0,
            direction="positive",
            signal="full_smb_signals.objectives.enemy",
            description="Reward for defeating enemies when the backend exposes it",
        ),
        RewardTermSpec(
            name="damage",
            default=0.0,
            direction="negative",
            signal="full_smb_signals.objectives.damage",
            description="Penalty for damage or unsafe collisions",
        ),
        RewardTermSpec(
            name="death",
            default=0.0,
            direction="negative",
            signal="full_smb_signals.death",
            description="Terminal penalty for death or game over",
        ),
        RewardTermSpec(
            name="frame_penalty",
            default=0.0,
            direction="negative",
            signal="time",
            description="Per-executed-backend-frame time cost",
        ),
    ),
)


@dataclass(frozen=True)
class FullSMBRewardConfig:
    """Adapter-owned reward-term config for Full SMB training and play."""

    emulator_progress: float = 1.0
    completion: float = 0.0
    survival: float = 0.0
    score: float = 0.0
    coin: float = 0.0
    enemy: float = 0.0
    damage: float = 0.0
    death: float = 0.0
    frame_penalty: float = 0.0

    def __post_init__(self) -> None:
        FULL_SMB_REWARD_SCHEMA.validate(self.as_dict())

    @property
    def term_names(self) -> tuple[str, ...]:
        return FULL_SMB_REWARD_SCHEMA.term_names

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.term_names}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "owner": "full_smb_adapter",
            "schema": FULL_SMB_REWARD_SCHEMA.game_name,
            "defaults_preserve_backend_reward": True,
            "terms": self.as_dict(),
            "term_signals": {
                term.name: {
                    "direction": term.direction,
                    "signal": term.signal,
                    "description": term.description,
                }
                for term in FULL_SMB_REWARD_SCHEMA.terms
            },
            "separated_from": "BlockSMBRewardConfig",
        }


DEFAULT_FULL_SMB_REWARD_CONFIG = FullSMBRewardConfig()


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
    screen_x_max: float = 256.0
    screen_y_max: float = 240.0
    score_max: float = 999_999.0
    coins_max: float = 99.0
    lives_max: float = 99.0

    def __post_init__(self) -> None:
        for name in (
            "position_x_max",
            "position_y_max",
            "screen_x_max",
            "screen_y_max",
            "score_max",
            "coins_max",
            "lives_max",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


FULL_SMB_COLOR_MODES = ("rgb", "grayscale")
FULL_SMB_HUD_POLICIES = ("preserve", "crop")


@dataclass(frozen=True)
class FullSMBObservationConfig:
    """Preprocessing contract for Full SMB policy observations."""

    frame_skip: int = 1
    frame_stack: int = 4
    resize_shape: Optional[tuple[int, int]] = (224, 256)
    crop_margins: tuple[int, int, int, int] = (0, 0, 0, 0)
    hud_policy: str = "preserve"
    hud_crop_top: int = 24
    color_mode: str = "rgb"
    normalization_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normalization_std: tuple[float, float, float] = (1.0, 1.0, 1.0)
    include_camera_state: bool = False

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
        if len(self.crop_margins) != 4:
            raise ValueError("crop_margins must contain (top, right, bottom, left)")
        if any(int(value) < 0 for value in self.crop_margins):
            raise ValueError("crop_margins values must be non-negative")
        if self.hud_policy not in FULL_SMB_HUD_POLICIES:
            raise ValueError(
                f"hud_policy must be one of {', '.join(FULL_SMB_HUD_POLICIES)}"
            )
        if self.hud_crop_top < 0:
            raise ValueError("hud_crop_top must be non-negative")
        if self.color_mode not in FULL_SMB_COLOR_MODES:
            raise ValueError(
                f"color_mode must be one of {', '.join(FULL_SMB_COLOR_MODES)}"
            )
        if len(self.normalization_mean) != 3 or len(self.normalization_std) != 3:
            raise ValueError("normalization_mean/std must contain three RGB values")
        if any(float(value) <= 0.0 for value in self.normalization_std):
            raise ValueError("normalization_std values must be positive")

    def effective_crop_margins(self) -> tuple[int, int, int, int]:
        top, right, bottom, left = (int(value) for value in self.crop_margins)
        if self.hud_policy == "crop":
            top += int(self.hud_crop_top)
        return top, right, bottom, left

    def to_manifest(self) -> dict[str, Any]:
        return {
            "frame_skip": self.frame_skip,
            "frame_stack": self.frame_stack,
            "resize_shape": self.resize_shape,
            "crop_margins": self.crop_margins,
            "effective_crop_margins": self.effective_crop_margins(),
            "hud_policy": self.hud_policy,
            "hud_crop_top": self.hud_crop_top,
            "color_mode": self.color_mode,
            "normalization_mean": self.normalization_mean,
            "normalization_std": self.normalization_std,
            "include_camera_state": self.include_camera_state,
        }


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
class FullSMBSignals(GameSignals):
    """Normalized view of game variables emitted by the Full SMB backend."""

    coins: Optional[int] = None
    screen: Optional[tuple[int, int]] = None
    level: Optional[str] = None
    world: Optional[int] = None
    stage: Optional[int] = None
    power_state: Optional[str] = None
    game_over: bool = False

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
        data = super().as_dict()
        data["coins"] = self.coins
        data["screen"] = self.screen
        data["level"] = self.level
        data["world"] = self.world
        data["stage"] = self.stage
        data["power_state"] = self.power_state
        data["game_over"] = self.game_over
        return data


class FullSMBSignalExtractor(GameSignalExtractor):
    """Game-neutral signal extractor for Full SMB stable-retro info mappings."""

    game_name = SMB_GAME_SPEC.name

    def extract(
        self,
        info: Mapping[str, Any],
        *,
        terminated: bool,
        truncated: bool,
    ) -> FullSMBSignals:
        return extract_full_smb_signals(
            info, terminated=terminated, truncated=truncated
        )


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
INFO_CONTAINER_KEYS = (
    "memory",
    "ram",
    "variables",
    "game_variables",
    "backend_info",
)
SCREEN_KEYS = ("screen", "screen_position", "screen_pos")
SCREEN_X_KEYS = ("screen_x", "screenX", "x_screen")
SCREEN_Y_KEYS = ("screen_y", "screenY", "y_screen")
SCROLL_X_KEYS = ("xscroll", "x_scroll", "scroll_x", "camera_x")
SCROLL_X_LO_KEYS = ("xscrollLo", "x_scroll_lo", "scroll_x_lo")
SCROLL_X_HI_KEYS = ("xscrollHi", "x_scroll_hi", "scroll_x_hi")
SCORE_KEYS = ("score", "Score")
COIN_KEYS = ("coins", "coin", "coin_count", "coins_collected")
LIFE_KEYS = ("lives", "life", "lives_left")
WORLD_KEYS = ("world", "world_number", "world_id")
LEVEL_KEYS = ("level", "level_name", "level_id", "state", "area", "area_name")
STAGE_KEYS = ("stage", "stage_number", "stage_id", "area_number")
POWER_STATE_KEYS = (
    "power_state",
    "powerup",
    "power_up",
    "power",
    "mario_power",
    "player_power",
    "status",
    "mario_status",
    "player_status",
)
COMPLETION_KEYS = (
    "completion",
    "complete",
    "level_complete",
    "level_clear",
    "flag_get",
    "flag",
    "flagpole",
    "goal_reached",
)
DEATH_KEYS = ("death", "dead", "died", "player_dead", "is_dead")
GAME_OVER_KEYS = ("game_over", "gameOver", "GameOver", "is_game_over")
ENEMY_OBJECTIVE_KEYS = (
    "enemy",
    "enemy_defeat",
    "enemy_defeated",
    "enemy_stomp",
    "enemies_defeated",
    "kills",
    "kill_count",
)
DAMAGE_OBJECTIVE_KEYS = (
    "damage",
    "damaged",
    "damage_taken",
    "enemy_hit",
    "hit",
    "hurt",
    "power_down",
    "powerdown",
)
TIMEOUT_KEYS = (
    "timeout",
    "time_up",
    "timeUp",
    "TimeLimit.truncated",
    "timer_done",
    "out_of_time",
)
REASON_KEYS = (
    "terminal_reason",
    "termination_reason",
    "done_reason",
    "end_reason",
    "reason",
)
POWER_STATE_NAMES = {0: "small", 1: "big", 2: "fire"}


def make_stable_retro_env(
    config: FullSMBEnvConfig = FullSMBEnvConfig(),
    content_spec: FullSMBContentSpec = DEFAULT_FULL_SMB_CONTENT,
    **kwargs: Any,
):
    """Create the stable-retro backend lazily so tests do not require ROM setup."""

    try:
        import retro
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            content_spec.setup_failure_message(
                "stable-retro is not installed", game=config.game
            )
        ) from exc

    make_kwargs: dict[str, Any] = {"game": config.game}
    if config.state is not None:
        make_kwargs["state"] = config.state
    if config.scenario is not None:
        make_kwargs["scenario"] = config.scenario
    make_kwargs.update(kwargs)
    try:
        return retro.make(**make_kwargs)
    except Exception as exc:
        raise RuntimeError(
            content_spec.setup_failure_message(str(exc), game=config.game)
        ) from exc


class FullSMBStage:
    """Stage adapter for the full Super Mario Bros stable-retro backend."""

    spec = FULL_SMB_SPEC

    def __init__(
        self,
        env: Any = None,
        env_config: FullSMBEnvConfig = FullSMBEnvConfig(),
        content_spec: FullSMBContentSpec = DEFAULT_FULL_SMB_CONTENT,
        signal_config: FullSMBSignalConfig = FullSMBSignalConfig(),
        observation_config: FullSMBObservationConfig = FullSMBObservationConfig(),
        reward_config: FullSMBRewardConfig = DEFAULT_FULL_SMB_REWARD_CONFIG,
        vision: Optional[VisionEncoder] = None,
        env_kwargs: Optional[Mapping[str, Any]] = None,
    ):
        self.env_config = env_config
        self.signal_config = signal_config
        self.observation_config = observation_config
        self.reward_config = reward_config
        self.env = (
            env
            if env is not None
            else make_stable_retro_env(
                env_config,
                content_spec=content_spec,
                **dict(env_kwargs or {}),
            )
        )
        self.backend = GymnasiumBackendAdapter(
            self.env,
            context="Full SMB backend",
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
        return self.backend.buttons

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        result = self.backend.reset(seed=seed)
        observation = self._rgb_observation(result.observation)
        self.last_info = self._annotated_info(
            result.info, terminated=False, truncated=False
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
        previous_info = self.last_info
        backend_reward = 0.0
        frame_rewards: list[float] = []
        observation = None
        info: Mapping[str, Any] = {}
        terminated = False
        truncated = False
        for _ in range(self.observation_config.frame_skip):
            result = self.backend.step(button_action)
            observation = self._rgb_observation(result.observation)
            terminated = result.terminated
            truncated = result.truncated
            info = result.info
            backend_reward += float(result.reward)
            frame_rewards.append(float(result.reward))
            self._append_frame(observation, valid=True)
            if terminated or truncated:
                break
        if observation is None:
            raise RuntimeError("Full SMB frame_skip must execute at least one frame")
        info = self._annotated_info(
            info, terminated=terminated, truncated=truncated
        )
        reward_terms = self._reward_terms(
            backend_reward=backend_reward,
            frame_count=len(frame_rewards),
            previous_info=previous_info,
            current_info=info,
        )
        reward = float(reward_terms["total"])
        info["reward_terms"] = reward_terms
        info["reward_total"] = reward
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
        return observation, reward, terminated, truncated, info

    def save_emulator_state(self) -> FullSMBEmulatorState:
        """Snapshot backend emulator state and adapter metadata."""

        if self._last_observation is None:
            raise RuntimeError("cannot save Full SMB emulator state before reset")
        return FullSMBEmulatorState(
            backend_state=copy.deepcopy(self.backend.get_state()),
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
        self.backend.set_state(copy.deepcopy(state.backend_state))
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
            state=self._encoded_state_vec(info),
            metadata={
                "raw_observation_shape": observation.shape,
                "observation": self._observation_metadata(
                    vision.position.device,
                    info,
                ),
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
        self.backend.close()

    @staticmethod
    def _rgb_observation(observation: Any) -> np.ndarray:
        array = np.asarray(observation)
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(
                "Full SMB observations must have shape [H, W, C] with RGB or "
                "RGBA channels"
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
        tensor = self._crop_observation(tensor)
        if self.observation_config.color_mode == "grayscale":
            weights = torch.tensor(
                [0.299, 0.587, 0.114],
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = (tensor * weights.view(1, 1, 3)).sum(dim=-1, keepdim=True)
            tensor = tensor.repeat(1, 1, 3)
        if self.observation_config.resize_shape is None:
            return self._normalize_observation(tensor).contiguous()
        target_shape = self.observation_config.resize_shape
        if tuple(tensor.shape[:2]) != target_shape:
            chw = tensor.permute(2, 0, 1).unsqueeze(0)
            tensor = (
                F.interpolate(
                    chw,
                    size=target_shape,
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
            )
        return self._normalize_observation(tensor).contiguous()

    def _crop_observation(self, tensor: torch.Tensor) -> torch.Tensor:
        top, right, bottom, left = self.observation_config.effective_crop_margins()
        height, width = tensor.shape[:2]
        if top + bottom >= height or left + right >= width:
            raise ValueError(
                "Full SMB crop margins must leave at least one pixel in both axes"
            )
        y_end = height - bottom if bottom else height
        x_end = width - right if right else width
        return tensor[top:y_end, left:x_end]

    def _normalize_observation(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(
            self.observation_config.normalization_mean,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3)
        std = torch.tensor(
            self.observation_config.normalization_std,
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3)
        return (tensor - mean) / std

    def _observation_metadata(
        self, device: torch.device, info: Mapping[str, Any]
    ) -> dict[str, Any]:
        frame_stack = torch.stack(tuple(self._frame_stack), dim=0).permute(
            0, 3, 1, 2
        )
        camera_vec = np.asarray(info.get("camera_vec", np.zeros(4)), dtype=np.float32)
        return {
            "frame_stack": frame_stack.unsqueeze(0).to(device),
            "frame_mask": torch.tensor(
                tuple(self._frame_mask), dtype=torch.bool, device=device
            ).unsqueeze(0),
            "frame_stack_size": self.observation_config.frame_stack,
            "frame_skip": self.observation_config.frame_skip,
            "resize_shape": self.observation_config.resize_shape,
            "normalized_range": (0.0, 1.0),
            "preprocessing": self.observation_config.to_manifest(),
            "crop_margins": self.observation_config.crop_margins,
            "effective_crop_margins": self.observation_config.effective_crop_margins(),
            "hud_policy": self.observation_config.hud_policy,
            "color_mode": self.observation_config.color_mode,
            "normalization": {
                "input_range": (0.0, 1.0),
                "mean": self.observation_config.normalization_mean,
                "std": self.observation_config.normalization_std,
            },
            "camera_vec": torch.as_tensor(
                camera_vec, dtype=torch.float32, device=device
            ).unsqueeze(0),
            "camera_state_enabled": self.observation_config.include_camera_state,
            "output_channels": 3,
        }

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
        camera_vec = _camera_vec(annotated, signals, self.signal_config)
        annotated["full_smb_signals"] = signals.as_dict()
        annotated["state_vec"] = state_vec
        annotated["camera_vec"] = camera_vec
        annotated["reward_config"] = self.reward_config.to_manifest()
        return annotated

    def _reward_terms(
        self,
        *,
        backend_reward: float,
        frame_count: int,
        previous_info: Mapping[str, Any],
        current_info: Mapping[str, Any],
    ) -> dict[str, float]:
        previous = _signal_mapping(previous_info)
        current = _signal_mapping(current_info)
        config = self.reward_config
        terms = {
            "emulator_progress": float(backend_reward) * config.emulator_progress,
            "completion": _event_started(previous, current, "completion")
            * config.completion,
            "survival": _survival_indicator(current) * config.survival,
            "score": _positive_signal_delta(previous, current, "score")
            * config.score,
            "coin": _positive_signal_delta(previous, current, "coins")
            * config.coin,
            "enemy": _objective_magnitude(current, "enemy") * config.enemy,
            "damage": _objective_magnitude(current, "damage") * config.damage,
            "death": _event_started(previous, current, "death") * config.death,
            "frame_penalty": float(frame_count) * config.frame_penalty,
        }
        terms["total"] = float(sum(terms.values()))
        return {name: float(value) for name, value in terms.items()}

    def _encoded_state_vec(self, info: Mapping[str, Any]) -> Optional[np.ndarray]:
        state = self._state_vec(info)
        if state is None or not self.observation_config.include_camera_state:
            return state
        camera = np.asarray(info.get("camera_vec", np.zeros(4)), dtype=np.float32)
        if camera.ndim != 1:
            raise ValueError("Full SMB info['camera_vec'] must be a 1D vector")
        return np.concatenate((state, np.nan_to_num(camera))).astype(np.float32)

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
    position = _position_value(info)
    screen = _screen_value(info)
    world = _int_value(info, WORLD_KEYS)
    stage = _int_value(info, STAGE_KEYS)
    level = _level_value(info, world=world, stage=stage)
    coins = _int_value(info, COIN_KEYS)
    power_state = _power_state_value(info)
    game_over = _bool_value(info, GAME_OVER_KEYS, default=False) or _reason_matches(
        reason, ("game_over", "game over")
    )
    completion = _bool_value(info, COMPLETION_KEYS, default=False) or _reason_matches(
        reason, ("complete", "completed", "clear", "cleared", "goal", "flag")
    )
    death = (
        game_over
        or _bool_value(info, DEATH_KEYS, default=False)
        or _reason_matches(reason, ("death", "dead", "died", "game_over", "game over"))
    )
    timeout = (
        bool(truncated)
        or _bool_value(info, TIMEOUT_KEYS, default=False)
        or _reason_matches(reason, ("timeout", "time up", "time_up", "out of time"))
    )
    objectives = _objective_values(info)
    return FullSMBSignals(
        position=position,
        progress=position[0] if position is not None else None,
        score=_int_value(info, SCORE_KEYS),
        lives=_int_value(info, LIFE_KEYS),
        collectibles={} if coins is None else {"coins": coins},
        coins=coins,
        screen=screen,
        level=level,
        world=world,
        stage=stage,
        power_state=power_state,
        completion=completion,
        death=death,
        timeout=timeout,
        game_over=game_over,
        objectives=objectives,
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=reason,
    )


def _position_value(info: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    direct = _value_for_keys(info, ("position",))
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
    scroll_x = _raw_scroll_x(info)
    if scroll_x is None:
        return None
    screen_x = _numeric_value(info, SCREEN_X_KEYS)
    return scroll_x + (screen_x or 0.0)


def _raw_scroll_x(info: Mapping[str, Any]) -> Optional[float]:
    scroll_x = _numeric_value(info, SCROLL_X_KEYS)
    if scroll_x is None:
        scroll_lo = _numeric_value(info, SCROLL_X_LO_KEYS)
        scroll_hi = _numeric_value(info, SCROLL_X_HI_KEYS)
        if scroll_lo is not None and scroll_hi is not None:
            scroll_x = scroll_hi * 256.0 + scroll_lo
    return scroll_x


def _camera_vec(
    info: Mapping[str, Any],
    signals: FullSMBSignals,
    config: FullSMBSignalConfig,
) -> np.ndarray:
    scroll_x = _raw_scroll_x(info)
    screen = signals.screen
    screen_x = _numeric_value(info, SCREEN_X_KEYS)
    screen_y = _numeric_value(info, SCREEN_Y_KEYS)
    if screen is not None:
        if screen_x is None:
            screen_x = float(screen[0])
        if screen_y is None:
            screen_y = float(screen[1])
    player_x = signals.position[0] if signals.position is not None else None
    player_camera_x = (
        player_x - scroll_x if player_x is not None and scroll_x is not None else None
    )
    return np.asarray(
        [
            _normalize_feature(scroll_x, config.position_x_max),
            _normalize_feature(screen_x, config.screen_x_max),
            _normalize_feature(screen_y, config.screen_y_max),
            _normalize_feature(player_camera_x, config.screen_x_max),
        ],
        dtype=np.float32,
    )


def _screen_value(info: Mapping[str, Any]) -> Optional[tuple[int, int]]:
    direct = _value_for_keys(info, SCREEN_KEYS)
    if isinstance(direct, Mapping):
        x = _int_value(direct, SCREEN_X_KEYS + POSITION_X_KEYS)
        y = _int_value(direct, SCREEN_Y_KEYS + POSITION_Y_KEYS)
        if x is not None or y is not None:
            return (x or 0, y or 0)
    elif direct is not None:
        try:
            array = np.asarray(direct, dtype=np.float32).flatten()
        except (TypeError, ValueError):
            array = np.asarray([], dtype=np.float32)
        if array.size >= 2 and np.isfinite(array[:2]).all():
            return (int(round(float(array[0]))), int(round(float(array[1]))))
        if array.size == 1 and np.isfinite(array[0]):
            return (int(round(float(array[0]))), 0)

    x = _int_value(info, SCREEN_X_KEYS)
    y = _int_value(info, SCREEN_Y_KEYS)
    if x is not None or y is not None:
        return (x or 0, y or 0)
    return None


def _level_value(
    info: Mapping[str, Any], *, world: Optional[int], stage: Optional[int]
) -> Optional[str]:
    value = _value_for_keys(info, LEVEL_KEYS)
    if value is not None and not isinstance(value, Mapping):
        text = str(value).strip()
        if text:
            return text
    if world is not None and stage is not None:
        return f"{world}-{stage}"
    return None


def _power_state_value(info: Mapping[str, Any]) -> Optional[str]:
    value = _value_for_keys(info, POWER_STATE_KEYS)
    if value is None or isinstance(value, Mapping):
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if not normalized:
            return None
        if normalized in {"tall", "big"}:
            return "big"
        if normalized in {"fireball", "fire"}:
            return "fire"
        return normalized
    try:
        array = np.asarray(value, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return str(value)
    if array.size != 1 or not np.isfinite(array[0]):
        return None
    numeric = int(round(float(array[0])))
    return POWER_STATE_NAMES.get(numeric, str(numeric))


def _int_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[int]:
    value = _numeric_value(info, keys)
    if value is None:
        return None
    return int(round(value))


def _objective_values(info: Mapping[str, Any]) -> dict[str, float]:
    objectives: dict[str, float] = {}
    enemy = _numeric_value(info, ENEMY_OBJECTIVE_KEYS)
    if enemy is not None:
        objectives["enemy"] = max(float(enemy), 0.0)
    damage = _numeric_value(info, DAMAGE_OBJECTIVE_KEYS)
    if damage is not None:
        objectives["damage"] = max(float(damage), 0.0)
    return objectives


def _signal_mapping(info: Mapping[str, Any]) -> Mapping[str, Any]:
    signals = info.get("full_smb_signals") if isinstance(info, Mapping) else None
    if isinstance(signals, Mapping):
        return signals
    return {}


def _event_started(
    previous: Mapping[str, Any], current: Mapping[str, Any], name: str
) -> float:
    return 1.0 if bool(current.get(name)) and not bool(previous.get(name)) else 0.0


def _survival_indicator(current: Mapping[str, Any]) -> float:
    if bool(current.get("death")) or bool(current.get("game_over")):
        return 0.0
    return 1.0


def _positive_signal_delta(
    previous: Mapping[str, Any], current: Mapping[str, Any], name: str
) -> float:
    current_value = _numeric_from_value(current.get(name))
    if current_value is None:
        return 0.0
    previous_value = _numeric_from_value(previous.get(name))
    baseline = previous_value if previous_value is not None else 0.0
    return max(current_value - baseline, 0.0)


def _objective_magnitude(current: Mapping[str, Any], name: str) -> float:
    objectives = current.get("objectives")
    if not isinstance(objectives, Mapping):
        return 0.0
    value = _numeric_from_value(objectives.get(name))
    if value is None:
        return 0.0
    return max(value, 0.0)


def _numeric_from_value(value: Any) -> Optional[float]:
    if value is None or isinstance(value, Mapping):
        return None
    try:
        array = np.asarray(value, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return None
    if array.size == 1 and np.isfinite(array[0]):
        return float(array[0])
    return None


def _numeric_value(info: Mapping[str, Any], keys: tuple[str, ...]) -> Optional[float]:
    for value in _values_for_keys(info, keys):
        numeric = _numeric_from_value(value)
        if numeric is not None:
            return numeric
    return None


def _bool_value(
    info: Mapping[str, Any], keys: tuple[str, ...], *, default: bool
) -> bool:
    for value in _values_for_keys(info, keys):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "1",
                "true",
                "yes",
                "y",
                "complete",
                "clear",
                "flag",
                "dead",
                "game_over",
                "game over",
                "timeout",
                "time_up",
            }:
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
    for value in _values_for_keys(info, keys):
        if value is not None:
            return str(value)
    return None


def _values_for_keys(info: Mapping[str, Any], keys: tuple[str, ...]):
    for mapping in _mapping_candidates(info):
        for key in keys:
            if key in mapping:
                yield mapping[key]


def _value_for_keys(info: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for value in _values_for_keys(info, keys):
        return value
    return None


def _mapping_candidates(info: Mapping[str, Any]):
    yield info
    for key in INFO_CONTAINER_KEYS:
        nested = info.get(key)
        if isinstance(nested, Mapping):
            yield nested


def _reason_matches(reason: Optional[str], needles: tuple[str, ...]) -> bool:
    if reason is None:
        return False
    normalized = reason.strip().lower()
    return any(needle in normalized for needle in needles)


def _normalize_feature(value: Optional[float], scale: float) -> float:
    if value is None:
        return 0.0
    return float(np.clip(float(value) / scale, 0.0, 1.0))
