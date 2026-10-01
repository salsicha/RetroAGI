"""Adapter from the block-SMB pygame environment to the shared training contract."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from retroagi.core import (
    SMB_GAME_SPEC,
    SMBAction,
    StageBatch,
    StageSpec,
    VisionEncoder,
    block_smb_action,
)
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.vision import BlockVisionTransformer

BLOCK_SMB_SPEC = StageSpec(
    name="block_smb",
    observation_kind="low-resolution pygame RGB plus observed screen geometry",
    action_kind="shared SMBAction vocabulary",
    seq_len_a=8,
    ratio_ab=2,
    ratio_bc=4,
    vocab_size=20,
    action_space_name=SMB_GAME_SPEC.name,
    action_count=SMB_GAME_SPEC.action_count,
    action_names=tuple(action.name for action in SMB_GAME_SPEC.action_space),
)

SCENARIOS_DIR = Path(__file__).with_name("scenarios")


@dataclass(frozen=True)
class BlockSMBObservationConfig:
    """Preprocessing contract for Block SMB policy observations."""

    frame_stack: int = 4

    def __post_init__(self) -> None:
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")


class BlockSMBStage:
    """Stage adapter for scriptable pygame scenarios."""

    spec = BLOCK_SMB_SPEC

    def __init__(
        self,
        env: Optional[MarioScenarioEnv] = None,
        scenario: Optional[dict] = None,
        vision: Optional[VisionEncoder] = None,
        observation_config: BlockSMBObservationConfig = BlockSMBObservationConfig(),
    ):
        from retroagi.core.smb_enemy_history import EnemyObservationHistory
        from retroagi.core.smb_scene import ObjectiveMemory, SMBProjector

        self.env = env or MarioScenarioEnv()
        self.scenario = scenario
        self.vision = vision or BlockVisionTransformer()
        self.observation_config = observation_config
        # The finish marker is simulator truth, not part of the visible game.
        self.env.render_goal = False
        if isinstance(self.vision, torch.nn.Module):
            self.vision.eval()
        self.vision_projector = SMBProjector(self.spec)
        self.last_info: Mapping[str, Any] = {}
        self._frame_stack: deque[torch.Tensor] = deque(maxlen=self.observation_config.frame_stack)
        self._frame_mask: deque[bool] = deque(maxlen=self.observation_config.frame_stack)
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        self._cached_vision_frame = None
        self._cached_vision = None
        self.objective_memory = ObjectiveMemory()
        self.enemy_history = EnemyObservationHistory()
        self._hazard_features = self.enemy_history.cached.copy()
        # Peak exposure is a world-model memory target, never a policy input.
        self._hazard_memory = self.enemy_history.memory_features()

    def reset(self, seed: Optional[int] = None):
        obs, info = self.env.reset(scenario=self.scenario, seed=seed)
        self.last_info = info
        self.enemy_history.reset()
        self._hazard_features = self.enemy_history.observe(self.env, self.env.steps)
        self._hazard_memory = self.enemy_history.memory_features()
        self._last_episode_mask = 1.0
        self._last_terminal = False
        self._last_truncated = False
        self._cached_vision_frame = None
        self._cached_vision = None
        self.objective_memory.reset()
        self._reset_frame_stack(obs)
        return obs

    def step(self, action: SMBAction | int):
        obs, reward, terminated, truncated, info = self.env.step(block_smb_action(action))
        self.last_info = info
        self._hazard_features = self.enemy_history.observe(self.env, self.env.steps)
        self._hazard_memory = self.enemy_history.memory_features()
        self._last_episode_mask = 0.0 if terminated or truncated else 1.0
        self._last_terminal = terminated
        self._last_truncated = truncated
        self._append_frame(obs, valid=True)
        return obs, reward, terminated, truncated, info

    def encode_observation(
        self, observation: np.ndarray, info: Optional[Mapping[str, Any]] = None
    ) -> StageBatch:
        """Convert block-SMB vision and observed geometry into the hierarchy."""
        from retroagi.core.smb_scene import canonical_vision

        info = info or self.last_info
        if not self._frame_stack:
            self._reset_frame_stack(observation)
        normalized_observation = self._normalize_observation(observation)
        if not torch.equal(self._frame_stack[-1], normalized_observation):
            self._append_frame(observation, valid=True)
        # A transition's next frame is the next decision's current frame.
        # The frozen encoder need only process those identical pixels once;
        # geometry and episode metadata are still projected on every call.
        if self._cached_vision_frame is None or not torch.equal(
            normalized_observation, self._cached_vision_frame
        ):
            with torch.no_grad():
                self._cached_vision = self.vision.encode(normalized_observation)
            self._cached_vision_frame = normalized_observation.clone()
        vision = canonical_vision(self._cached_vision, "block")
        geometry = self.geometry(info)
        return self.vision_projector.project(
            vision,
            self._observed(geometry),
            metadata={
                "smb_geometry": geometry,
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

    def geometry(self, info=None):
        """The current frame's geometry record, with the visible local objective."""
        from retroagi.core.smb_scene import (
            apply_local_target,
            block_oracle_scene,
            preserve_objective,
        )

        info = info or self.last_info
        geometry = block_oracle_scene(
            self.env,
            terminated=self._last_terminal,
            truncated=self._last_truncated,
            objective_kind=(self.scenario or {}).get("task_objective"),
        )
        if self.env.mario["on_ground"]:
            self.objective_memory.bouncing = False
        elif info.get("reward_terms", {}).get("enemy_stomp", 0) > 0:
            self.objective_memory.bouncing = True
        geometry["bouncing"] = self.objective_memory.bouncing
        geometry["enemy_history"] = self._hazard_features
        return apply_local_target(preserve_objective(geometry, self.objective_memory))

    def state_features(self, info=None):
        """The geometry observer's C-stream slots for the env's current frame."""
        return self._observed(self.geometry(info))

    @staticmethod
    def _observed(geometry):
        from retroagi.core.smb_scene import observed_features

        return observed_features(geometry, geometry["enemy_history"])

    def _reset_frame_stack(self, observation: np.ndarray) -> None:
        self._frame_stack.clear()
        self._frame_mask.clear()
        normalized = self._normalize_observation(observation)
        padding = self.observation_config.frame_stack - 1
        for _ in range(padding):
            self._frame_stack.append(normalized.clone())
            self._frame_mask.append(False)
        self._frame_stack.append(normalized)
        self._frame_mask.append(True)

    def _append_frame(self, observation: np.ndarray, *, valid: bool) -> None:
        self._frame_stack.append(self._normalize_observation(observation))
        self._frame_mask.append(valid)

    @staticmethod
    def _normalize_observation(observation: np.ndarray | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(observation, dtype=torch.float32)
        if tensor.ndim != 3 or tensor.shape[-1] not in (3, 4):
            raise ValueError(
                "Block SMB observations must have shape [H, W, C] with RGB or RGBA channels"
            )
        tensor = tensor[..., :3]
        if bool(tensor.numel()) and float(tensor.max()) > 1.0:
            tensor = tensor / 255.0
        return tensor.clamp(0.0, 1.0)

    def _observation_metadata(self, device: torch.device) -> dict[str, Any]:
        frame_stack = torch.stack(tuple(self._frame_stack), dim=0).permute(0, 3, 1, 2)
        return {
            "frame_stack": frame_stack.unsqueeze(0).to(device),
            "frame_mask": torch.tensor(
                tuple(self._frame_mask), dtype=torch.bool, device=device
            ).unsqueeze(0),
            "frame_stack_size": self.observation_config.frame_stack,
            "normalized_range": (0.0, 1.0),
            "state_range": (-1.0, 1.0),
        }


def block_smb_deterministic_critic_slots() -> dict[str, float]:
    """Absolute C-stream indices for deterministic critic gates.

    Progress is the mechanistic decrease of the predicted distance to the local
    objective; death is read directly from the LSTM world model's predicted
    death flag (trained by the terminal_outcome dynamics slot). The terminated
    flag is deliberately NOT used for the death gate because it also fires on
    goal completion.
    """
    from retroagi.core.smb_scene import c_feature_index

    return {
        "goal_distance": c_feature_index("goal_distance"),
        "position_x": c_feature_index("x"),
        "position_y": c_feature_index("y"),
        "death": c_feature_index("death"),
        "progress_epsilon": 0.002,
        "death_threshold": 0.5,
    }
