"""Checkpoint-owned SMB inference contract (independent of trainer defaults)."""

from dataclasses import asdict, dataclass

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor
from retroagi.core.smb_geometry import SCHEMA


@dataclass(frozen=True)
class SMBRuntimeContract:
    objective_contract: str = "legacy"
    schema: str = SCHEMA
    motion_observations: bool = True
    adaptive_duration: bool = False
    walk_primitives: bool = False
    steady_primitives: bool = True
    recurrent_state: bool = False
    critic_feedback: bool = True
    world_model: bool = True
    deterministic_critic: bool = True
    skill_goals: bool = True
    engine_support: bool = True
    frame_skip: int = 1
    hold_run_button: bool = True
    visual_tokens: str = "native_unaligned"
    geometry_source: str = "nes_collision_ram"
    observation_provider: str = "oracle"
    jump_hold_frames: tuple[int, ...] = tuple(range(1, 17))
    physics_profile: str = "unadapted"
    wait_duration_scale: float = 4.0
    min_wait_frames: int = 4
    max_wait_frames: int = 64

    def __post_init__(self):
        object.__setattr__(self, "jump_hold_frames", tuple(self.jump_hold_frames))
        if len(self.jump_hold_frames) != 16 or any(
            not isinstance(n, int) or n < 1 or n > 60 for n in self.jump_hold_frames
        ):
            raise ValueError("A physical jump hold is required for each of the 16 duration bins")
        if sorted(self.jump_hold_frames) != list(self.jump_hold_frames):
            raise ValueError("Jump-duration mapping must be monotone")
        if self.adaptive_duration and self.jump_hold_frames != tuple(range(1, 17)):
            raise ValueError("Calibrated durations require fixed commitments")
        if self.schema not in (SCHEMA, "smb_scene_v2"):
            raise ValueError(f"Unsupported SMB observation schema: {self.schema}")
        if self.frame_skip != 1:
            raise ValueError("SMB geometry contract requires one emulator frame per decision")
        if self.visual_tokens not in (
            "native_unaligned",
            "native_adapted",
            "zero_ablation",
            "canonical",
        ):
            raise ValueError("Unsupported visual feature adapter")
        if self.observation_provider not in ("oracle", "perceived"):
            raise ValueError("Unsupported observation provider")
        if self.observation_provider == "perceived" and self.schema != "smb_scene_v2":
            raise ValueError("Perceived geometry requires canonical scene interfaces")
        expected_source = (
            ("canonical_pixels" if self.observation_provider == "perceived" else "canonical_oracle")
            if self.schema == "smb_scene_v2"
            else "nes_collision_ram"
        )
        if self.geometry_source != expected_source:
            raise ValueError("Geometry provider does not match observation contract")

    @classmethod
    def from_block_config(cls, config):
        ablation = config.get("ablation", {})
        return cls(
            motion_observations=bool(config.get("motion_observations", False)),
            adaptive_duration=bool(config.get("adaptive_duration_control", True)),
            walk_primitives=bool(config.get("walk_duration_primitives", True)),
            steady_primitives=bool(config.get("steady_duration_primitives", True)),
            recurrent_state=bool(ablation.get("recurrent_state_enabled", True)),
            critic_feedback=bool(ablation.get("critic_feedback_enabled", True)),
            world_model=bool(ablation.get("world_model_enabled", True)),
            deterministic_critic=bool(config.get("deterministic_critic_gates", False)),
            skill_goals=bool(config.get("skill_goal_conditioning", True)),
            engine_support=bool(config.get("engine_support_override", False)),
        )

    def manifest(self):
        return asdict(self)


def attach_runtime(model, manifest):
    if manifest is None:
        return
    contract = SMBRuntimeContract(**manifest)
    model.smb_runtime_contract = contract
    if contract.schema == "smb_scene_v2" and hasattr(model, "motor_controller"):
        import torch

        model.motor_controller.duration_bin_values.copy_(
            torch.tensor(
                contract.jump_hold_frames, device=model.motor_controller.duration_bin_values.device
            )
        )
    if hasattr(model, "ranked_candidate_search"):
        model.ranked_candidate_search = False
    if hasattr(model, "deterministic_critic_slots"):
        from retroagi.stages.block_smb.adapter import block_smb_deterministic_critic_slots

        model.deterministic_critic_slots = (
            block_smb_deterministic_critic_slots() if contract.deterministic_critic else None
        )


def make_smb_executor(model, *, deterministic=True, seed=None):
    contract = getattr(model, "smb_runtime_contract", None)
    if contract is None:
        return SMBParameterizedPrimitiveExecutor()
    executor = ContractExecutor(
        adaptive_duration=contract.adaptive_duration,
        steady_primitives=contract.steady_primitives,
        walk_primitives=contract.walk_primitives,
        duration_sampling=not deterministic,
        duration_seed=seed,
        wait_duration_scale=contract.wait_duration_scale,
        min_wait_frames=contract.min_wait_frames,
        max_wait_frames=contract.max_wait_frames,
    )
    executor.jump_hold_frames = contract.jump_hold_frames
    executor.engine_support = contract.engine_support
    executor.nes_press_edges = contract.schema == "smb_scene_v2"
    model.smb_executor = executor
    return executor


class ContractExecutor(SMBParameterizedPrimitiveExecutor):
    jump_hold_frames = tuple(range(1, 17))
    engine_support = True

    def _select_hold_frames(self, motor_primitives):
        frames, index = super()._select_hold_frames(motor_primitives)
        if getattr(self, "_mapping_jump", False) and index is not None:
            frames = self.jump_hold_frames[index]
        return frames, index

    def prepare(self, batch):
        """Resolve observed primitive boundaries before selecting/labeling A.

        Idempotent for one observation; neither this hook nor execute invokes a
        teacher. Both collectors and playback see the same decision boundary.
        """
        metadata = (batch.metadata or {}).get("smb_geometry", {}) if batch else {}
        if getattr(self, "nes_press_edges", False):
            if (
                self._active_jump is not None
                and self._released
                and self._left_support
                and metadata.get("support") in ("ground", "platform")
            ):
                self.reset()
            phase = getattr(metadata.get("objective"), "kind", None)
            previous = getattr(self, "_previous_bridge_phase", None)
            if self._active_steady == 0 and (previous, phase) in (
                ("bridge_wait", "bridge_board"),
                ("bridge_ride", "bridge_exit"),
            ):
                self.reset()
            self._previous_bridge_phase = phase
        if metadata.get("bouncing"):
            self.reset()
        return self.committed_action

    def execute(self, action, *, batch=None, **kwargs):
        self.prepare(batch)
        self._mapping_jump = int(action) in (2, 4, 5)
        metadata = (batch.metadata or {}).get("smb_geometry", {}) if batch else {}
        if metadata.get("bouncing"):
            from retroagi.core.actions import SMBPrimitiveExecution, smb_jump_release_action

            return SMBPrimitiveExecution(action=int(smb_jump_release_action(action)))
        if self.engine_support:
            kwargs.setdefault("support_override", metadata.get("support"))
            kwargs.setdefault("enemy_contact_override", metadata.get("enemy_contact", False))
        return super().execute(action, batch=batch, **kwargs)
