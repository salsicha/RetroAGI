"""Checkpoint-owned SMB inference contract (independent of trainer defaults)."""

from dataclasses import asdict, dataclass

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor
from retroagi.core.smb_geometry import SCHEMA


@dataclass(frozen=True)
class SMBRuntimeContract:
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
    jump_hold_frames: tuple[int, ...] = tuple(range(1, 17))
    physics_profile: str = "unadapted"

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
        if self.schema != SCHEMA:
            raise ValueError(f"Unsupported SMB observation schema: {self.schema}")
        if self.frame_skip != 1:
            raise ValueError("SMB geometry contract requires one emulator frame per decision")
        if self.visual_tokens not in ("native_unaligned", "native_adapted", "zero_ablation"):
            raise ValueError("Unsupported visual feature adapter")
        if self.geometry_source != "nes_collision_ram":
            raise ValueError("Unsupported geometry provider")

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
    )
    executor.jump_hold_frames = contract.jump_hold_frames
    executor.engine_support = contract.engine_support
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

    def execute(self, action, *, batch=None, **kwargs):
        self._mapping_jump = int(action) in (2, 4, 5)
        metadata = (batch.metadata or {}).get("smb_geometry", {}) if batch else {}
        if metadata.get("bouncing"):
            from retroagi.core.actions import SMBPrimitiveExecution, smb_jump_release_action

            self.reset()
            return SMBPrimitiveExecution(action=int(smb_jump_release_action(action)))
        if self.engine_support:
            kwargs.setdefault("support_override", metadata.get("support"))
            kwargs.setdefault("enemy_contact_override", metadata.get("enemy_contact", False))
        return super().execute(action, batch=batch, **kwargs)
