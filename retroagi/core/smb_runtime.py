"""Checkpoint-owned SMB inference contract (independent of trainer defaults).

The observation is the shared SMB observation (smb_scene) and the physics is
the NES's, in both games; the contract records only control settings.
"""

from dataclasses import asdict, dataclass, fields

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor
from retroagi.core.smb_physics import NES_JUMP_FRAMES


@dataclass(frozen=True)
class SMBRuntimeContract:
    adaptive_duration: bool = False
    walk_primitives: bool = False
    steady_primitives: bool = True
    recurrent_state: bool = False
    critic_feedback: bool = True
    world_model: bool = True
    deterministic_critic: bool = True
    skill_goals: bool = True
    # The policy's strategy network chooses skill goals instead of the scripted
    # local-objective selector.
    learned_skill_goals: bool = False
    engine_support: bool = True
    frame_skip: int = 1
    hold_run_button: bool = True
    wait_duration_scale: float = 4.0
    min_wait_frames: int = 4
    max_wait_frames: int = 64

    def __post_init__(self):
        if self.frame_skip != 1:
            raise ValueError("SMB geometry contract requires one emulator frame per decision")

    @classmethod
    def from_manifest(cls, manifest):
        unknown = set(manifest) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(
                f"Checkpoint predates the shared SMB observation ({sorted(unknown)}); retrain it"
            )
        return cls(**manifest)

    @classmethod
    def from_block_config(cls, config):
        ablation = config.get("ablation", {})
        return cls(
            adaptive_duration=bool(config.get("adaptive_duration_control", True)),
            walk_primitives=bool(config.get("walk_duration_primitives", True)),
            steady_primitives=bool(config.get("steady_duration_primitives", True)),
            recurrent_state=bool(ablation.get("recurrent_state_enabled", True)),
            critic_feedback=bool(ablation.get("critic_feedback_enabled", True)),
            world_model=bool(ablation.get("world_model_enabled", True)),
            deterministic_critic=bool(config.get("deterministic_critic_gates", False)),
            skill_goals=bool(config.get("skill_goal_conditioning", True)),
            learned_skill_goals=bool(config.get("learned_skill_goals", False)),
            engine_support=bool(config.get("engine_support_override", False)),
        )

    def manifest(self):
        return asdict(self)


def attach_runtime(model, manifest):
    if manifest is None:
        return
    contract = SMBRuntimeContract.from_manifest(manifest)
    model.smb_runtime_contract = contract
    if hasattr(model, "ranked_candidate_search"):
        model.ranked_candidate_search = False
    if hasattr(model, "learned_skill_goals"):
        model.learned_skill_goals = contract.learned_skill_goals
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
    executor.engine_support = contract.engine_support
    model.smb_executor = executor
    return executor


class ContractExecutor(SMBParameterizedPrimitiveExecutor):
    jump_hold_frames = NES_JUMP_FRAMES
    engine_support = True

    def _select_hold_frames(self, motor_primitives):
        frames, index = super()._select_hold_frames(motor_primitives)
        if getattr(self, "_reobserve_wait", False):
            return 1, 0
        if getattr(self, "_mapping_jump", False) and index is not None:
            frames = self.jump_hold_frames[index]
        return frames, index

    def prepare(self, batch):
        """Resolve observed primitive boundaries before selecting/labeling A.

        Idempotent for one observation; neither this hook nor execute invokes a
        teacher. Both collectors and playback see the same decision boundary.
        """
        metadata = (batch.metadata or {}).get("smb_geometry", {}) if batch else {}
        if (
            self._active_jump is not None
            and self._released
            and self._left_support
            and metadata.get("support") in ("ground", "platform")
        ):
            self.reset()
        phase = getattr(metadata.get("objective"), "kind", None)
        previous = getattr(self, "_previous_bridge_phase", None)
        # Bridge windows can open while the coarse phase is unchanged or
        # occluded. Reconsider every bridge wait at the next observation.
        self._bridge_wait_context = bool(
            (phase or "").startswith("bridge_")
            or (previous or "").startswith("bridge_")
            or 6 in metadata.get("motion_memory", {})
            or any(p.get("moving") for p in getattr(metadata.get("scene"), "platforms", []))
        )
        if self._active_steady == 0 and self._bridge_wait_context:
            self.reset()
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
        self._reobserve_wait = int(action) == 0 and getattr(self, "_bridge_wait_context", False)
        metadata = (batch.metadata or {}).get("smb_geometry", {}) if batch else {}
        if metadata.get("bouncing") or (
            self._mapping_jump and self._active_jump is None and metadata.get("support") == "air"
        ):
            # A press before physical landing is consumed by NES. Release it
            # and let the policy initiate a fresh press on confirmed support.
            from retroagi.core.actions import SMBPrimitiveExecution, smb_jump_release_action

            return SMBPrimitiveExecution(action=int(smb_jump_release_action(action)))
        if self.engine_support:
            kwargs.setdefault("support_override", metadata.get("support"))
            kwargs.setdefault("enemy_contact_override", metadata.get("enemy_contact", False))
        return super().execute(action, batch=batch, **kwargs)
