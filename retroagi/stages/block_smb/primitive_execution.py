"""Physical duration mapping shared by Block training and evaluation."""

from copy import copy
from dataclasses import is_dataclass, replace

import torch

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor, SMBPrimitiveExecution
from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE


class BlockSMBPrimitiveExecutor(SMBParameterizedPrimitiveExecutor):
    def __init__(self, env, **kwargs):
        self.jump_frames = (
            NES_JUMP_FRAMES if env.physics_profile == NES_PHYSICS_PROFILE else tuple(range(1, 17))
        )
        self.reobserve_bridge_wait = bool(env._bridge_jump_task)
        super().__init__(max_hold_frames=max(self.jump_frames), **kwargs)

    def motor_parameters(self, action, motor):
        """Map jump bins without changing a shared model's other scenario rows."""
        if motor is None:
            return None
        values = self.jump_frames if int(action) in (2, 4, 5) else tuple(range(1, 17))
        logits = motor.hold_duration_logits
        values = torch.tensor(values, device=logits.device if logits is not None else None)
        if is_dataclass(motor):
            return replace(motor, duration_bin_values=values)
        result = copy(motor)
        result.duration_bin_values = values
        return result

    def execute(self, action, **kwargs):
        # The NES jump menu must not enlarge adaptive walking commitments.
        jumping = self._active_jump is not None or int(action) in (2, 4, 5)
        self.max_hold_frames = max(self.jump_frames) if jumping else 16
        return super().execute(action, **kwargs)

    def _start_steady_primitive(self, action_value, motor_primitives):
        if self.reobserve_bridge_wait and int(action_value) == 0:
            # The moving target can enter and leave jump range before a long
            # timer or track-end cue fires. Match canonical bridge playback.
            self.reset()
            return SMBPrimitiveExecution(
                action=0, started=True, active=True, released=True, hold_frames=1
            )
        return super()._start_steady_primitive(action_value, motor_primitives)
