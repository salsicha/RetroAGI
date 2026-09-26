"""Physical duration mapping shared by Block training and evaluation."""

from copy import copy
from dataclasses import dataclass, is_dataclass, replace

import torch

from retroagi.core.actions import SMBParameterizedPrimitiveExecutor, SMBPrimitiveExecution
from retroagi.core.smb_physics import NES_JUMP_FRAMES, NES_PHYSICS_PROFILE


class BlockSMBPrimitiveExecutor(SMBParameterizedPrimitiveExecutor):
    def __init__(self, env, **kwargs):
        self.jump_frames = (
            NES_JUMP_FRAMES if env.physics_profile == NES_PHYSICS_PROFILE else tuple(range(1, 17))
        )
        # Every plant layout re-decides waits each frame. Keying this on the
        # private timed_crossing flag gave observationally identical clearance
        # layouts 4-64 frame committed waits that could not react to retraction.
        self.reobserve_bridge_wait = bool(env._bridge_jump_task) or any(
            e.get("kind") == "piranha_plant" for e in env.enemies
        )
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


@dataclass
class JumpReleaseState:
    """Track executor-owned landing frames while replaying executed actions.

    A normal jump releases on landing and suppresses a new jump for one more
    frame. Stomps reset the executor, and ordinary falls own no release frames.
    Teachers and repair splices must preserve this state across their prefix.
    """

    remaining: int = 0
    action: int = 1
    jumping: bool = False
    airborne: bool = False
    bouncing: bool = False

    def observe(self, env, action, info):
        if self.remaining:
            self.remaining -= 1
        if info["reward_terms"]["enemy_stomp"] > 0:
            self.jumping = self.airborne = False
            self.remaining = 0
            self.bouncing = True
        if self.bouncing:
            if env.mario["on_ground"]:
                self.bouncing = False
            return
        if not self.jumping and action in (2, 4, 5):
            self.jumping = True
            self.action = {2: 1, 4: 3, 5: 0}[action]
        if self.jumping:
            self.airborne |= not env.mario["on_ground"]
            if self.airborne and env.mario["on_ground"]:
                self.remaining = 2
                self.jumping = self.airborne = False


def teacher_route_reachable(env, actions, *, release_state=None):
    """Validate a raw teacher suffix through fixed jump execution, then restore.

    Walking/waiting are already expanded into physics frames. A splice may
    start inside a landing release; consume its inherited frames first.
    """
    from types import SimpleNamespace

    from .geometry_expert import restore_env_state, snapshot_env_state

    saved = snapshot_env_state(env)
    release = replace(release_state) if release_state is not None else JumpReleaseState()
    executor = BlockSMBPrimitiveExecutor(
        env, duration_sampling=False, adaptive_duration=False, steady_primitives=False
    )
    bouncing = release.bouncing
    try:
        for frame, requested in enumerate(actions):
            if release.remaining:
                action = release.action
                release.remaining -= 1
            elif bouncing:
                action = {2: 1, 4: 3, 5: 0}.get(requested, requested)
            else:
                chosen = executor.committed_action
                chosen = requested if chosen is None else chosen
                motor = None
                if not executor.active and chosen in (2, 4, 5):
                    held = 0
                    for future in actions[frame:]:
                        if future != requested:
                            break
                        held += 1
                    menu = executor.jump_frames
                    slot = min(range(len(menu)), key=lambda i: abs(menu[i] - held))
                    logits = torch.full((1, 16), -30.0)
                    logits[0, slot] = 30.0
                    motor = SimpleNamespace(
                        hold_duration_logits=logits, duration_bin_values=torch.tensor(menu)
                    )
                action = executor.execute(
                    chosen,
                    motor_primitives=motor,
                    support_override="ground" if env.mario["on_ground"] else "air",
                ).action
            if action != requested:
                return False
            _, _, done, truncated, info = env.step(action)
            if info["reward_terms"]["enemy_stomp"] > 0:
                executor.reset()
                bouncing = True
            elif env.mario["on_ground"]:
                bouncing = False
            if done or truncated:
                break
        return bool(env._goal_credited)
    finally:
        restore_env_state(env, saved)
