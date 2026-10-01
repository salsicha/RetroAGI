"""Training-only physical coaching for the canonical executor.

Oracle state supplies labels, never model observations. Counterfactual probes
restore all state and require collision outcomes, including bounce recovery.
"""

from contextlib import contextmanager
from types import SimpleNamespace

from retroagi.core.smb_physics import NES_JUMP_FRAMES
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import (
    local_objective,
    local_target_distance,
    plant_clearance_target,
)


@contextmanager
def probe_state(env):
    saved = snapshot_env_state(env)
    render = env.__dict__.get("render")
    wait_survival = env._wait_survival
    # Shaping is irrelevant to collision certification; its legacy bridge
    # predictor would otherwise run recursively inside every wait probe.
    env._wait_survival = 0
    env.render = lambda: None
    try:
        yield saved
    finally:
        restore_env_state(env, saved)
        env._wait_survival = wait_survival
        if render is None:
            del env.__dict__["render"]
        else:
            env.render = render


def training_target(env):
    from retroagi.core.smb_objectives import required_stomp

    if getattr(env, "_bridge_jump_task", None):
        from retroagi.stages.block_smb.local_traversal import LocalObjective

        # The jump task succeeds only on the required collision landing.
        return LocalObjective("finish", env.goal.left, env.goal.right, env.goal.bottom)

    if (env._goal_on_stomp or env._require_stomp_before_goal) and not env._stomp_credited:
        target = required_stomp(env)
        if target is not None:
            return target
    return plant_clearance_target(env, local_objective(env))


def probe_executor(model):
    # Do not replace the live collector's executor reference on the model.
    from retroagi.core.smb_runtime import make_smb_executor

    previous = getattr(model, "smb_executor", None)
    executor = make_smb_executor(model)
    model.smb_executor = previous
    return executor


def physical_batch(env, *, bouncing=False):
    return SimpleNamespace(
        metadata={
            "smb_geometry": {
                "support": "ground" if env.mario["on_ground"] else "air",
                "enemy_contact": False,
                "bouncing": bouncing,
            }
        }
    )


def primitive(index):
    """A motor output that selects one NES hold with certainty."""
    import torch

    logits = torch.full((1, 1, 16), -20.0)
    logits[..., index] = 20.0
    return SimpleNamespace(
        hold_duration_logits=logits, duration_bin_values=torch.tensor(NES_JUMP_FRAMES)
    )


def teacher_runtime():
    """Fixed-duration execution for teachers that replay one primitive at a time."""
    from retroagi.core.smb_runtime import SMBRuntimeContract

    return SMBRuntimeContract(
        recurrent_state=False,
        adaptive_duration=False,
        walk_primitives=False,
        steady_primitives=True,
        critic_feedback=False,
        wait_duration_scale=1.0,
        min_wait_frames=1,
        max_wait_frames=32,
    )


def safe_jump_indices(model, env, action, *, verify_recovery=True):
    """Certify each physical hold using the same executor as greedy playback."""
    if not env.mario["on_ground"] or action not in (2, 4, 5):
        return []
    target = training_target(env)
    valid = []
    with probe_state(env) as saved:
        for index in range(16):
            restore_env_state(env, saved)
            executor = probe_executor(model)
            airborne = bouncing = False
            for _ in range(96):
                execution = executor.execute(
                    action,
                    batch=physical_batch(env, bouncing=bouncing),
                    motor_primitives=primitive(index),
                )
                _, _, done, truncated, info = env.step(execution.action)
                airborne |= not env.mario["on_ground"]
                bouncing |= info["reward_terms"]["enemy_stomp"] > 0
                landed = airborne and env.mario["on_ground"]
                if info["death"]:
                    break
                reached = env._goal_credited if env._single_jump_attempt else target.reached(env)
                if reached and (landed or env._goal_credited):
                    recoverable = True
                    if landed and not env._goal_credited and verify_recovery:
                        following = training_target(env)
                        if (
                            following.kind in ("enemy", "stomp")
                            and local_target_distance(env, following) < 50
                        ):
                            recoverable = bool(
                                safe_jump_indices(
                                    model,
                                    env,
                                    2 if following.direction > 0 else 4,
                                    verify_recovery=False,
                                )
                            )
                    if recoverable:
                        valid.append(index)
                    break
                if done or truncated or landed:
                    break
    return valid
