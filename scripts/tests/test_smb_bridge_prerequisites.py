"""Moving-bridge prerequisite tasks, progress credit, and replay allocation."""

import pytest

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state


def touching_bridge(task):
    return dict(
        world_width=380,
        mario=[74 if task == "mount" else 175, 208],
        platforms=[
            [0, 220, 85, 20],
            dict(x=75, y=220, w=110, h=20, moving=[75, 145, 1], direction=1),
            [180, 220, 200, 20],
        ],
        goal=[240, 200, 16, 20],
        require_bridge_before_goal=True,
        bridge_jump_task=task,
        reward_wait_survival=0,
        reward_goal_distance_shaping=2,
    )


@pytest.mark.parametrize("task", ["mount", "dismount"])
def test_walking_does_not_satisfy_jump_task(task):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge(task))
        for _ in range(120):
            _, _, done, truncated, _ = env.step(1)
            if done or truncated:
                break
        assert env._bridge_boarded
        assert not env._goal_credited
    finally:
        env.close()


def test_passive_carry_receives_existing_progress_without_right_and_no_double_payment():
    env = MarioScenarioEnv()
    scenario = touching_bridge("dismount")
    scenario["mario"] = [110, 208]
    try:
        env.reset(scenario=scenario)
        before = env.mario["x"]
        _, _, _, _, info = env.step(0)
        assert env.mario["x"] > before and env.mario["vx"] == 0
        assert info["reward_terms"]["progress"] > 0 and info["reward_terms"]["goal_distance"] > 0
        assert info["bridge_carry_progress"] == pytest.approx(info["reward_terms"]["progress"])
        env._max_x_reached = 300
        _, _, _, _, info = env.step(0)
        assert info["bridge_carry_progress"] == 0
        state = snapshot_env_state(env)
        env.step(2)
        assert env._bridge_jump_launched
        restore_env_state(env, state)
        assert not env._bridge_jump_launched
    finally:
        env.close()


@pytest.mark.parametrize("task", ["mount", "dismount"])
def test_jump_must_start_on_the_correct_surface(task):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge(task))
        support = env.platforms[1 if task == "mount" else 0]
        env.mario.update(
            x=110.0 if task == "mount" else 60.0, y=208.0, on_ground=True, _platform=support
        )
        env.step(2)
        assert not env.mario["on_ground"]
        assert not env._bridge_jump_launched
        assert not env._goal_credited
    finally:
        env.close()


def test_mount_credits_actual_edge_landing_without_full_body_containment():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=touching_bridge("mount"))
        bridge = env.platforms[1]
        bridge["rect"].x = 100
        bridge["move_x"] = 100.0
        env.mario.update(x=94.0, y=207.0, on_ground=False, _platform=None)
        env.motion.y_speed = 1
        env.motion.y_force = 0
        env._bridge_jump_launched = True
        env._airborne_started_with_jump = True
        _, _, _, _, info = env.step(0)
        assert env.mario["on_ground"] and env.mario["_platform"] is bridge
        assert env.mario["x"] < bridge["rect"].left
        assert env._goal_credited and not info["death"]
    finally:
        env.close()
