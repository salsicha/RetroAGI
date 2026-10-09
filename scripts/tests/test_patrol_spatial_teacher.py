"""Patrol demonstrations must work as spatial commands, not just button routes."""

from copy import deepcopy
from dataclasses import replace

import pytest

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.patrol_teacher import destination, rollout
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_skill


def patrol(difficulty="easy", index=100):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family="enemy_patrol", difficulty=difficulty
    ).scenario


@pytest.mark.parametrize(
    "difficulty,index",
    [(d, 100 + i) for i, d in enumerate(["easy"] * 3 + ["medium"] * 3 + ["hard"] * 3)],
)
def test_patrol_teacher_completes_reviewed_layout_through_unchanged_executor(difficulty, index):
    env = MarioScenarioEnv()
    try:
        scenario = patrol(difficulty, index)
        screen, _ = env.reset(scenario=scenario)
        teacher = episode_teacher(scenario)

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        agent = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")
        goals = []

        def given(copies, scenes):
            teacher.execution = agent.copies[0].spatial
            teacher.scene = scenes[0]
            goal = teacher_skill(env, teacher, None)
            assert goal is not None, (env.steps, env.mario)
            assert goal.x >= 4
            center = env.mario["x"] + env.mario["w"] / 2
            assert center + goal.x <= env.platforms[0]["rect"].right - env.mario["w"] / 2
            if goal.mode == "run":
                assert goal.y == round(220 - env.mario["y"] - env.mario["h"])
            goals.append(goal)
            return {"skill": [goal]}

        for _ in range(env.frame_budget):
            teacher.observe_frame(env)
            step = agent.act([screen], [0], given=given)[0]
            screen, _, done, truncated, _ = env.step(step.button)
            if done or truncated:
                break
        assert env._goal_credited
        assert len(goals) <= 12
    finally:
        env.close()


def test_candidate_probes_restore_simulator_and_execution_feedback():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=patrol())
        spatial = SpatialFeedback()
        spatial.observe(scene_from_labels(env.scene_labels()))
        before = snapshot_env_state(env)
        previous = deepcopy(spatial)
        result = rollout(env, SkillToken("run", 180, 0), spatial)
        assert not result.safe  # An approach that runs into the patrol is never a label.
        assert snapshot_env_state(env) == before
        assert spatial == previous
    finally:
        env.close()


def test_early_visual_contact_gets_a_floor_target_to_finish_landing():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=patrol())
        for enemy in env.enemies:
            enemy["dead"] = True
        env.mario.update(x=157.0, y=202, vx=0.375, vy=4.125, on_ground=False, _platform=None)
        env.motion.x_speed, env.motion.y_speed, env.motion.y_force = 6, 4, 32
        scene = scene_from_labels(env.scene_labels())
        scene = replace(scene, mario=replace(scene.mario, on_something=True))
        spatial = SpatialFeedback(motion=deepcopy(env.motion))
        spatial.observe(scene)
        goal = destination(env, spatial, scene)
        assert goal == SkillToken("run", 4, 6)
        result = rollout(env, goal, spatial, scene)
        assert result.safe and result.progress < 4
    finally:
        env.close()


def test_finish_target_can_correct_a_small_overshoot():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=patrol())
        env.mario["x"] = 246.0
        goal = destination(env)
        assert goal is not None and goal.mode == "run" and goal.x < 0
        assert rollout(env, goal).won
    finally:
        env.close()
