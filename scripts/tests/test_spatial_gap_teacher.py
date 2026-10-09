"""Spatial gap lessons survive braking, small target errors and stalled requests."""

from dataclasses import replace

import pytest

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.tokens import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.layered_train import learner_families
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.skill_families import UPPER_GAP_SKILL_FAMILIES
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_plan, teacher_skill


def scenario(family, difficulty, index):
    return sample_block_smb_monte_carlo_scenario(
        family=family, split="validation", seed=0, sample_index=index, difficulty=difficulty
    ).scenario


@pytest.mark.parametrize("family", UPPER_GAP_SKILL_FAMILIES)
@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_upper_route_is_tactics_and_each_local_crossing_has_one_goal(family, difficulty):
    skills = learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    tactics = learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    assert family in skills and family not in tactics
    assert "low_choice_advance" in tactics and "low_choice_advance" not in skills
    s = scenario(family, difficulty, 100)
    assert len(s["tactics"]) == 1 and len(s["tactics"][0]["route"]) == 1
    assert s["tactics"][0]["forbidden"] == [3, 4]
    target = s["platforms"][s["tactics"][0]["route"][0]]
    assert s["goal"] == [target[0], target[1] - 20, target[2], 20]


def play_teacher(family, difficulty, index, *, error=0, stall_frames=0):
    s = scenario(family, difficulty, index)
    env = MarioScenarioEnv()
    try:
        screen, _ = env.reset(scenario=s)
        teacher = episode_teacher(s)

        class Observer:
            def observe(self, screens):
                return [scene_from_labels(env.scene_labels())]

        agent = SMBAgents(Observer(), LayeredSMBPolicy().eval(), "cpu")
        labels, stalled = [], False

        def given(copies, scenes):
            if env.steps < stall_frames:
                return {"skill": [SkillToken("run", 0, 0)]}
            plan, _ = teacher_plan(env, teacher, certify_holds=False)
            assert plan is not None
            goal = teacher_skill(env, teacher, plan)
            labels.append(goal)
            assert goal.mode != "run" or abs(goal.x) >= 4
            return {"skill": [replace(goal, x=goal.x + error) if goal.mode == "jump" else goal]}

        for _ in range(360):
            teacher.observe_frame(env)
            step = agent.act([screen], [0], given=given)[0]
            if step.decision and env.steps >= stall_frames:
                stalled |= bool(step.decision.execution_feedback[1])
            screen, _, done, truncated, _ = env.step(step.button)
            if done or truncated:
                break
        assert env._goal_credited
        assert len(labels) <= 8
        if stall_frames:
            assert stalled
    finally:
        env.close()


@pytest.mark.parametrize("error", [-2, 0, 2])
@pytest.mark.parametrize(
    "family,difficulty,index",
    [
        ("choice_advance", "easy", 100),
        ("choice_advance", "hard", 108),
        ("skill_upper_gap_entry", "hard", 108),
        ("skill_upper_gap_exit", "hard", 108),
        ("skill_gap_descent", "hard", 106),
    ],
)
def test_spatial_waypoints_land_after_braking_with_small_target_errors(
    family, difficulty, index, error
):
    play_teacher(family, difficulty, index, error=error)


def test_stalled_zero_distance_request_gets_a_progressing_training_target():
    play_teacher("choice_advance", "hard", 108, stall_frames=35)
