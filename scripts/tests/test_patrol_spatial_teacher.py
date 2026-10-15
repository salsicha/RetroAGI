"""Patrol demonstrations must work as spatial commands, not just button routes
(the main teacher, with no route proposal: teacher_tokens.teacher_skill)."""

import pytest

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_skill


def patrol(difficulty="easy", index=100):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family="enemy_patrol", difficulty=difficulty
    ).scenario


@pytest.mark.timeout(900)
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
