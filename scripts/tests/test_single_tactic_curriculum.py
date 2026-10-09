"""Curriculum ownership follows actual teacher tactics, not family names."""

import pytest

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.layered_train import learner_families
from retroagi.stages.block_smb.monte_carlo import (
    BLOCK_SMB_MC_FAMILIES,
    sample_block_smb_monte_carlo_scenario,
)
from retroagi.stages.block_smb.tactic_families import MULTI_TACTIC_FAMILIES
from retroagi.stages.block_smb.teacher_tokens import episode_teacher, teacher_tactic


def played_tactics(family, difficulty, index):
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family=family, difficulty=difficulty
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        env.render = lambda: None
        teacher = episode_teacher(sample.scenario)
        tactics = set()
        for button in sample.oracle["actions"]:
            teacher.observe_frame(env)
            tactics.add(teacher_tactic(env, teacher).stance)
            if env.step(button)[2]:
                break
        assert env._goal_credited
        return tactics
    finally:
        env.close()


@pytest.mark.parametrize("family", learner_families("skill", BLOCK_SMB_MC_FAMILIES))
@pytest.mark.parametrize("difficulty,index", [("easy", 100), ("medium", 103), ("hard", 106)])
def test_skill_demonstrations_keep_one_tactic_until_completion(family, difficulty, index):
    assert len(played_tactics(family, difficulty, index)) == 1


@pytest.mark.parametrize("family", MULTI_TACTIC_FAMILIES)
def test_switching_demonstrations_belong_to_tactics(family):
    witnesses = {
        "enemy_patrol": ("medium", 104),
        "retreat_recovery": ("easy", 101),
        "action_climb": ("hard", 106),
    }
    difficulty, index = witnesses.get(family, ("easy", 100))
    assert len(played_tactics(family, difficulty, index)) > 1
    assert family in learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    assert family not in learner_families("skill", BLOCK_SMB_MC_FAMILIES)


def test_curricula_partition_every_family_once():
    skills = learner_families("skill", BLOCK_SMB_MC_FAMILIES)
    tactics = learner_families("tactic", BLOCK_SMB_MC_FAMILIES)
    assert len(skills) == len(set(skills))
    assert len(tactics) == len(set(tactics))
    assert set(skills).isdisjoint(tactics)
    assert set(skills) | set(tactics) == set(BLOCK_SMB_MC_FAMILIES)
