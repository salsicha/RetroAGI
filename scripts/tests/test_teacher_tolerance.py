"""The teacher teaches moves that still work when started a little later, and
a run can give its weak families extra layouts that are mostly hard."""

import dataclasses
import random

import pytest

from retroagi.stages.block_smb import controller_teacher as ct
from retroagi.stages.block_smb import layered_train as lt
from retroagi.stages.block_smb.controller_teacher import SkillToken

from test_teacher_actions import Decision


@pytest.mark.timeout(600)
def test_a_move_that_works_only_from_this_moment_is_not_taught():
    with Decision("landing_enemy", 3) as d:
        env, state = d.env, d.state
        target = ct.teacher_target(env)
        # A hop toward the walker coming at Mario works now, but not started
        # two or four frames later: the walker is nearer.
        hop = SkillToken("jump", 18, 0)
        assert ct._works_now(ct._trial(env, state, hop, target))
        assert not ct._tolerant(env, state, target, hop)
        assert ct._tolerant(env, state, target, SkillToken("hold", 0, 0))
        taught = ct.destination(env, state, None)
        assert taught != hop and ct._tolerant(env, state, target, taught)


@pytest.mark.timeout(600)
@pytest.mark.parametrize("family", ["action_climb", "action_climb_back", "action_jump_gap"])
def test_a_single_jump_lesson_is_one_jump_from_where_mario_stands(family):
    from retroagi.stages.block_smb.family_atlas import atlas_layout, play_teacher

    # The settings where the teacher ran up (or stepped back) before jumping.
    for difficulty, q in [("medium", 0.0), ("medium", 1.0), ("hard", 0.0), ("easy", 1.0)]:
        scenario, why = atlas_layout(family, difficulty, q)
        if scenario is None:
            # Not a one-action lesson from there: the sampler redraws it.
            assert why in ("two_actions", "teacher_loses", "no_verified_route")
            continue
        assert scenario["action_jump_direction"] in (1, -1)
        played = play_teacher(scenario, family, difficulty)
        assert played["won"] and len(played["decisions"]) == 1


def test_extra_layouts_can_be_mostly_hard_without_changing_the_others():
    config = lt.LayeredTrainConfig(learner="skill", seed=7, extra_layouts="enemy_hop=16")
    extra = {"enemy_hop": 16}
    before = [(t.family, t.difficulty) for t in lt._tasks(config, "train", 8, 2, 1.0, True, extra=extra)]
    # By default the extra layouts are drawn as the others are: the layouts
    # are those drawn before extra_difficulty_weights existed.
    drawn = []
    for family in lt.learner_families("skill", config.families):
        rng = random.Random(f"{config.seed}|train|2|{family}")
        drawn += [
            (family, d)
            for d in rng.choices(lt.DIFFICULTIES, weights=config.difficulty_weights, k=8 + extra.get(family, 0))
        ]
    assert before == drawn
    hard = dataclasses.replace(config, extra_difficulty_weights=(0.0, 1.0, 3.0))
    after = [(t.family, t.difficulty) for t in lt._tasks(hard, "train", 8, 2, 1.0, True, extra=extra)]
    hop_before = [d for f, d in before if f == "enemy_hop"]
    hop_after = [d for f, d in after if f == "enemy_hop"]
    # A family's usual layouts are unchanged; its extra ones are mostly hard
    # and never easy; other families are unchanged.
    assert hop_after[:8] == hop_before[:8]
    assert "easy" not in hop_after[8:] and hop_after[8:].count("hard") > 8
    assert [x for x in after if x[0] != "enemy_hop"] == [x for x in before if x[0] != "enemy_hop"]
