"""The teacher's commands named as actions: a verb aimed at a reported object."""

import pytest

from retroagi.core.action_tokens import (
    POINTER_TOKENS,
    ActionToken,
    pointer_index,
    pointer_target,
)
from retroagi.core.smb_observer import SCENE_SLOTS
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.stages.block_smb.controller_teacher import SkillToken
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_actions import (
    surface_slot,
    teacher_action,
    teacher_example,
    tried_actions,
)
from retroagi.stages.block_smb.teacher_tokens import episode_teacher


def sample(family, index=1):
    return sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=0,
        sample_index=index,
        family=family,
        difficulty="medium",
        calibrate_deadline=False,
    ).scenario


class Decision:
    """The first decision of a layout: the simulator and the teacher's state."""

    def __init__(self, family, index=1):
        self.scenario = sample(family, index)
        self.env = MarioScenarioEnv()
        self.env.reset(scenario=self.scenario)
        self.env.render = lambda: None
        self.state = episode_teacher(self.scenario)
        self.state.scene = scene_from_labels(self.env.scene_labels())
        self.state.execution = SpatialFeedback()
        self.state.execution.observe(self.state.scene)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.env.close()


def test_pointer_names_every_slot_once():
    indices = [
        pointer_index((name, slot)) for name in SCENE_SLOTS for slot in range(SCENE_SLOTS[name])
    ]
    assert sorted(indices) == list(range(2, 2 + POINTER_TOKENS))
    assert all(pointer_target(pointer_index(t)) == t for t in [("enemies", 0), ("gaps", 4)])


def test_actions_aim_only_at_their_kinds():
    assert ActionToken("hold").mode == "hold"
    assert ActionToken("stomp", ("enemies", 1)).mode == "jump"
    for verb, target in [
        ("hold", ("surfaces", 0)),
        ("stomp", ("surfaces", 0)),
        ("land_on", ("enemies", 0)),
        ("run_to", None),
        ("jump_over", ("enemies", 6)),
    ]:
        with pytest.raises(ValueError):
            ActionToken(verb, target)


@pytest.mark.timeout(240)
def test_every_jump_is_named_by_what_it_does():
    """Beside an enemy in a bypass lesson: jumps landing on the floor, onto
    the enemy (which loses this lesson) and past it are three actions."""
    with Decision("skill_enemy_bypass") as d:
        tried = {t["action"]: t for t in tried_actions(d.env, d.state)}
        over = tried[ActionToken("jump_over", ("enemies", 0))]
        stomp = tried[ActionToken("stomp", ("enemies", 0))]
        assert ActionToken("land_on", ("surfaces", 0)) in tried
        assert over["safe_versions"] > 0 and over["outcome"]["safe"]
        assert stomp["safe_versions"] == 0 and not stomp["outcome"]["safe"]
        # Past it: the feet end beyond where the enemy then is.
        assert over["outcome"]["dx"] > over["outcome"]["object_dx"] + 8
        # On it: the feet end on its top.
        assert abs(stomp["outcome"]["dy"] - stomp["outcome"]["object_dy"]) <= 3


@pytest.mark.timeout(120)
def test_runs_and_holds():
    with Decision("enemy_stomp") as d:
        env, state = d.env, d.state
        assert teacher_action(env, state, SkillToken("hold", 0, 0)) == ActionToken("hold")
        example = teacher_example(env, state, SkillToken("run", 40, 0))
        assert example["action"] == ActionToken("run_to", ("surfaces", 0))
        assert abs(example["outcome"]["dx"] - 40) <= 2
        # A run ending beyond the screen: the floor the vision sees there (the
        # floor is cut at the enemy: the piece past it).
        feet = env.mario["y"] + env.mario["h"]
        assert surface_slot(env, state.scene, env.camera_x + 400, feet) == ("surfaces", 1)
    with Decision("skill_enemy_bypass") as d:
        # Away from the goal with the enemy near: backing away from it.
        assert teacher_action(d.env, d.state, SkillToken("run", -20, 0)) == ActionToken(
            "back_away", ("enemies", 0)
        )


@pytest.mark.timeout(120)
def test_a_run_up_approaches_what_its_jump_lands_on():
    with Decision("skill_enemy_bypass") as d:
        env, state = d.env, d.state
        m = env.mario
        run = SkillToken("run", 8, 0)
        enemy = env.enemies[0]
        on_enemy = round(enemy["x"] + enemy["w"] / 2 - (m["x"] + 8 + m["w"] / 2))
        state.notes["planned_jump"] = (
            run,
            SkillToken("jump", on_enemy, round(enemy["y"] - (m["y"] + m["h"]))),
            m["x"] + 8,
        )
        assert teacher_action(env, state, run) == ActionToken("approach", ("enemies", 0))
        state.notes["planned_jump"] = (run, SkillToken("jump", -40, 0), m["x"] + 8)
        assert teacher_action(env, state, run) == ActionToken("approach", ("surfaces", 0))


@pytest.mark.timeout(240)
def test_the_teacher_carries_out_the_skills_chosen_action():
    from retroagi.stages.block_smb.teacher_actions import carry_out

    with Decision("action_jump_gap") as d:
        safe = [t["action"] for t in tried_actions(d.env, d.state) if t["safe_versions"]]
        assert safe
        for action in safe[:3]:
            command = carry_out(d.env, d.state, action)
            assert command is not None and command.mode == "jump"
            assert teacher_action(d.env, d.state, command) == action
        # Its own choice, already found, is carried out as it is.
        mine = SkillToken("jump", 30, 0)
        assert carry_out(d.env, d.state, safe[0], safe[0], mine) == mine
        # A hold holds; an object the vision does not report has no version.
        assert carry_out(d.env, d.state, ActionToken("hold")).mode == "hold"
        assert carry_out(d.env, d.state, ActionToken("land_on", ("surfaces", 13))) is None
        # Nor does an action aimed at a gap: the teacher names none.
        assert carry_out(d.env, d.state, ActionToken("approach", ("gaps", 0))) is None
        assert carry_out(d.env, d.state, ActionToken("jump_over", ("gaps", 0))) is None


@pytest.mark.timeout(600)
def test_a_run_the_teacher_follows_with_a_jump_is_labelled_an_approach():
    from retroagi.core.layered_policy import POINTER_START, LayeredSMBPolicy
    from retroagi.core.action_tokens import VERBS
    from retroagi.stages.block_smb import layered_train as lt
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    # The teacher runs to the pit's edge and then jumps; whether it had
    # planned the jump before running is not in the scene, so the run is an
    # approach to where the jump lands either way.
    task = lt.EpisodeTask(
        index=0,
        family="single_gap",
        split="train",
        seed=2026100909,
        sample_index=0,
        difficulty="hard",
        teacher_share=1.0,
        label=True,
    )
    record = lt.play_episodes(
        reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1
    )[0]
    named = [
        (VERBS[int(v)], pointer_target(int(p) + POINTER_START))
        for v, p, ok in zip(
            record.labels["verb"], record.labels["pointer"], record.labels["action_valid"]
        )
        if ok
    ]
    jumps = [i for i, (verb, _) in enumerate(named) if verb == "land_on"]
    assert record.won and jumps
    assert all(named[i - 1][0] == "approach" for i in jumps if i and named[i - 1][0] != "land_on")
