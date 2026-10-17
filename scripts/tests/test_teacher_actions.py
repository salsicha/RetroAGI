"""The teacher's commands named as actions: run or jump to the reported
object Mario ends on, or wait."""

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
    assert ActionToken("wait").mode == "hold"
    assert ActionToken("run", ("surfaces", 0)).mode == "run"
    assert ActionToken("jump", ("enemies", 1)).mode == "jump"
    for verb, target in [
        ("wait", ("surfaces", 0)),  # a wait aims at nothing
        ("hold", None),  # no such verb
        ("run", None),  # a run ends somewhere
        ("jump", ("gaps", 0)),  # nothing ends in a gap
        ("jump", ("enemies", 6)),  # no such slot
    ]:
        with pytest.raises(ValueError):
            ActionToken(verb, target)


@pytest.mark.timeout(240)
def test_every_jump_is_named_by_where_it_ends():
    """Beside an enemy in a bypass lesson: jumps landing on the floor before
    it, onto it (which loses this lesson) and on the floor past it are three
    actions."""
    with Decision("skill_enemy_bypass") as d:
        tried = {t["action"]: t for t in tried_actions(d.env, d.state)}
        stomp = tried[ActionToken("jump", ("enemies", 0))]
        over = tried[ActionToken("jump", ("surfaces", 1))]
        assert ActionToken("jump", ("surfaces", 0)) in tried
        assert over["outcome"]["success"] and not stomp["outcome"]["success"]
        # Past it: the feet end beyond where the enemy is.
        m, e = d.env.mario, d.env.enemies[0]
        assert over["outcome"]["dx"] > e["x"] + e["w"] - (m["x"] + m["w"] / 2)
        # On it: the feet end on its top.
        assert abs(stomp["outcome"]["dy"] - stomp["outcome"]["object_dy"]) <= 3


@pytest.mark.timeout(120)
def test_runs_and_waits():
    with Decision("enemy_stomp") as d:
        env, state = d.env, d.state
        floor = ActionToken("run", ("surfaces", 0))
        assert teacher_action(env, state, SkillToken("hold", 0, 0)) == ActionToken("wait")
        assert teacher_example(env, state, SkillToken("hold", 0, 0))["action"] == ActionToken("wait")
        example = teacher_example(env, state, SkillToken("run", 40, 0))
        assert example["action"] == floor
        assert abs(example["outcome"]["dx"] - 40) <= 2
        # A run ending beyond the screen: the floor the vision sees there (the
        # floor is cut at the enemy: the piece past it).
        feet = env.mario["y"] + env.mario["h"]
        assert surface_slot(env, state.scene, env.camera_x + 400, feet) == ("surfaces", 1)
    with Decision("skill_enemy_bypass") as d:
        # Away from the goal: a run to where it ends, the same floor.
        assert teacher_action(d.env, d.state, SkillToken("run", -20, 0)) == floor


@pytest.mark.timeout(300)
def test_a_run_up_is_a_run_to_where_it_ends():
    from retroagi.stages.block_smb.teacher_actions import carry_out

    with Decision("skill_enemy_bypass") as d:
        env, state = d.env, d.state
        floor = ActionToken("run", ("surfaces", 0))
        # Keeping its speed into the jump after it (carry) is part of
        # carrying it out, not another action.
        run = SkillToken("run", 8, 0, carry=True)
        assert teacher_action(env, state, run) == floor == teacher_action(env, state, SkillToken("run", 8, 0))
        # The teacher's own run-up is how its choice is carried out.
        assert carry_out(env, state, floor, floor, run) == run
        # Chosen when the teacher would do something else: a run along the
        # floor.
        assert carry_out(env, state, floor).mode == "run"
    with Decision("action_jump_gap") as d:
        env, state = d.env, d.state
        far = min(
            (p["rect"] for p in env.platforms if p["rect"].left > env.mario["x"] + env.mario["w"]),
            key=lambda r: r.left,
        )
        # Mario stands on a surface's edge with his middle past it (by up to
        # half his width): his landing there is on it.
        assert surface_slot(env, state.scene, far.left - 3, far.top) == ("surfaces", 1)
        assert surface_slot(env, state.scene, far.left - 8, far.top) is None


@pytest.mark.timeout(240)
def test_the_world_models_records_hold_the_teachers_runs_as_well_as_its_jumps():
    from retroagi.core.world_model import COLUMN
    from retroagi.core.action_tokens import VERBS
    from retroagi.stages.block_smb.teacher_actions import action_rows, tried_runs

    with Decision("enemy_stomp") as d:
        runs = {r["action"]: r["outcome"] for r in tried_runs(d.env, d.state)}
        assert ActionToken("run", ("surfaces", 0)) in runs
        rows = action_rows(d.env, d.state, SkillToken("hold", 0, 0))
        verbs = [VERBS[int(v)] for v in rows[:, COLUMN["verb"]] if v >= 0]
        assert verbs[0] == "wait" and {"run", "jump"} <= set(verbs)


@pytest.mark.timeout(120)
def test_a_point_where_the_floor_is_cut_at_an_enemy_is_on_the_nearest_piece():
    with Decision("enemy_stomp") as d:
        env, scene = d.env, d.state.scene
        e = env.enemies[0]
        under = e["x"] + e["w"] / 2
        feet = next(p["rect"].top for p in env.platforms if p["rect"].left <= under <= p["rect"].right)
        # The scene cuts the floor out under the enemy.
        assert not any(s.x0 <= under - env.camera_x <= s.x1 for s in scene.surfaces)
        slot = surface_slot(env, scene, under, feet)
        assert slot is not None and slot[0] == "surfaces"


@pytest.mark.timeout(120)
def test_a_steer_that_comes_down_on_an_enemy_ends_on_it():
    from retroagi.stages.block_smb.teacher_actions import _trial_of

    with Decision("enemy_stomp") as d:
        env, m, e = d.env, d.env.mario, d.env.enemies[0]
        m["x"], m["y"] = e["x"] + e["w"] / 2 - m["w"] / 2, e["y"] - m["h"] - 30
        m["on_ground"] = False
        steer = SkillToken("run", 0, round(e["y"] + e["h"] - (m["y"] + m["h"])))
        result = _trial_of(env, d.state, steer)
        assert 0 in result.outcome[2]
        assert teacher_action(env, d.state, steer, result) == ActionToken("run", ("enemies", 0))


@pytest.mark.timeout(240)
def test_the_teacher_carries_out_the_skills_chosen_action():
    from retroagi.stages.block_smb.teacher_actions import carry_out

    with Decision("action_jump_gap") as d:
        working = [t["action"] for t in tried_actions(d.env, d.state) if t["outcome"]["success"]]
        assert working
        for action in working[:3]:
            command = carry_out(d.env, d.state, action)
            assert command is not None and command.mode == "jump"
            assert teacher_action(d.env, d.state, command) == action
        # Its own choice, already found, is carried out as it is.
        mine = SkillToken("jump", 30, 0)
        assert carry_out(d.env, d.state, working[0], working[0], mine) == mine
        # A wait where the teacher would move holds for one frame; an object
        # the vision does not report: a hold.
        assert carry_out(d.env, d.state, ActionToken("wait")) == SkillToken("hold", 0, 0)
        assert carry_out(d.env, d.state, ActionToken("jump", ("surfaces", 13))).mode == "hold"
        # An action the teacher names no version of (a run across the pit)
        # is still carried out, straight at where it ends (user, 2026-10-17).
        across = carry_out(d.env, d.state, ActionToken("run", ("surfaces", 1)))
        assert across.mode == "run" and across.x > 0


@pytest.mark.timeout(600)
def test_a_held_out_episode_records_the_teachers_choice_at_each_skill_decision():
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.stages.block_smb import layered_train as lt
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    # Unlabelled, the skill playing every decision itself (as on held-out
    # layouts): the teacher's choice at each one is still recorded (carry_out
    # asks for it), so the skill's choices are judged against it.
    task = lt.EpisodeTask(
        index=0,
        family="action_walk",
        split="validation",
        seed=0,
        sample_index=0,
        difficulty="easy",
        teacher_share=0.0,
        label=False,
    )
    record = lt.play_episodes(
        reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1
    )[0]
    count = len(record.decision_frames)
    assert count and len(record.teacher_verb) == count == len(record.same_as_teacher)
    assert not record.played_teacher.any() and (record.teacher_verb >= 0).all()
    same = record.same_as_teacher
    assert (record.picks["verb"][same] == record.teacher_verb[same]).all()
    assert lt.choices_against_teacher([record])["action_walk"]["decisions"] == count
