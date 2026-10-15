"""Teacher gates must use the learner's real episodes and preserve failed layouts."""

import json
from concurrent.futures import Future
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from retroagi.stages.block_smb import layered_train as lt
from retroagi.stages.block_smb.teacher_qualification import qualify_teachers


def task(**changes):
    original = lt.EpisodeTask(
        0,
        "flat_run",
        "validation",
        7,
        5,
        0.0,
        False,
        frames=600,
        difficulty="medium",
        scenario={"mario": [40, 208], "platforms": [[0, 220, 256, 20]], "goal": [210, 200, 16, 20]},
    )
    return replace(original, **changes)


@pytest.mark.parametrize("won,valid", [(False, [True]), (True, [False]), (True, [])])
def test_failed_or_unlabelled_demonstration_blocks_qualification_and_saves_layout(
    tmp_path, won, valid
):
    original = task()
    record = SimpleNamespace(
        won=won,
        end="goal" if won else "timeout",
        frames=600,
        labels={"valid": np.array(valid, dtype=bool)},
    )

    def play(tasks, *, teacher_only):
        assert teacher_only
        assert tasks[0].scenario is original.scenario
        assert tasks[0].frames == 600 and tasks[0].sample_index == 5
        assert tasks[0].teacher_share == 1 and tasks[0].label and not tasks[0].explore
        return [record]

    report = tmp_path / "qualification.json"
    with pytest.raises(RuntimeError, match="Teacher qualification failed"):
        qualify_teachers(SimpleNamespace(play=play), [original], report)
    saved = json.loads(report.read_text())
    assert not saved["all_passed"]
    assert saved["episodes"][0]["scenario"] == original.scenario
    assert saved["episodes"][0]["frame_limit"] == 600
    assert original.teacher_share == 0 and not original.label


def test_prepared_scenarios_are_not_redrawn():
    pool = object.__new__(lt.EpisodePool)

    class Executor:
        def submit(self, *args):
            pytest.fail("An already prepared layout must not be sampled again")

    pool.pool = Executor()
    original = task()
    prepared = pool.with_scenarios([original])
    assert prepared[0].scenario is original.scenario


def test_tactic_qualification_runs_spatial_teacher_instead_of_learned_skill():
    pool = object.__new__(lt.EpisodePool)
    pool.config = SimpleNamespace(learner="tactic", lanes=1)
    pool.weights, pool.version = "unused.pt", 1

    class Executor:
        def submit(self, function, job):
            assert job[2] == "skill"
            future = Future()
            future.set_result(["demonstration"])
            return future

    pool.pool = Executor()
    assert pool.play([task()], teacher_only=True) == ["demonstration"]


@pytest.mark.parametrize("failed_split", ["validation", "train"])
def test_training_stops_before_updates_when_qualification_fails(
    monkeypatch, tmp_path, failed_split
):
    from retroagi.stages.block_smb import teacher_qualification

    calls = []

    class Pool:
        def __init__(self, *args):
            pass

        def publish(self, *args):
            pass

        def with_scenarios(self, tasks):
            return [replace(t, scenario=task().scenario) for t in tasks]

        def play(self, *args, **kwargs):
            pytest.fail("Learner collection must wait for qualification")

        def close(self):
            calls.append("close")

    def qualify(pool, tasks, report):
        split = tasks[0].split
        calls.append(split)
        if split == failed_split:
            raise RuntimeError("Teacher qualification failed")
        return []

    def update(*args, **kwargs):
        pytest.fail("Weights must not change before qualification passes")

    from retroagi.core import action_predictor

    monkeypatch.setattr(lt, "EpisodePool", Pool)
    monkeypatch.setattr(teacher_qualification, "qualify_teachers", qualify)
    monkeypatch.setattr(lt.torch.optim.AdamW, "step", update)
    # The stand-in pool loads no predictor; its file only records what it reads.
    monkeypatch.setattr(action_predictor, "reading_digest", lambda policy: "stand-in")
    lt.torch.save({"reads": "stand-in"}, tmp_path / "predictor.pt")
    config = lt.LayeredTrainConfig(
        families=("flat_run",),
        rounds=1,
        output=str(tmp_path),
        device="cpu",
        predictor=str(tmp_path / "predictor.pt"),
    )
    with pytest.raises(RuntimeError, match="Teacher qualification failed"):
        lt.train_layer(config)
    expected = ["validation"] + (["train"] if failed_split == "train" else []) + ["close"]
    assert calls == expected


def test_actual_episode_stall_limit_also_fails_teacher_qualification(monkeypatch, tmp_path):
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.core.smb_scene_labels import scene_from_labels
    from retroagi.core.tokens import SkillToken
    from retroagi.stages.block_smb import teacher_tokens
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    env = MarioScenarioEnv()
    env.reset(scenario=task().scenario)
    scene = scene_from_labels(env.scene_labels())
    env.close()
    observer = SimpleNamespace(observe=lambda screens: [scene for _ in screens])
    policy = LayeredSMBPolicy().eval()
    monkeypatch.setattr(teacher_tokens, "teacher_skill", lambda *args: SkillToken("hold", 0, 0))

    def play(tasks, *, teacher_only):
        assert teacher_only
        return lt.play_episodes(observer, policy, "skill", tasks, "cpu", 1)

    report = tmp_path / "stall.json"
    with pytest.raises(RuntimeError, match="timeout"):
        qualify_teachers(SimpleNamespace(play=play), [task()], report)
    episode = json.loads(report.read_text())["episodes"][0]
    assert episode["frames"] == lt.STALL_FRAMES + 2
    assert episode["frames"] < episode["frame_limit"]


def test_a_teacher_without_a_command_fails_qualification_instead_of_crashing(
    monkeypatch, tmp_path
):
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.core.smb_scene_labels import scene_from_labels
    from retroagi.stages.block_smb import teacher_tokens
    from retroagi.stages.block_smb.env import MarioScenarioEnv

    env = MarioScenarioEnv()
    env.reset(scenario=task().scenario)
    scene = scene_from_labels(env.scene_labels())
    env.close()
    observer = SimpleNamespace(observe=lambda screens: [scene for _ in screens])
    policy = LayeredSMBPolicy().eval()
    # No command anywhere, and no action predictor for the skill's own actions.
    monkeypatch.setattr(teacher_tokens, "teacher_skill", lambda *args: None)

    def play(tasks, *, teacher_only):
        return lt.play_episodes(observer, policy, "skill", tasks, "cpu", 1)

    report = tmp_path / "unlabelled.json"
    with pytest.raises(RuntimeError, match="Teacher qualification failed"):
        qualify_teachers(SimpleNamespace(play=play), [task()], report)
    episode = json.loads(report.read_text())["episodes"][0]
    assert not episode["complete_labels"] and not episode["passed"]
    # It ends there, not after holding until the stall limit.
    assert episode["end"] == "no_teacher_command" and episode["frames"] <= 2


@pytest.mark.timeout(180)
def test_retreating_enemy_teacher_finishes_inside_training_budget():
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_replay import reference_vision, replay

    sample = sample_block_smb_monte_carlo_scenario(
        split="validation",
        seed=2026100909,
        sample_index=5,
        family="enemy_stomp",
        difficulty="medium",
    )
    result = replay(
        sample.scenario, family="enemy_stomp", max_frames=600, observer=reference_vision()[0]
    )
    assert result["won"]
    assert result["frames"] < 400
    assert len(result["commands"]) < 15


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "family,split,index,difficulty",
    [
        ("action_jump_down_back", "validation", 2, "easy"),
        ("action_jump_down", "train", 5, "medium"),
        ("action_jump_down_back", "train", 1004, "medium"),
        ("action_jump_down_back", "train", 1007, "medium"),
    ],
)
def test_ledge_teacher_never_falls_back_to_policy(family, split, index, difficulty, tmp_path):
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    sample = sample_block_smb_monte_carlo_scenario(
        split=split, seed=2026100909, sample_index=index, family=family, difficulty=difficulty
    )
    original = task(
        family=family,
        split=split,
        seed=2026100909,
        sample_index=index,
        difficulty=difficulty,
        scenario=sample.scenario,
    )
    policy = LayeredSMBPolicy().eval()
    observer = reference_vision()[0]

    def play(tasks, *, teacher_only):
        assert teacher_only
        return lt.play_episodes(observer, policy, "skill", tasks, "cpu", 1)

    records = qualify_teachers(SimpleNamespace(play=play), [original], tmp_path / "edges.json")
    assert records[0].frames < 150


def test_the_skill_plays_its_own_choice_unless_the_teacher_plays_the_decision():
    from types import SimpleNamespace

    from retroagi.core.tokens import SkillToken

    class Lane:
        def __init__(self, plays_teacher, skill):
            self.teacher = SimpleNamespace()
            self.asked = {"tactic": None, "skill": skill, "plays_teacher": plays_teacher}

        def ask_teacher(self, learner, scene):
            return self.asked

    command = SkillToken("jump", 40, 0)
    lanes = {0: Lane(False, command), 1: Lane(True, command), 2: Lane(True, None)}
    tokens = lt._teacher_given("skill", lanes)([0, 1, 2], [None, None, None])
    # Its own (None: the policy chooses), the teacher's, and a hold where the
    # teacher plays but has no command (that decision has no label).
    assert tokens["skill"] == [None, command, SkillToken("hold", 0, 0)]
