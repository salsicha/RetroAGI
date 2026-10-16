"""Play without the teacher (user, 2026-10-15/16): the world model's predicted
outcome is the executor's destination; the collision trackers end a wait or
a walk when a hit is due within the lead a jump needs (smb_threat), and aim a
stomp at their contact point. Stand-in networks: the tests check the wiring,
not the networks."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from types import SimpleNamespace

import numpy as np
import torch

from retroagi.core.action_tokens import VERBS, ActionToken
from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents, destination_of
from retroagi.core.smb_observer import VisionObserver
from retroagi.core.smb_scene_labels import EnemyView, MarioView, SceneObservation, Surface
from retroagi.core.smb_threat import THREAT_SLACK, jump_lead
from retroagi.core.target_tracker import BINS, Collision, CollisionTracker, CollisionTrackers
from retroagi.core.tokens import SkillToken, tactic_token


def test_a_predicted_outcome_is_the_executors_destination():
    outcome = [37.4, -10.2, 0.0, 0.0, 41.6]
    assert destination_of(ActionToken("hold"), outcome) == SkillToken("hold", 0, 0, frames=42)
    run = ActionToken("run_to", ("surfaces", 0))
    assert destination_of(run, outcome) == SkillToken("run", 37, -10)
    approach = ActionToken("approach", ("surfaces", 1))
    assert destination_of(approach, outcome) == SkillToken("run", 37, -10, carry=True)
    jump = ActionToken("jump_over", ("enemies", 0))
    assert destination_of(jump, outcome) == SkillToken("jump", 37, -10)
    # A stomp goes where the trackers expect to meet the enemy.
    stomp = ActionToken("stomp", ("enemies", 0))
    assert destination_of(stomp, outcome, contact=(44.0, -8.0)) == SkillToken("jump", 44, -8)
    assert destination_of(ActionToken("hold"), [0, 0, 0, 0, 900.0]).frames == 255


def scene(enemy_x=None, air=False):
    enemies = [EnemyView((enemy_x, 202, enemy_x + 10, 208), "walker")] if enemy_x else []
    return SceneObservation(
        mario=MarioView(
            box=(100, 192 - (20 if air else 0), 112, 208 - (20 if air else 0)),
            facing_right=True,
            support="air" if air else "ground",
            on_something=not air,
        ),
        enemies=tuple(enemies),
        surfaces=(Surface(8, 248, 208, False),),
        gaps=(),
    )


class Pictures:
    def __init__(self, pictures):
        self.pictures = list(pictures)

    def eval(self):
        return self

    def scene(self, screens):
        return [self.pictures.pop(0) for _ in screens]


def test_the_world_model_steps_once_per_action_and_its_state_carries():
    policy = LayeredSMBPolicy().eval()
    calls = []
    original = policy.memory.predict

    def predict(*args):
        calls.append(len(args[5]))
        out, state = original(*args)
        out["mean"] = out["mean"] * 0 + torch.tensor([20.0, 0.0, 0.0, 0.0, 30.0])
        return out, state

    policy.memory.predict = predict
    agents = SMBAgents(VisionObserver(Pictures([scene()])), policy, "cpu", world_model=True)
    made = [
        SimpleNamespace(
            action=ActionToken("hold"), skill=None, execution_feedback=[0.0] * 9, predicted=None
        )
    ]
    from retroagi.core.smb_agent import scene_rows, stack_rows

    pictures = [scene()]
    now = policy.encode_scene(stack_rows(scene_rows(pictures), "cpu"))
    agents._actions([agents.copies[0]], made, [0], [0], now, pictures)
    assert calls == [1]
    assert made[0].skill == SkillToken("hold", 0, 0, frames=30)
    first = agents.copies[0].world
    assert first is not None and first[0].shape == (policy.memory.width,)
    agents._actions([agents.copies[0]], made, [0], [0], now, pictures)
    assert calls == [1, 1] and not torch.equal(agents.copies[0].world[0], first[0])


def hit_at(frames):
    """A stand-in estimate: a hit ``frames`` from now (or none)."""
    hit_by = np.zeros(BINS)
    if frames is not None:
        hit_by[(frames - 1) // 2 :] = 1.0
    return Collision(
        contact=float(frames is not None),
        hit_by=hit_by,
        stomp_by=np.zeros(BINS),
        frames=float(frames or 0),
        place=(0.0, 0.0),
        sigma=(1.0, 1.0),
    )


def play_wait(hit_frames, frames=12, air=False):
    """A copy waiting (a hold of 60 frames) while a walker comes at 0.5 px a
    frame; the trackers (stand-in) say a hit comes ``hit_frames`` from now.
    Returns each frame's end reason."""
    pictures = [scene(enemy_x=150 - t // 2, air=air) for t in range(frames)]
    torch.manual_seed(0)
    agents = SMBAgents(
        VisionObserver(Pictures(pictures)),
        LayeredSMBPolicy().eval(),
        "cpu",
        tracker=CollisionTracker(8),
    )
    original = CollisionTrackers.estimate
    CollisionTrackers.estimate = lambda self, identity, features=None: hit_at(hit_frames)
    try:
        wait = SkillToken("hold", 0, 0, frames=60)

        def given(copies, scenes):
            return {"tactic": [tactic_token("hold_ground")], "skill": [wait]}

        screen = [np.zeros((240, 256, 3), np.uint8)]
        return [agents.act(screen, [0], given=given)[0].ended for _ in range(frames)]
    finally:
        CollisionTrackers.estimate = original


def test_a_wait_ends_when_a_hit_comes_within_the_lead_a_jump_needs():
    lead = jump_lead(0.5, 10, 6, (12, 16), 0.0, 1)
    assert lead is not None
    soon = play_wait(int(lead + THREAT_SLACK))
    assert "threat" in soon
    later = play_wait(int(lead + THREAT_SLACK) + 20)
    assert "threat" not in later
    assert "threat" not in play_wait(None)


def test_nothing_is_ended_in_the_air():
    assert "threat" not in play_wait(1, air=True)


def test_a_faster_enemy_must_be_jumped_from_farther_away():
    leads = {c: jump_lead(c, 10, 6, (12, 16), 0.0, 1) for c in (0.3, 0.6, 1.2)}
    distances = {c: c * (lead + THREAT_SLACK) for c, lead in leads.items()}
    assert distances[0.3] < distances[0.6] < distances[1.2]


def test_with_the_teacher_nothing_is_ended_by_the_trackers():
    agents = SMBAgents(
        VisionObserver(Pictures([scene()])),
        LayeredSMBPolicy().eval(),
        "cpu",
        tracker=CollisionTracker(8),
        carry_out=lambda number, action: SkillToken("hold", 0, 0),
    )
    assert agents.copies[0].threat_watch is None


def test_each_execution_plays_and_only_the_teachers_uses_the_teacher(monkeypatch):
    import pytest

    from retroagi.stages.block_smb import layered_train as lt
    from retroagi.stages.block_smb.teacher_replay import reference_vision

    calls = []
    original = lt._Lane.carry_out

    def carry_out(self, action):
        calls.append(self.task.execution)
        return original(self, action)

    monkeypatch.setattr(lt._Lane, "carry_out", carry_out)
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    for execution in lt.EXECUTIONS:
        task = lt.EpisodeTask(
            index=0,
            family="action_wait",
            split="validation",
            seed=0,
            sample_index=0,
            difficulty="easy",
            teacher_share=0.0,
            label=False,
            frames=90,
            execution=execution,
        )
        (record,) = lt.play_episodes(
            reference_vision()[0],
            policy,
            "skill",
            [task],
            "cpu",
            1,
            tracker=CollisionTracker(8) if execution in ("trackers", "deployed") else None,
        )
        assert record.end and len(record.ended) == len(record.buttons)
    # The teacher carries out actions only where it is asked to.
    assert set(calls) <= {"teacher", "trackers"}
    with pytest.raises(ValueError):
        lt.play_episodes(
            reference_vision()[0],
            policy,
            "skill",
            [task, dataclasses_replace(task, execution="teacher")],
            "cpu",
            2,
        )


def dataclasses_replace(task, **changes):
    import dataclasses

    return dataclasses.replace(task, **changes)
