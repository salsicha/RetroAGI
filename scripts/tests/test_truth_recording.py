"""The simulator's truth recorded with an episode, for labels made afterwards
from what really happened (user, 2026-10-16)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from retroagi.core.action_tokens import VERBS
from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_kinematics import SPRITE_SLOTS
from retroagi.stages.block_smb import layered_train as lt
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.teacher_replay import reference_vision
from retroagi.stages.block_smb.truth_recording import match_slots


def played(record_truth=True, family="stomp_mount"):
    task = lt.EpisodeTask(
        index=0,
        family=family,
        split="validation",
        seed=0,
        sample_index=0,
        difficulty="easy",
        teacher_share=1.0,
        label=True,
        record_truth=record_truth,
    )
    return lt.play_episodes(
        reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1
    )[0]


# Each episode is played through the vision model on the CPU.
pytestmark = pytest.mark.timeout(900)


@pytest.fixture(scope="module")
def stomp():
    return played()


def test_truth_has_a_row_per_frame_and_one_for_the_state_the_last_button_left(stomp):
    truth = stomp.truth
    frames = len(stomp.buttons)
    assert stomp.won and len(truth["mario"]) == frames + 1
    assert truth["enemies"].shape[:2] == truth["enemy_state"].shape == (frames + 1, 1)
    # The lesson ends at the stomp: alive before the last button, dead after.
    assert truth["enemy_state"][-2, 0] == 1 and truth["enemy_state"][-1, 0] == 2
    assert stomp.sprite_kinematics.shape == (frames, SPRITE_SLOTS, 6)
    assert stomp.mario_kinematics.shape == (frames, 3)
    assert stomp.slot_match.shape == (frames, SPRITE_SLOTS)
    assert len(stomp.ended) == frames


def test_the_walkers_slot_is_matched_to_it_while_it_is_seen(stomp):
    seen = stomp.sprite_kinematics[:, 0, 0] == 1
    assert seen.any() and (stomp.slot_match[seen, 0] == 0).all()
    assert (stomp.slot_match[:, 1:] == -1).all()


def test_executed_commands_and_the_actions_they_carried_out_are_kept(stomp):
    count = len(stomp.decision_frames)
    assert stomp.commands.shape == (count, len(lt.COMMAND_FIELDS))
    assert set(stomp.commands[:, 0]) <= {0, 1, 2}
    # The teacher played every decision: its named choice is the action.
    named = stomp.labels["action_valid"].astype(bool)
    assert (stomp.action_verb[named] == stomp.labels["verb"][named]).all()
    assert VERBS[stomp.action_verb[named][-1]] == "stomp"
    # A decision comes when the action before it ended.
    for frame in stomp.decision_frames[1:]:
        assert stomp.ended[frame] > 0


def test_nothing_more_is_recorded_without_asking():
    record = played(record_truth=False, family="action_walk")
    assert record.truth is None and record.slot_match is None
    assert record.sprite_kinematics is None and record.ended is not None


def test_slots_are_matched_only_when_unambiguous():
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 400,
                "mario": [40, 208],
                "platforms": [[0, 220, 400, 20]],
                "enemies": [[150, 206, 150, 150, 0.0, 1], [151, 206, 151, 151, 0.0, 1]],
                "goal": [380, 200, 16, 20],
            }
        )
        from retroagi.core.smb_scene_labels import scene_from_labels

        frame = env.true_frame()
        scene = scene_from_labels(env.scene_labels())
        assert (match_slots(scene, frame) == -1).all()
    finally:
        env.close()
