"""Collision labels and executed outcomes made from recorded truth: the
collision trackers' and the world model's targets (user, 2026-10-16)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

from retroagi.core.action_outcomes import COLUMN, executed_outcomes
from retroagi.core.action_tokens import VERBS
from retroagi.core.collision_labels import KINDS, action_ends, collision_labels, slot_labels
from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.stages.block_smb import layered_train as lt
from retroagi.stages.block_smb.collision_truth import collisions
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.teacher_replay import reference_vision
from retroagi.stages.block_smb.truth_recording import TruthRecorder

pytestmark = pytest.mark.timeout(900)


def standing_recording(speed, frames=200):
    """Mario stands while a walker comes at him: the truth of every frame
    (and the state after the last), and the live collision truth then."""
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 400,
                "mario": [40, 204],
                "platforms": [[0, 220, 400, 20]],
                "enemies": [[150, 206, 0, 300, speed, -1]],
                "goal": [380, 200, 16, 20],
            }
        )
        for _ in range(4):
            env.step(0)
        recorder, live = TruthRecorder(), []
        for _ in range(frames):
            recorder.add(env.true_frame())
            live.append(collisions(env))
            _, _, done, truncated, _ = env.step(0)
            if done or truncated:
                break
        recorder.add(env.true_frame())
        return recorder.arrays(), live
    finally:
        env.close()


def test_a_standing_marios_labels_are_the_simulators_collision_truth():
    truth, live = standing_recording(0.7)
    frames = len(live)
    labels = collision_labels(truth, action_ends([0], frames))
    found = 0
    for t, now in enumerate(live):
        if now:
            (contact,) = now
            assert labels.frames[t, 0] == contact.frames
            assert KINDS[labels.kind[t, 0]] == contact.kind == "hit"
            assert labels.place[t, 0].tolist() == [0.0, 0.0]  # where he stands
            found += 1
        else:
            assert labels.frames[t, 0] == 0
    assert found > 50
    # Up to the hit: every frame within 96 of it knows it.
    assert labels.censor[:, 0].max() == 0


def test_nothing_is_known_past_the_recordings_end():
    truth, live = standing_recording(0.2, frames=60)
    frames = len(live)
    labels = collision_labels(truth, action_ends([0], frames))
    # The walker is too slow to arrive in 60 frames: the last frames know
    # nothing past the recording's end.
    t = frames - 10
    assert labels.frames[t, 0] == 0 and labels.censor[t, 0] == frames - t + 1


def test_a_recording_without_enemies_labels_no_slot():
    truth, live = standing_recording(0.2, frames=20)
    for name in ("enemies", "enemies_drawn"):
        truth[name] = truth[name][:, :0]
    truth["enemy_state"], truth["stompable"] = truth["enemy_state"][:, :0], truth["stompable"][:0]
    labels = collision_labels(truth, action_ends([0], len(live)))
    slots = slot_labels(labels, np.full((len(live), 9), -1), 6)
    assert slots["frames"].shape == (len(live), 6) and slots["place"].shape == (len(live), 6, 2)
    assert not slots["labelled"].any()


@pytest.fixture(scope="module")
def stomp():
    task = lt.EpisodeTask(
        index=0,
        family="stomp_mount",
        split="validation",
        seed=0,
        sample_index=0,
        difficulty="easy",
        teacher_share=1.0,
        label=True,
        record_truth=True,
    )
    return lt.play_episodes(
        reference_vision()[0], LayeredSMBPolicy().eval(), "skill", [task], "cpu", 1
    )[0]


def test_a_stomp_in_progress_is_labelled_a_stomp_at_the_kill(stomp):
    frames = len(stomp.buttons)
    labels = collision_labels(stomp.truth, action_ends(stomp.decision_frames, frames))
    jump = int(stomp.decision_frames[-1])
    kill = frames  # the stomp ends the lesson: the row after the last button
    assert KINDS[labels.kind[jump, 0]] == "stomp"
    assert labels.frames[jump, 0] == kill - jump
    by_slot = slot_labels(labels, stomp.slot_match, 6)
    seen = by_slot["labelled"][:, 0]
    assert (by_slot["frames"][seen, 0] == labels.frames[seen, 0]).all()


def test_the_executed_stomps_outcome_is_measured_where_the_walker_died(stomp):
    outcomes = executed_outcomes(stomp, lt.END_REASONS.index("no_progress"))
    last = outcomes[-1]
    assert VERBS[int(last[COLUMN["verb"]])] == "stomp"
    assert last[COLUMN["safe"]] == 1.0 and last[COLUMN["won"]] == 1.0
    frames = len(stomp.buttons)
    start = int(stomp.decision_frames[-1])
    assert last[COLUMN["frames"]] == frames - start
    truth = stomp.truth
    walker = truth["enemies"][frames, 0].astype(float)
    feet = truth["mario"][start].astype(float)
    assert last[COLUMN["object_dx"]] == pytest.approx(
        (walker[0] + walker[2]) / 2 - (feet[0] + feet[2]) / 2
    )
    assert last[COLUMN["object_dy"]] == pytest.approx(walker[1] - feet[3])
    assert np.isfinite(outcomes).all()
