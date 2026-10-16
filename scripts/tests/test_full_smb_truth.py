"""Full SMB's truth from the emulator's memory and recorded playthroughs, in
the arrays the collision trackers' and the world model's labels are made
from (user, 2026-10-16: the same labels for Full SMB as for Block SMB)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
import torch

from retroagi.core.actions import SMBAction
from retroagi.core.collision_labels import KINDS, action_ends, collision_labels
from retroagi.stages.full_smb.play import FullSMBGame
from retroagi.stages.full_smb.truth import FullTruthRecorder, true_frame

pytestmark = pytest.mark.timeout(900)


def scripted(frames):
    """1-1 played by a fixed script (run right, jump every 60 frames): the
    truth of every frame and of the state after the last."""
    game = FullSMBGame("Level1-1")
    try:
        game.reset()
        recorder = FullTruthRecorder()
        ids = []
        for t in range(frames):
            ids.append(recorder.add(true_frame(game.env.get_ram(), game.start)))
            game.step(SMBAction.RIGHT if t % 60 < 50 else SMBAction.RIGHT_JUMP)
        recorder.add(true_frame(game.env.get_ram(), game.start))
        return recorder, ids
    finally:
        game.close()


@pytest.fixture(scope="module")
def first_goomba():
    return scripted(320)


def test_the_games_own_boxes_are_read_in_world_pixels(first_goomba):
    recorder, _ = first_goomba
    frame = recorder.rows[104]
    mario, goomba = frame["mario"], frame["slots"][0]
    assert goomba["kind"] == 0x06 and goomba["active"] and not goomba["defeated"]
    # Small Mario's body is 10 x 12, a Goomba's 10 x 6 inside its 16 x 16 picture.
    assert (mario[2] - mario[0], mario[3] - mario[1]) == (10, 12)
    box, drawn = goomba["box"], goomba["drawn"]
    assert (box[2] - box[0], box[3] - box[1]) == (10, 6)
    assert drawn[0] <= box[0] and box[2] <= drawn[2] and drawn[1] <= box[1] <= box[3] <= drawn[3]
    assert frame["level_x"] == mario[0] - 3 and frame["time"] > 0


def test_a_refilled_slot_is_a_new_enemy(first_goomba):
    recorder, ids = first_goomba
    assert ids[104][0] == 0 and ids[300][0] == 1
    arrays = recorder.arrays()
    assert arrays["enemies"].shape[1] == 2 and arrays["stompable"].all()
    # The first is stomped: alive, then defeated; the score rises by 100.
    states = arrays["enemy_state"][:, 0].tolist()
    stomp = states.index(2)
    assert set(states[100:stomp]) == {1} and arrays["score"][stomp + 8] == 100


def test_full_smb_labels_count_down_to_the_stomp(first_goomba):
    recorder, _ = first_goomba
    arrays = recorder.arrays()
    frames = len(recorder.rows) - 1
    labels = collision_labels(arrays, action_ends([0], frames))
    stomp = arrays["enemy_state"][:, 0].tolist().index(2)
    for t in (stomp - 40, stomp - 20, stomp - 2):
        assert KINDS[labels.kind[t, 0]] == "stomp"
        # The touch is the frame the stomp shows (or the one before).
        assert abs(labels.frames[t, 0] - (stomp - t)) <= 1


def test_a_recorded_playthrough_has_what_the_trainers_read():
    from retroagi.core.layered_policy import LayeredSMBPolicy
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver
    from retroagi.core.target_tracker import CollisionTracker
    from retroagi.stages.block_smb.tracker_train import track_sequences
    from retroagi.stages.block_smb.world_model_train import episode_data
    from retroagi.stages.full_smb.practice import play_recorded
    from retroagi.stages.full_smb.vision import DEFAULT_FULL_VIT_CHECKPOINT, load_full_vit_checkpoint

    torch.manual_seed(0)
    vision = load_full_vit_checkpoint(DEFAULT_FULL_VIT_CHECKPOINT, device="cpu").model
    # Untrained networks: the recording, not the play, is tested.
    agents = SMBAgents(
        VisionObserver(vision),
        LayeredSMBPolicy().eval(),
        "cpu",
        tracker=CollisionTracker(8),
        world_model=True,
    )
    played = play_recorded(agents, "Level1-1", frames=150)
    frames = len(played.sprite_kinematics)
    assert len(played.truth["mario"]) == frames + 1 and len(played.ended) == frames
    assert played.slot_match.shape == (frames, 9) and len(played.decision_frames) > 0
    assert played.commands.shape == (len(played.decision_frames), 6)
    assert len(played.rows[0]) == len(played.decision_frames)
    assert not np.isnan(played.predicted).any()
    track_sequences(played)  # the trackers' trainer reads it
    data = episode_data(played)  # and the world model's
    assert len(data["executed"]) == len(played.decision_frames)
