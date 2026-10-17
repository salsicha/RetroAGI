"""The collision trackers: per enemy and frame, whether, when, where and how
Mario and it will touch, from the executor's kinematics, what vision shows
and the action in progress (user, 2026-10-15/16)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
import torch

from retroagi.core.action_tokens import VERBS, pointer_index
from retroagi.core.collision_labels import action_ends, collision_labels
from retroagi.core.layered_policy import POINTER_START, LayeredSMBPolicy
from retroagi.core.smb_kinematics import (
    SPRITE_SLOTS,
    SPRITE_STATE,
    enemy_context,
    kinematics,
    slot_identities,
)
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.core.target_tracker import (
    BINS,
    CLASSES,
    FEATURES,
    NONE,
    OBSERVATION_WIDTH,
    ActionState,
    CollisionTracker,
    CollisionTrackers,
    action_state,
    allowed_classes,
    collision_loss,
    frame_features,
    record_features,
)
from retroagi.stages.block_smb import layered_train as lt
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.teacher_replay import reference_vision
from retroagi.stages.block_smb.tracker_train import _baseline_hit_frames, batch_loss
from retroagi.stages.block_smb.truth_recording import TruthRecorder

pytestmark = pytest.mark.timeout(900)


def test_the_features_the_agent_tracked_with_are_the_recordings():
    # The skill plays its own actions (the teacher carrying them out); the
    # trackers' inputs at play and those rebuilt from the recording agree.
    seen = []
    original = CollisionTrackers.observe

    def observe(self, features, identities):
        seen.append((features.copy(), np.asarray(identities).copy()))
        return original(self, features, identities)

    CollisionTrackers.observe = observe
    try:
        task = lt.EpisodeTask(
            index=0,
            family="enemy_hop",
            split="validation",
            seed=0,
            sample_index=0,
            difficulty="easy",
            teacher_share=0.0,
            label=False,
            record_truth=True,
        )
        torch.manual_seed(0)
        policy = LayeredSMBPolicy().eval()
        # The untrained skill runs: a wait the teacher would not make lasts a
        # frame, and each frame's decision asks the teacher again (seconds).
        with torch.no_grad():
            policy.skill.heads["verb"].weight.zero_()
            policy.skill.heads["verb"].bias.fill_(-20)
            policy.skill.heads["verb"].bias[VERBS.index("run")] = 20
        record = lt.play_episodes(
            reference_vision()[0],
            policy,
            "skill",
            [task],
            "cpu",
            1,
            tracker=CollisionTracker(16),
        )[0]
    finally:
        CollisionTrackers.observe = original
    offline = record_features(record)
    assert len(seen) == len(offline)
    decisions = set(int(f) for f in record.decision_frames)
    observation = slice(0, OBSERVATION_WIDTH)
    for t, (features, identities) in enumerate(seen):
        assert (identities[:6] == record.slot_identity[t, :6]).all()
        # The GRU's inputs agree every frame; the action's from the frame
        # after it began (at play it is set after the frame's step).
        assert np.allclose(features[:, observation], offline[t][:, observation], atol=1e-3)
        if t not in decisions:
            assert np.allclose(features, offline[t], atol=1e-3)


def test_unknown_tails_allow_every_later_contact_and_none():
    allowed = allowed_classes([0, 5, 0], [0, 2, 0], [0, 0, 11])
    assert allowed[0].tolist() == [c == NONE for c in range(CLASSES)]
    assert allowed[1].sum() == 1 and allowed[1, BINS + 2]  # a hit 5 frames on
    # Unknown from offset 11: contacts at 11 and later (bins 5..) and none.
    assert not allowed[2, :5].any() and allowed[2, 5:BINS].all() and allowed[2, NONE]
    logits = torch.zeros(3, CLASSES, requires_grad=True)
    out = {
        "logits": logits,
        "place": torch.zeros(3, 2),
        "log_sigma": torch.zeros(3, 2),
    }
    loss, _ = collision_loss(
        out,
        torch.as_tensor(allowed),
        torch.zeros(3, 2),
        torch.tensor([False, True, False]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    # Learning raises what each label allows.
    assert logits.grad[0, NONE] < 0 and logits.grad[1, BINS + 2] < 0
    assert logits.grad[2, :5].gt(0).all() and logits.grad[2, NONE] < 0


def walker_sequence(speed, start=110, frames=260):
    """Mario stands while a walker comes at him at ``speed``: the trackers'
    inputs per frame (the walker's enemy slot) and its labels."""
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 400,
                "mario": [40, 204],
                "platforms": [[0, 220, 400, 20]],
                "enemies": [[start, 206, 0, 300, speed, -1]],
                "goal": [380, 200, 16, 20],
            }
        )
        for _ in range(4):
            env.step(0)
        spatial, recorder, features = SpatialFeedback(), TruthRecorder(), []
        for t in range(frames):
            scene = scene_from_labels(env.scene_labels())
            spatial.observe(scene)
            sprites, mario = kinematics(scene, spatial)
            identities = slot_identities(scene, spatial)
            features.append(
                frame_features(
                    sprites, mario, enemy_context(scene), identities, (0.0, 204.0), t, ActionState()
                )[0]
            )
            recorder.add(env.true_frame())
            spatial.executed(0, scene)
            _, _, done, truncated, _ = env.step(0)
            if done or truncated:
                break
        recorder.add(env.true_frame())
    finally:
        env.close()
    features = np.asarray(features, np.float32)
    labels = collision_labels(recorder.arrays(), action_ends([0], len(features)))
    keep = features[:, 0] > 0
    return {
        "observations": features[keep, :OBSERVATION_WIDTH],
        "features": features[keep],
        "allowed": allowed_classes(
            labels.frames[keep, 0], labels.kind[keep, 0], labels.censor[keep, 0]
        ),
        "place": labels.place[keep, 0],
        "seen": labels.frames[keep, 0] > 0,
        "labelled": np.ones(keep.sum(), bool),
        "frames": labels.frames[keep, 0],
    }


def test_a_small_tracker_learns_when_an_approaching_walker_arrives():
    torch.manual_seed(0)
    train = [walker_sequence(v) for v in (0.3, 0.45, 0.6, 0.75, 0.9, 1.05, 1.2)]
    held = walker_sequence(0.68)
    model = CollisionTracker(32)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(300):
        loss, _ = batch_loss(model, train, "cpu")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        out = model.answer(
            model.observe(torch.as_tensor(held["observations"])[None]),
            torch.as_tensor(held["features"])[None],
        )
    probability = torch.softmax(out["logits"][0], -1).numpy()
    contact = probability[:, :NONE].reshape(len(probability), 2, BINS).sum(1)
    predicted = np.argmax(contact, -1) * 2 + 1.5
    frames = held["frames"]
    seen = frames > 0
    error = np.abs(predicted[seen] - frames[seen])
    baseline = np.abs(_baseline_hit_frames(held["features"])[seen] - frames[seen])
    assert np.median(error) <= 4
    assert np.median(error) <= np.median(baseline) + 2


def test_an_enemy_aimed_actions_destination_is_the_enemy_seen_not_the_command():
    sprites = np.zeros((SPRITE_SLOTS, len(SPRITE_STATE)), np.float32)
    sprites[1] = (1.0, 40.0, -6.0, -0.5, 0.0, 1.0)
    identities = np.zeros(SPRITE_SLOTS, np.int32)
    identities[1] = 7
    stomp = VERBS.index("jump")  # a jump onto the enemy
    pointer = pointer_index(("enemies", 1)) - POINTER_START
    one = action_state(stomp, pointer, [1, 31, -10, 0, 0, 0], sprites, identities, (0, 0), 3)
    other = action_state(stomp, pointer, [1, 52, -10, 0, 0, 0], sprites, identities, (0, 0), 3)
    assert one == other and one.destination == (40.0, -6.0) and one.target == 7
    # A wait's length and threat (the teacher's truth) are never inputs.
    wait = VERBS.index("wait")
    short = action_state(wait, -1, [2, 0, 0, 0, 3, 0], sprites, identities, (0, 0), 3)
    long = action_state(wait, -1, [2, 0, 0, 0, 90, 1], sprites, identities, (0, 0), 3)
    assert short == long and short.destination == (0.0, 0.0)
    features = frame_features(
        sprites, np.zeros(3), np.zeros((6, 8)), identities, (10.0, 0.0), 5, one
    )
    assert features.shape == (6, FEATURES) and features[1, -1] == 1.0 and not features[0].any()


def test_runtime_estimates_and_forgetting():
    torch.manual_seed(0)
    trackers = CollisionTrackers(CollisionTracker(16))
    features = np.zeros((6, FEATURES), np.float32)
    features[0, 0] = 1.0
    identities = [5, 0, 0, 0, 0, 0]
    trackers.observe(features, identities)
    estimate = trackers.estimate(5)
    assert estimate is not None and trackers.estimate(6) is None
    within = [estimate.hit_within(f) for f in (1, 8, 32, 96)]
    assert within == sorted(within) and 0.0 <= within[-1] <= 1.0
    assert estimate.contact < 0.5  # untrained: no contact
    for _ in range(97):
        trackers.observe(np.zeros((6, FEATURES), np.float32), [0] * 6)
    assert trackers.estimate(5) is None
