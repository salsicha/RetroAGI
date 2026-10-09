"""Timed forecasts share scene memory and remain causal across runtime and replay."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from retroagi.core.layered_policy import MEMORY_INTERVAL, LayeredSMBPolicy
from retroagi.core.smb_agent import SMBAgents
from retroagi.core.smb_executor import ActionPlan
from retroagi.core.smb_scene_labels import MarioView, SceneObservation, Surface
from retroagi.core.smb_trajectory import Track, VisualTracks
from retroagi.stages.block_smb.layered_train import (
    action_end_targets,
    action_memory,
    endpoint_prediction_losses,
)


def picture(x=100, platform=150):
    return SceneObservation(
        MarioView((x, 208, x + 10, 220), True, "ground", True),
        surfaces=(Surface(8, 248, 220, False),),
        moving_platforms=((platform, 200, platform + 30, 210),),
    )


def test_runtime_and_replay_memory_match_between_decisions_and_after_reset():
    torch.manual_seed(3)
    pictures = [picture(platform=150 + t) for t in range(13)]

    class Observer:
        def observe(self, screens):
            return [pictures.pop(0)]

    policy = LayeredSMBPolicy().eval()
    # Ensure parity really tests timing, rather than a zero-initialized time input.
    torch.nn.init.normal_(policy.memory.timing.weight, std=0.1)
    agents = SMBAgents(Observer(), policy, "cpu")
    steps, states, cameras = [], [], []
    for _ in range(13):
        step = agents.act([None], [0], given=lambda *_: {"action": [ActionPlan(0, 6)]})[0]
        steps.append(step)
        states.append(agents.copies[0].hidden.clone())
        cameras.append(agents.copies[0].spatial.camera_position)
    starts = [i for i, step in enumerate(steps) if step.decision is not None]
    a, b, c = [torch.tensor(np.stack([s.rows[k] for s in steps])).unsqueeze(0) for k in range(3)]
    d = {"episode": torch.zeros(len(starts), dtype=torch.long), "frame": torch.tensor(starts)}
    with torch.no_grad():
        replay = action_memory(
            policy,
            a.long(),
            b.long(),
            c.float(),
            d,
            [SimpleNamespace(frames=13, camera_positions=np.array(cameras))],
        )
    assert torch.allclose(replay, torch.stack([states[t] for t in starts]), atol=1e-5)
    assert not torch.equal(states[MEMORY_INTERVAL - 1], states[MEMORY_INTERVAL])
    agents.reset(0)
    assert agents.copies[0].hidden.count_nonzero() == 0
    assert not agents.copies[0].spatial.tracks.tracks


def test_forecasts_report_learned_action_end_time_and_start_uncertain():
    policy = LayeredSMBPolicy()
    hidden = torch.zeros(2, policy.memory.width)
    result = policy.memory.platforms(hidden)
    assert torch.allclose(result["frames"], torch.full((2,), 32.0))
    assert result["displacement"].shape == (2, 3, 2)
    assert (result["sigma"] >= 31).all()
    assert (result["visible"].sigmoid() < 0.8).all()
    # The same readout can represent arbitrary durations, including over 64.
    for frames in (1, 2, 7, 65, 168, 257):
        with torch.no_grad():
            policy.memory.end_duration.bias[0] = (
                -30 if frames == 1 else torch.log(torch.expm1(torch.tensor((frames - 1) / 32)))
            )
        forecast = policy.memory.platforms(hidden)
        assert forecast["frames"].tolist() == pytest.approx([frames, frames], abs=1e-4)


def endpoint_episode(frames=241):
    from retroagi.core.smb_observer import SEQ_LEN_C

    return SimpleNamespace(
        frames=frames,
        decision_frames=np.array([0, 7, 72, 240]),
        src_c=np.zeros((frames, SEQ_LEN_C), np.float32),
        platform_observations=np.zeros((frames, 3, 3), np.float32),
        final_src_c=None,
        final_platform_observations=None,
    )


def test_targets_are_next_action_end_including_nonstandard_and_long_durations():
    episode = endpoint_episode()
    ticks = [[0, 4, 7, 8, 72, 80, 240]]
    pairs = action_end_targets(ticks, [episode])
    assert [(f, end, end - f) for _, _, f, end in pairs] == [
        (0, 7, 7),
        (4, 7, 3),
        (7, 72, 65),
        (8, 72, 64),
        (72, 240, 168),
        (80, 240, 160),
    ]
    # Budget truncation is censored, but a recorded terminal endpoint trains
    # the final action even though no next decision is made.
    episode.final_src_c = episode.src_c[-1].copy()
    assert action_end_targets(ticks, [episode])[-1] == (0, 6, 240, 241)


def test_future_targets_follow_identity_at_actual_endpoint_not_fixed_horizon():
    policy = LayeredSMBPolicy()
    episode = endpoint_episode()
    episode.platform_observations[0, 0] = (7, 100, 200)
    episode.platform_observations[7, 2] = (7, 122, 200)  # Next action end, another slot.
    episode.platform_observations[16, 0] = (7, 200, 200)  # Must NOT become the target.
    states = torch.zeros(1, 1, policy.memory.width, requires_grad=True)
    losses, stats = endpoint_prediction_losses(policy, states, [[0]], [episode], 1.0)
    assert stats["platform_targets"] == 1
    assert stats["platform_error_pixels"] == pytest.approx(11)
    assert stats["max_end_duration_frames"] == 7
    sum(losses.values()).backward()
    assert policy.memory.platform_prediction.bias.grad.abs().sum() > 0
    assert policy.memory.end_duration.bias.grad.abs().sum() > 0
    assert states.detach().count_nonzero() == 0


def test_final_platform_endpoint_and_scene_are_supervised_together():
    policy = LayeredSMBPolicy()
    episode = endpoint_episode(frames=80)
    episode.decision_frames = np.array([0])
    episode.platform_observations[0, 0] = (1, 50, 200)
    episode.final_src_c = np.ones_like(episode.src_c[0])
    episode.final_platform_observations = np.array([[1, 90, 200], [0, 0, 0], [0, 0, 0]])
    states = torch.zeros(1, 1, policy.memory.width)
    losses, stats = endpoint_prediction_losses(policy, states, [[0]], [episode], 1.0)
    assert stats["max_end_duration_frames"] == 80
    assert stats["platform_error_pixels"] == 20
    assert set(losses) == {"expectation", "end_duration", "platform_prediction"}


def test_distant_forecast_expires_rebases_with_scroll_and_does_not_jump_identity():
    track = Track(
        (100, 200, 130, 210), "platform", [(1, 0)], identity=8, distant=[(16, 20, 0, 2, 0.99)]
    )
    tracks = VisualTracks([track], next_identity=9)
    tracks.observe(picture(platform=97), scroll=4)  # World displacement +1, screen -3.
    assert tracks.tracks[0] is track
    horizon, future, uncertainty = track.distant_forecast(9)
    assert horizon == 15 and future[0] == 116 and uncertainty == 2
    track.forecast_age = 16
    assert track.distant_forecast(1) is None
    track.forecast_age = 0
    track.distant = [(16, 20, 0, 20, 0.99)]
    assert track.distant_forecast(9) is None
    tracks.observe(picture(platform=180), scroll=0)
    assert tracks.tracks[0] is not track and not tracks.tracks[0].distant


def test_old_checkpoint_migrates_only_new_memory_readouts(tmp_path):
    from dataclasses import asdict

    from retroagi.core.smb_observer import observation_layout
    from retroagi.core.tokens import token_layout
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    policy = LayeredSMBPolicy()
    old = {
        k: v
        for k, v in policy.state_dict().items()
        if not k.startswith(("memory.platform_prediction.", "memory.timing."))
    }
    path = tmp_path / "legacy.pt"
    torch.save(
        dict(
            settings=asdict(policy.settings),
            state_dict=old,
            observation_layout=observation_layout(),
            token_layout=token_layout(),
            trained_layers=["skill"],
        ),
        path,
    )
    loaded, metadata = load_layered_checkpoint(path)
    assert metadata["trained_layers"] == []
    for name, weight in old.items():
        assert torch.equal(loaded.state_dict()[name], weight)


def test_distant_forecast_does_not_veto_a_requested_transfer():
    from collections import deque

    from retroagi.core.smb_ground_control import GroundMove
    from retroagi.core.smb_spatial_feedback import SpatialFeedback

    scene = SceneObservation(
        MarioView((185, 208, 195, 220), True, "moving_platform", True),
        surfaces=(Surface(220, 248, 220, False),),
        moving_platforms=((100, 220, 200, 232),),
    )
    statuses = []
    for sigma in (2, 32):
        track = Track(
            scene.moving_platforms[0],
            "platform",
            [(-1, 0)] * 4,
            distant=[(32, -30, 0, sigma, 0.99)],
        )
        feedback = SpatialFeedback(tracks=VisualTracks([track]), velocities=deque([0, 0]))
        move = GroundMove(feedback, 40, 0, 220)
        move.press(scene)
        statuses.append(move.status)
    assert statuses == ["running", "running"]


def test_boarding_request_moves_onto_platform_instead_of_braking_back_to_shore():
    from collections import deque

    from retroagi.core.smb_ground_control import GroundMove
    from retroagi.core.smb_physics import NESPlayerMotion
    from retroagi.core.smb_spatial_feedback import SpatialFeedback

    scene = SceneObservation(
        MarioView((81, 208, 91, 220), True, "moving_platform", True),
        surfaces=(Surface(8, 85, 220, False),),
        moving_platforms=((90, 220, 190, 232),),
    )
    track = Track(scene.moving_platforms[0], "platform", [(-2.1, 0)] * 4)
    feedback = SpatialFeedback(
        tracks=VisualTracks([track]),
        velocities=deque([0, 0]),
        motion=NESPlayerMotion(x_speed=0, moving=1),
    )
    move = GroundMove(feedback, 14, 0, 220)
    assert move.press(scene) == 1
    assert not move.done  # Partial contact is not completion.
    assert move.target is track


def test_fixed_horizon_checkpoint_discards_only_incompatible_readouts(tmp_path):
    from dataclasses import asdict

    from retroagi.core.smb_observer import observation_layout
    from retroagi.core.tokens import token_layout
    from retroagi.stages.block_smb.layered_train import load_layered_checkpoint

    policy = LayeredSMBPolicy()
    old = {
        k: v
        for k, v in policy.state_dict().items()
        if not k.startswith(("memory.end_duration.", "memory.end_time_embedding."))
    }
    old["memory.platform_prediction.weight"] = torch.ones(45, policy.memory.width)
    old["memory.platform_prediction.bias"] = torch.ones(45)
    path = tmp_path / "fixed.pt"
    torch.save(
        dict(
            settings=asdict(policy.settings),
            state_dict=old,
            memory_version=2,
            observation_layout=observation_layout(),
            token_layout=token_layout(),
            trained_layers=["skill"],
        ),
        path,
    )
    loaded, metadata = load_layered_checkpoint(path)
    assert "action_endpoint_requires_requalification" in metadata["load_migrations"]
    assert metadata["trained_layers"] == []
    for name, value in old.items():
        if not name.startswith("memory.platform_prediction."):
            assert torch.equal(loaded.state_dict()[name], value)
    assert loaded.memory.platform_prediction.weight.shape[0] == 15


def test_duration_audit_counts_completed_actions_and_censors_final_cutoff():
    from retroagi.stages.block_smb.layered_train import action_duration_summary

    episode = endpoint_episode()
    assert action_duration_summary([episode]) == {
        "completed": 3,
        "max_frames": 168,
        "over_64_frames": 2,
    }
    episode.final_src_c = episode.src_c[-1]
    assert action_duration_summary([episode])["completed"] == 4


def test_decision_layers_receive_the_predicted_endpoint_time():
    from retroagi.core.layered_policy import MemoryState

    policy = LayeredSMBPolicy().eval()
    hidden = torch.zeros(1, policy.memory.width)
    with torch.no_grad():
        policy.memory.end_time_embedding.weight.fill_(1.0)
        first_numbers, first = policy.expect(MemoryState(hidden, hidden))
        policy.memory.end_duration.bias[0] += 2
        later_numbers, later = policy.expect(MemoryState(hidden, hidden))
    assert torch.equal(first_numbers, later_numbers)  # Same predicted scene.
    assert not torch.equal(first[0][:, 0], later[0][:, 0])  # Different time reaches consumers.
    assert torch.equal(first[0][:, 1:], later[0][:, 1:])
