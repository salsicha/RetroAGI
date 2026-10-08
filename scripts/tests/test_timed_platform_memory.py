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
from retroagi.stages.block_smb.layered_train import action_memory, platform_prediction_loss


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


def test_forecasts_report_explicit_times_and_start_uncertain():
    policy = LayeredSMBPolicy()
    result = policy.memory.platforms(torch.zeros(2, policy.memory.width))
    assert result["frames"] == (16, 32, 64)
    assert result["displacement"].shape == (2, 3, 3, 2)
    assert (result["sigma"] >= 31).all()
    assert (result["visible"].sigmoid() < 0.8).all()


def test_future_targets_follow_identity_when_platform_slots_swap_and_mask_episode_end():
    policy = LayeredSMBPolicy()
    observations = np.zeros((33, 3, 3), np.float32)
    observations[0, 0] = (7, 100, 200)
    observations[16, 2] = (7, 120, 200)  # Same object, a different visual slot.
    observations[32, 1] = (7, 140, 200)
    episode = SimpleNamespace(frames=33, platform_observations=observations)
    states = torch.zeros(1, 1, policy.memory.width, requires_grad=True)
    loss, stats = platform_prediction_loss(policy, states, [[0]], [episode])
    assert stats["platform_targets"] == 2  # No fictitious 64-frame target.
    assert stats["platform_error_pixels"] == pytest.approx(15)
    loss.backward()
    assert policy.memory.platform_prediction.bias.grad.abs().sum() > 0
    # Future labels do not enter the recurrent state or its forecast.
    assert states.detach().count_nonzero() == 0


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


def test_confident_distant_forecast_requests_transfer_reconsideration_near_edge():
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
    assert statuses[0] == "transfer_recheck"
    assert statuses[1] == "running"


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
