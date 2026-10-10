"""Enemy families: prefilled history, the memory's enemy forecast, and its use.

The skill decides a stomp or a bypass from where the enemy will be when the
next action ends. One picture cannot show how an enemy moves, so these layouts
start with pre-episode frames the agent watches before its first decision.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy, forecast_features
from retroagi.core.smb_scene_labels import EnemyView, MarioView, SceneObservation, Surface
from retroagi.core.smb_trajectory import Track, VisualTracks, plan_flight
from retroagi.core.tokens import SkillToken, encode_tactic, tactic_token
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.skill_families import WATCH_FRAMES


def sample(family, index=0, difficulty="medium"):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family=family, difficulty=difficulty
    ).scenario


@pytest.mark.parametrize("family", ["enemy_stomp", "skill_enemy_bypass", "skill_enemy_bypass_back"])
def test_enemy_families_start_after_pre_episode_frames_the_agent_watches(family):
    scenario = sample(family)
    assert scenario["watch_frames"] == WATCH_FRAMES
    enemy = scenario["enemies"][0]
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        # The world ran while Mario stood; the episode starts after it.
        assert len(env.watched_screens) == len(env.watched_labels) == WATCH_FRAMES
        assert env.steps == 0 and not env._goal_credited
        moved = env.enemies[0]["x"] - enemy[0]
        assert moved == pytest.approx(enemy[4] * enemy[5] * WATCH_FRAMES, abs=1.0)
    finally:
        env.close()


class SceneEcho:
    """A stand-in vision transformer: every screen shows the same fixed scene."""

    def __init__(self, picture):
        self.picture = picture

    def eval(self):
        return self

    def scene(self, screens):
        return [self.picture] * len(screens)


def scene(enemy_x=150):
    return SceneObservation(
        mario=MarioView(
            box=(100, 196, 110, 208), facing_right=True, support="ground", on_something=True
        ),
        enemies=(EnemyView((enemy_x, 194, enemy_x + 12, 208), "walker"),),
        surfaces=(Surface(8, 248, 208, False),),
    )


def test_the_agent_decides_nothing_while_it_watches():
    from retroagi.core.smb_agent import SMBAgents
    from retroagi.core.smb_observer import VisionObserver

    torch.manual_seed(0)
    agents = SMBAgents(VisionObserver(SceneEcho(scene())), LayeredSMBPolicy().eval(), "cpu")
    agents.reset(0, watch=WATCH_FRAMES)
    screen = [np.zeros((240, 256, 3), np.uint8)]
    steps = [agents.act(screen, [0])[0] for _ in range(WATCH_FRAMES + 1)]
    assert all(step.decision is None and step.button == 0 for step in steps[:WATCH_FRAMES])
    assert steps[WATCH_FRAMES].decision is not None  # the first decision follows the watch


def test_an_untrained_enemy_forecast_is_never_trusted():
    policy = LayeredSMBPolicy()
    forecast = policy.memory.enemies(torch.zeros(1, policy.memory.width))
    assert forecast["displacement"].shape == (1, 6, 2)
    track = Track((150, 194, 162, 208), "walker")
    track.distant = [
        (
            float(forecast["frames"][0]),
            *forecast["displacement"][0][0].tolist(),
            float(forecast["sigma"][0][0].max()),
            float(forecast["visible"][0][0].sigmoid()),
        )
    ]
    assert track.landing_box() == track.box  # wide uncertainty, unlikely visibility


def test_a_stomp_lands_where_the_memory_forecasts_the_walker_will_be():
    now = scene(enemy_x=150)
    tracks = VisualTracks()
    tracks.observe(now, 0)
    walker = tracks.tracks[0]
    stomp = SkillToken("jump", 51, -14)  # the walker's top, where it stands now
    still = plan_flight(now, tracks, 0.0, stomp)
    assert still.target is walker and still.goal[0] == pytest.approx(156)
    # Trusted forecast: 24 pixels to the right when the jump ends.
    walker.distant = [(30.0, 24.0, 0.0, 1.0, 0.95)]
    led = plan_flight(now, tracks, 0.0, stomp)
    assert led.target is walker and led.goal[0] == pytest.approx(180)


def test_the_skill_reads_each_enemys_forecast_and_starts_without_its_influence():
    torch.manual_seed(0)
    policy = LayeredSMBPolicy().eval()
    from retroagi.core.smb_observer import policy_input

    inputs = policy_input(scene())
    rows = (inputs.src_a[None], inputs.src_b[None], inputs.src_c[None])
    hidden = torch.randn(1, policy.memory.width)
    with torch.no_grad():
        encoded = policy.encode_scene(rows)
        _, expected = policy.expect(SimpleNamespace(hidden=hidden))
        tactic = encode_tactic(tactic_token("advance"))[None]
        plain = policy.run_skill(encoded, expected, tactic)
        read = policy.run_skill(encoded, expected, tactic, memory=hidden)
        assert torch.equal(plain["x"], read["x"])  # zero influence until trained
        policy.skill.forecast.weight.normal_()
        changed = policy.run_skill(encoded, expected, tactic, memory=hidden)
    assert not torch.equal(plain["x"], changed["x"])
    features = forecast_features(policy.memory.enemies(hidden))
    assert features.shape == (1, 6, 5)


def test_the_forecast_learns_where_the_same_enemy_is_at_the_action_end():
    from retroagi.stages.block_smb.layered_train import _forecast_loss

    policy = LayeredSMBPolicy()
    frames = 10
    enemies = np.zeros((frames, 6, 3), np.float32)
    enemies[:, 0, 0] = 7  # one tracked identity, walking right 2 px a frame
    enemies[:, 0, 1] = 100 + 2 * np.arange(frames)
    enemies[:, 0, 2] = 194
    episode = SimpleNamespace(frames=frames, enemy_observations=enemies)
    hidden = torch.zeros(1, policy.memory.width, requires_grad=True)
    loss, stats = _forecast_loss(
        [(0, 0, 2, 8)],
        [episode],
        hidden,
        "enemy_observations",
        "final_enemy_observations",
        policy.memory.enemies,
    )
    # It moved 12 px right and none down; the untrained forecast says 0, 0.
    assert stats["targets"] == 1 and stats["error_pixels"] == pytest.approx((12 + 0) / 2)
    loss.backward()
    assert policy.memory.enemy_prediction.weight.grad is not None


def test_an_older_checkpoint_loads_with_the_new_forecast_at_zero_influence(tmp_path):
    from retroagi.stages.block_smb.layered_train import (
        LayeredTrainConfig,
        load_layered_checkpoint,
        save_layered_checkpoint,
    )

    torch.manual_seed(0)
    policy = LayeredSMBPolicy()
    path = tmp_path / "old.pt"
    save_layered_checkpoint(path, policy, LayeredTrainConfig(), ["skill"], [])
    saved = torch.load(path, weights_only=False)
    for name in list(saved["state_dict"]):
        if name.startswith(("memory.enemy_prediction.", "skill.forecast.")):
            del saved["state_dict"][name]
    torch.save(saved, path)
    loaded, checkpoint = load_layered_checkpoint(path)
    assert "enemy_forecast_zero_initialized" in checkpoint["load_migrations"]
    assert checkpoint["trained_layers"] == ["skill"]
    assert not loaded.skill.forecast.weight.any()
