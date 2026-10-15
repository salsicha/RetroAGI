"""Enemy lessons start after frames the agent only watches: its tracks (and
target tracker) see the enemies move before its first decision."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest
import torch

from retroagi.core.layered_policy import LayeredSMBPolicy
from retroagi.core.smb_scene_labels import EnemyView, MarioView, SceneObservation, Surface
from retroagi.core.tokens import SkillToken
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
    agents = SMBAgents(
        VisionObserver(SceneEcho(scene())),
        LayeredSMBPolicy().eval(),
        "cpu",
        carry_out=lambda number, action: SkillToken("hold", 0, 0),
    )
    agents.reset(0, watch=WATCH_FRAMES)
    screen = [np.zeros((240, 256, 3), np.uint8)]
    steps = [agents.act(screen, [0])[0] for _ in range(WATCH_FRAMES + 1)]
    assert all(step.decision is None and step.button == 0 for step in steps[:WATCH_FRAMES])
    assert steps[WATCH_FRAMES].decision is not None  # the first decision follows the watch
