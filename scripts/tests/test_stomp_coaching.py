"""Stomp contact geometry, duration decisions, and single-attempt diagnostics."""

from dataclasses import replace

import pygame
import pytest
import torch

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.stomp import stomp_collision_geometry


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def geometry(x=120, y=192, vy=4, enemy_x=110):
    return stomp_collision_geometry(
        pygame.Rect(x, y, 14, 16), pygame.Rect(enemy_x, 206, 12, 14), vy
    )


def legacy_sample():
    sample = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=2, sample_index=0, family="stomp_mount", difficulty="easy"
    )
    scenario = dict(sample.scenario)
    # From a standstill at x=40, NES holds of 4-16 frames land on a static
    # enemy at x=58; 1-2 frame hops run into its side and 18+ frames clear it.
    scenario["enemies"] = [[58, 206, 58, 58, 0.0]]
    return replace(sample, scenario=scenario)


def test_new_motion_phases_reverse_before_the_release_window_closes_and_stay_reachable():
    directions = set()
    turns = set()
    for difficulty in ("medium", "hard"):
        for seed in range(12):
            sample = sample_block_smb_monte_carlo_scenario(
                split="validation",
                seed=seed,
                sample_index=0,
                family="stomp_mount",
                difficulty=difficulty,
            )
            assert sample.reachability["reachable"]
            env = MarioScenarioEnv()
            try:
                env.reset(scenario=sample.scenario)
                initial = env.enemies[0]["direction"]
                directions.add(initial)
                for frame in range(1, 17):
                    env.step(2)
                    if env.enemies[0]["direction"] != initial:
                        turns.add(frame)
                        break
                else:
                    pytest.fail("enemy never reversed within the controllable hold window")
            finally:
                env.close()
    assert directions == {-1, 1}
    assert len(turns) >= 3
