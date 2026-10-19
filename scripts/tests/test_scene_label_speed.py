"""The simulator's scene labels, made faster, are exactly what they were
(user, 2026-10-19: the teacher's trials build a scene every simulated
frame)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random

import numpy as np
import pygame
import pytest

from retroagi.stages.block_smb.env import MarioScenarioEnv, _Canvas
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

FAMILIES = ("mixed_section", "piranha_avoidance", "monster_retreat", "moving_bridge", "enemy_hop")


def _two_drawings(env):
    """render_labels and render_instances as they were: one drawing each."""
    env._draw(_Canvas(env._label_screen, "types"))
    types = np.ascontiguousarray(pygame.surfarray.array_red(env._label_screen).T)
    env._draw(_Canvas(env._label_screen, "instances"))
    red = pygame.surfarray.array_red(env._label_screen).T.astype(np.int32)
    green = pygame.surfarray.array_green(env._label_screen).T.astype(np.int32)
    return types, red + (green << 8) - 1


@pytest.mark.timeout(600)
@pytest.mark.parametrize("family", FAMILIES)
def test_one_drawing_gives_the_types_and_objects_two_did(family):
    scenario = sample_block_smb_monte_carlo_scenario(
        split="train", seed=0, sample_index=3, family=family, difficulty="medium",
        calibrate_deadline=False,
    ).scenario
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    rng = random.Random(0)
    try:
        for _ in range(240):
            types, instances = env._render_types_and_instances()
            old_types, old_instances = _two_drawings(env)
            assert types.dtype == old_types.dtype and np.array_equal(types, old_types)
            assert np.array_equal(instances, old_instances)
            _, _, terminated, truncated, _ = env.step(rng.choice((0, 1, 1, 2, 2, 3, 5)))
            if terminated or truncated:
                break
    finally:
        env.close()


@pytest.mark.timeout(600)
@pytest.mark.parametrize("family", FAMILIES)
def test_a_structure_found_before_is_the_structure_found_now(family):
    import retroagi.core.smb_scene_labels as sl
    from retroagi.core.smb_pixel_types import VISIBLE_COLUMNS, VISIBLE_ROWS

    scenario = sample_block_smb_monte_carlo_scenario(
        split="train", seed=0, sample_index=5, family=family, difficulty="hard",
        calibrate_deadline=False,
    ).scenario
    env = MarioScenarioEnv()
    env.reset(scenario=scenario, seed=0)
    rng = random.Random(1)
    sl._STRUCTURES.clear()
    (top, bottom), (left, right) = VISIBLE_ROWS, VISIBLE_COLUMNS
    try:
        for _ in range(240):
            types = env.scene_labels().types
            window = np.asarray(types)[top:bottom, left:right]
            fresh = sl._structure_of(
                sl._terrain_behind_objects(window),
                np.where(np.isin(window, sl._BLOCK_IDS), window, 0).astype(np.uint8),
            )
            assert sl._window_structure(window) == fresh
            assert sl._window_structure(window) == fresh  # read back
            _, _, terminated, truncated, _ = env.step(rng.choice((0, 1, 1, 2, 2, 3, 5)))
            if terminated or truncated:
                break
        assert len(sl._STRUCTURES) >= 1
    finally:
        env.close()
