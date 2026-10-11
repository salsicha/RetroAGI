"""A landing needs room to brake, and a local route lesson is one clean
maneuver."""

import copy
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from retroagi.core.smb_executor import SMBExecutor
from retroagi.core.smb_scene_labels import scene_from_labels
from retroagi.core.smb_spatial_feedback import SpatialFeedback
from retroagi.stages.block_smb.controller_teacher import stop_safe
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import _route_lesson_fault
from retroagi.stages.block_smb.tactic_schedule import segment


def flat(enemy_x, mario_x=100):
    return {
        "world_width": 400,
        "mario": [mario_x, 208],
        "platforms": [[0, 220, 400, 20]],
        "enemies": [[enemy_x, 206, enemy_x, enemy_x, 0.0]],
        "goal": [360, 200, 16, 20],
        "tactics": [segment("advance", 1)],
    }


def running_into(enemy_x):
    """Whether holding from a run at 1.5 pixels a frame, ``enemy_x`` ahead,
    is judged safe (the simulator left as it was)."""
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=flat(enemy_x))
        env.render = lambda: None
        env.mario["vx"], env.motion.x_speed = 1.5, 24
        spatial = SpatialFeedback(motion=copy.deepcopy(env.motion))
        spatial.observe(scene_from_labels(env.scene_labels()))
        before = snapshot_env_state(env)
        safe = stop_safe(env, spatial, SMBExecutor())
        assert snapshot_env_state(env) == before
        return safe
    finally:
        env.close()


def test_a_landing_beside_a_standing_enemy_needs_room_to_brake():
    assert not running_into(116)  # 4 pixels from Mario's front
    assert running_into(146)  # 34 pixels: room to stop
    assert running_into(300)  # nothing near: not even watched


def test_a_route_lesson_that_drops_off_its_route_is_a_detour():
    # From a ledge, a lower route platform across a gap; walking off the
    # ledge drops to the floor instead.
    scenario = {
        "world_width": 400,
        "mario": [100, 122],
        "platforms": [[0, 220, 400, 20], [80, 134, 40, 10], [200, 169, 40, 10]],
        "goal": [200, 149, 40, 20],
        "goal_requires_support": True,
        "tactics": [segment("advance", 1, route=[2])],
    }
    walk_then_climb = [1] * 60 + [2] * 20 + [1] * 60
    assert _route_lesson_fault(scenario, walk_then_climb) == "route_detour"


def test_a_descent_landing_on_the_ledge_edge_is_redrawn():
    from collections import Counter

    from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario

    # Its first draw: a 61-pixel gap from a 40-pixel ledge, crossed only by a
    # landing with Mario's middle off the lower ledge.
    rejected = Counter()
    sample = sample_block_smb_monte_carlo_scenario(
        split="train",
        seed=2026100909,
        sample_index=6,
        family="skill_gap_descent",
        difficulty="hard",
        rejection_counter=rejected,
    )
    assert rejected["edge_landing"] >= 1
    assert _route_lesson_fault(sample.scenario, sample.oracle["actions"]) is None
