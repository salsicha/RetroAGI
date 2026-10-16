"""The teachers of the waiting and retreating tactic lessons: a plant's pipe is
crossed with a run-up while the plant is down, and a monster is jumped in the
open once it leaves its tunnel (both lessons were lost by the teacher until
2026-10-16)."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from retroagi.stages.block_smb import controller_teacher as ct
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.teacher_replay import replay
from retroagi.stages.block_smb.teacher_tokens import episode_teacher


def held_out(family, index):
    return sample_block_smb_monte_carlo_scenario(
        split="validation", seed=0, sample_index=index, family=family, difficulty="easy"
    ).scenario


def test_runs_along_the_floor_stop_before_a_pipe_in_their_way():
    # Ranked by progress, runs aimed into or behind a pipe came first; every
    # one stopped at the pipe and none to its foot was tried.
    env = MarioScenarioEnv()
    try:
        env.reset(
            scenario={
                "world_width": 352,
                "mario": [37, 208],
                "platforms": [[0, 220, 352, 20], [114, 194, 36, 26]],
                "goal": [324, 204, 16, 16],
            }
        )
        for _ in range(3):
            env.step(0)
        state = episode_teacher({"goal": [324, 204, 16, 16], "mario": [37, 208]})
        runs = ct._screened_runs(env, state, ct.teacher_target(env))
        m = env.mario
        ends = [m["x"] + m["w"] + run.x for run in runs if run.x > 0]
        assert ends and max(ends) <= 114
    finally:
        env.close()


def test_a_hidden_plant_is_predicted_to_rise_again():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=held_out("piranha_avoidance", 100))
        plant = env.enemies[0]
        while plant["h"] > 0:
            env.step(0)
        boxes = ct._enemy_paths(env, 200)[0]
        heights = [box[3] - box[1] for box in boxes]
        assert heights[0] == 0 and max(heights) == plant["plant_height"]
        # Frame by frame as the simulator moves it.
        seen = [plant["h"]]
        for _ in range(200):
            env.step(0)
            seen.append(env.enemies[0]["h"])
        assert heights == seen
    finally:
        env.close()


@pytest.mark.timeout(900)
@pytest.mark.parametrize("family", ["piranha_avoidance", "monster_retreat"])
def test_the_teacher_wins_the_lessons_it_lost(family):
    # piranha_avoidance: it walked to the pipe's foot, from where no jump
    # crosses, and stepped back and forth until the episode ran out.
    # monster_retreat: every jump over the monster was refused (its floor
    # compared with platform objects the trials had replaced), and Mario
    # held until it cornered him.
    played = replay(held_out(family, 101), family=family)
    assert played["won"]
