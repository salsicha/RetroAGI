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


def test_a_landing_seen_a_frame_early_is_taught_what_he_can_do_standing():
    # Vision saw Mario land on a tall pipe a frame before the simulator did
    # (his feet a row above its top), and the jump ended there. In the air the
    # teacher only steered within 32 pixels; at 2.5 pixels a frame every such
    # run slid off the 30-pixel pipe, and it had no command (tall_pipe_jump).
    scenario = {
        "world_width": 320,
        "mario": [20, 204],
        "platforms": [[0, 220, 320, 20], [180, 163, 30, 57]],
        "goal": [266, 200, 16, 20],
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        for _ in range(4):  # Mario is held still for the first frames
            env.step(0)
        # A row above the pipe, falling, at 2.5 pixels a frame (the motion
        # model's sixteenths of a pixel), as the pipeline had him.
        env.mario.update(x=187.0, y=150.0, on_ground=False, _platform=None)
        motion = env.motion
        motion.x_speed, motion.moving, motion.facing = 40, 1, 1
        motion.y_speed, motion.y_fraction, motion.force = 2, 168, motion.force_down
        assert ct._landing_support(env) is env.platforms[1]
        state = episode_teacher(scenario)
        options = ct._run_options(env, state, ct.teacher_target(env), {})
        assert any(result.safe and goal.y > 0 for goal, result in options)
    finally:
        env.close()


def test_a_plant_is_down_between_its_fall_and_its_next_rise():
    plant = {"rise_frames": 10, "exposed_frames": 30, "hidden_frames": 40, "plant_tick": 0}
    assert ct._down_for(plant, 0) == 0  # rising
    assert ct._down_for(plant, 49) == 0  # one frame before it is down
    assert ct._down_for(plant, 50) == 40 and ct._down_for(plant, 85) == 5
    assert ct._down_for(plant, 90) == 0  # rising again


def test_no_run_is_aimed_into_the_goal_before_the_points_are_collected():
    # Reaching the goal without the strategy's points ends the lesson as
    # missed; ranked first, such runs took every trial (max_points_retreat).
    scenario = {
        "world_width": 256,
        "mario": [120, 208],
        "platforms": [[0, 220, 256, 20]],
        "coins": [[200, 209, 10, 10]],
        "goal": [8, 200, 16, 20],
        "goal_requires_support": True,
        "strategy": "max_points",
        "strategy_objective": {"points": 1},
    }
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        for _ in range(3):
            env.step(0)
        state = episode_teacher(scenario)
        m = env.mario
        runs = ct._screened_runs(env, state, ct.teacher_target(env))
        ends = [m["x"] + run.x for run in runs]
        assert ends and not any(env.goal.left - m["w"] < x < env.goal.right for x in ends)
        env.coins[0]["collected"] = True  # the point is made: the goal may be run to
        runs = ct._screened_runs(env, state, ct.teacher_target(env))
        assert any(env.goal.left - m["w"] < m["x"] + run.x < env.goal.right for run in runs)
    finally:
        env.close()


@pytest.mark.timeout(900)
def test_the_bridge_is_boarded_before_the_goal():
    # The goal counts only after a crossing on the bridge. Aimed at the floor
    # beyond, a jump over this hard layout's 116-pixel gap was taught, and
    # Mario stood in the goal uncredited until the time ran out.
    scenario = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=2026101201, sample_index=8, family="moving_bridge", difficulty="hard"
    ).scenario
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=scenario)
        target = ct.teacher_target(env)
        bridge = next(i for i, p in enumerate(env.platforms) if p.get("moving"))
        assert (target.kind, target.platform_index, target.stand) == ("mount", bridge, True)
    finally:
        env.close()
    played = replay(scenario, family="moving_bridge", max_frames=600)
    assert played["won"]


def test_a_segment_keeps_off_what_the_next_one_avoids():
    # From the top of a stairs section the teacher jumped onto the dead end's
    # high path, which the next segment avoids (tactics_bridge_then_gap).
    tactics = sample_block_smb_monte_carlo_scenario(
        split="validation", seed=2026101201, sample_index=0, family="tactics_bridge_then_gap", difficulty="easy"
    ).scenario["tactics"]
    # The advance into the dead end's corridor comes just before its retreat.
    corridor = next(i for i, seg in enumerate(tactics) if seg["stance"] == "retreat") - 1
    assert tactics[corridor]["avoid"]
    assert set(tactics[corridor]["avoid"]) <= set(tactics[corridor - 1]["avoid"])
