"""Plant avoidance must use real lethal contact and observable exposure."""

import pytest

from retroagi.core.smb_scene import block_oracle_scene
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.geometry_expert import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.piranha import parse_plant
from retroagi.stages.block_smb.primitive_execution import teacher_route_reachable


def contact_scenario(*, phase=20, mario=(120, 181)):
    return dict(
        world_width=288,
        mario=list(mario),
        platforms=[[0, 220, 288, 20]],
        enemies=[dict(kind="piranha_plant", x=120, pipe_top=220, phase=phase)],
        goal=[260, 204, 16, 16],
    )


@pytest.mark.parametrize("mario", [(120, 181), (112, 204)])
def test_plant_contact_is_fatal_even_from_above(mario):
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=contact_scenario(mario=mario))
        env.mario["vy"] = 2
        _, _, done, _, info = env.step(0)
        assert done and info["death"]
        assert info["reward_terms"]["enemy_hit"] < 0
        assert info["reward_terms"]["enemy_stomp"] == 0
        assert not env.enemies[0]["dead"] and not env._stomp_credited
    finally:
        env.close()


def test_ordinary_enemy_remains_stompable():
    env = MarioScenarioEnv()
    try:
        scenario = contact_scenario(mario=(120, 191))
        scenario["enemies"] = [[120, 206, 120, 120, 0]]
        env.reset(scenario=scenario)
        env.mario["vy"] = 2
        _, _, _, _, info = env.step(0)
        assert not info["death"] and env.enemies[0]["dead"]
        assert info["reward_terms"]["enemy_stomp"] > 0
    finally:
        env.close()


def test_hidden_plant_has_no_collision_or_observable_enemy_then_emerges():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=contact_scenario(phase=85, mario=(120, 204)))
        assert env.enemies[0]["h"] == 0
        assert not block_oracle_scene(env)["scene"].enemies
        assert local_objective(env).kind == "finish"
        for _ in range(27):
            _, _, done, _, info = env.step(0)
            assert not done and not info["death"]
        _, _, done, _, info = env.step(0)
        assert env.enemies[0]["h"] > 0 and block_oracle_scene(env)["scene"].enemies
        assert done and info["death"]
    finally:
        env.close()


def test_plant_cycle_and_probes_restore_phase_exactly():
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=contact_scenario(mario=(40, 204)))
        saved = snapshot_env_state(env)
        heights = []
        for _ in range(112):
            env.step(0)
            heights.append(env.enemies[0]["h"])
        assert 0 in heights and 24 in heights and any(0 < h < 24 for h in heights)
        assert env.enemies[0]["x"] == 120
        assert env.enemies[0]["y"] == saved["enemies"][0]["y"]
        restore_env_state(env, saved)
        safe_jump_holds(env, local_objective(env), 1)
        assert snapshot_env_state(env) == saved
    finally:
        env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_generated_routes_clear_live_plants_without_stomp_credit(difficulty):
    exposed_crossings = 0
    for seed in range(12):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            split="train",
            seed=seed,
            difficulty=difficulty,
            sample_index=0,
        )
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=sample.scenario)
            assert teacher_route_reachable(env, sample.oracle["actions"])
            for action in sample.oracle["actions"]:
                _, _, done, _, info = env.step(action)
                assert not info["death"] and info["reward_terms"]["enemy_stomp"] == 0
                plant = env.enemies[0]
                assert not plant["dead"]
                if (
                    plant["h"] > 0
                    and env.mario["x"] < plant["x"] + plant["w"]
                    and env.mario["x"] + env.mario["w"] > plant["x"]
                ):
                    exposed_crossings += 1
                if done:
                    break
            assert env._goal_credited
        finally:
            env.close()
    assert exposed_crossings > 0, "Family must teach passing exposed plants, not only empty pipes"


def test_invalid_plant_timing_is_rejected():
    with pytest.raises(ValueError):
        parse_plant(dict(x=120, pipe_top=180, rise_frames=0))


def test_plant_duration_labels_do_not_depend_on_cycle_phase():
    from retroagi.stages.block_smb.piranha import position_plant

    sample = sample_block_smb_monte_carlo_scenario(
        family="piranha_avoidance",
        split="train",
        seed=12,
        difficulty="medium",
        sample_index=0,
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        for action in sample.oracle["actions"]:
            if action == 2:
                break
            env.step(action)
        results = []
        for phase in (0, 2, 20, 65, 100):
            env.enemies[0]["plant_tick"] = phase
            position_plant(env.enemies[0])
            before = snapshot_env_state(env)
            results.append(safe_jump_holds(env, local_objective(env), 1))
            assert snapshot_env_state(env) == before
        assert results[0] and all(holds == results[0] for holds in results)
    finally:
        env.close()


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_conservative_routes_survive_shifted_cycles_through_real_executor(difficulty):
    from copy import deepcopy

    from retroagi.stages.block_smb.demonstrations import varied_demonstration

    sample = sample_block_smb_monte_carlo_scenario(
        family="piranha_avoidance",
        split="train",
        seed=13,
        difficulty=difficulty,
        sample_index=0,
    )
    variant = varied_demonstration(sample, 13, robust=True)
    assert variant is not None and variant.oracle["actions"] != sample.oracle["actions"]
    for phase in (0, 20, 70, 100):
        scenario = deepcopy(sample.scenario)
        scenario["enemies"][0]["phase"] = phase
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=scenario)
            assert teacher_route_reachable(env, sample.oracle["actions"])
            assert teacher_route_reachable(env, variant.oracle["actions"])
        finally:
            env.close()
