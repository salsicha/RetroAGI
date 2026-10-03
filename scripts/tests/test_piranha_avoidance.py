"""Plant avoidance must use real lethal contact and observable exposure."""

import pytest

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.local_traversal import local_objective, safe_jump_holds
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.piranha import parse_plant


def contact_scenario(*, phase=20, mario=(120, 181)):
    return dict(
        world_width=288,
        mario=list(mario),
        platforms=[[0, 220, 288, 20]],
        enemies=[dict(kind="piranha_plant", x=120, pipe_top=220, phase=phase)],
        goal=[260, 204, 16, 16],
    )


# Mario's feet start 1 px above the plant's top, or beside its stem.
@pytest.mark.parametrize("mario", [(120, 185), (112, 204)])
def test_plant_contact_is_fatal_even_from_above(mario):
    env = MarioScenarioEnv()
    try:
        # Velocity lives in the NES motion state, so it is set by the scenario.
        env.reset(scenario={**contact_scenario(mario=mario), "mario_velocity": [0, 2]})
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
        # Feet 1 px above the Goomba's damage body (4 px above its feet).
        scenario = contact_scenario(mario=(120, 197))
        scenario["enemies"] = [[120, 206, 120, 120, 0]]
        scenario["mario_velocity"] = [0, 2]
        env.reset(scenario=scenario)
        _, _, _, _, info = env.step(0)
        assert not info["death"] and env.enemies[0]["dead"]
        assert info["reward_terms"]["enemy_stomp"] > 0
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
