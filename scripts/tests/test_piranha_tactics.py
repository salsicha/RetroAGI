"""Temporal decisions must be observable, executable, and jointly learned."""

from copy import deepcopy
from functools import lru_cache

import pytest
import torch

from retroagi.core.smb_enemy_history import EnemyObservationHistory
from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.env_state import restore_env_state, snapshot_env_state
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.piranha_tactics import (
    fresh_retraction,
    tactical_choice,
    timed_safe_holds,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@lru_cache(None)
def timed_sample(difficulty="hard"):
    return crossing_sample("timed", difficulty)


@lru_cache(None)
def crossing_sample(mode, difficulty="hard"):
    for seed in range(20):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            split="train",
            difficulty=difficulty,
            seed=seed,
            sample_index=0,
        )
        if sample.parameters["crossing_mode"] == mode:
            return sample
    raise AssertionError(f"No {mode} practice in the family")


def test_timing_targets_do_not_read_hidden_cycle_and_unknown_empty_pipe_waits():
    env = MarioScenarioEnv()
    sample = timed_sample()
    try:
        env.reset(scenario=sample.scenario)
        history = EnemyObservationHistory()
        for action in sample.oracle["actions"]:
            features = history.observe(env, env.steps)
            if action == 2 and env.mario["on_ground"]:
                break
            env.step(action)
        before = snapshot_env_state(env)
        reference = tactical_choice(env, features)
        assert reference[0:2] == ("advance", 2)
        # Same observed state, arbitrarily different privileged timers.
        for phase, hidden in [(0, 48), (50, 64), (120, 55)]:
            env.enemies[0].update(plant_tick=phase, hidden_frames=hidden, rise_frames=20)
            mutated = snapshot_env_state(env)
            assert tactical_choice(env, features) == reference
            assert snapshot_env_state(env) == mutated
        restore_env_state(env, before)
        unknown = EnemyObservationHistory().observe(env, env.steps)
        assert not fresh_retraction(unknown)
        assert tactical_choice(env, unknown)[0:2] == ("hold_area", 0)
    finally:
        env.close()


def _timed_samples(count=12, difficulty="medium"):
    samples = []
    for seed in range(60):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            split="train",
            difficulty=difficulty,
            seed=seed,
            sample_index=0,
        )
        if sample.parameters["crossing_mode"] == "timed":
            samples.append(sample)
            if len(samples) == count:
                break
    return samples


def _departure_state(env, sample):
    """Replay a teacher route to its first grounded departure; return its history."""
    env.reset(scenario=sample.scenario)
    history = EnemyObservationHistory()
    for action in sample.oracle["actions"]:
        features = history.observe(env, env.steps)
        if action == 2 and env.mario["on_ground"]:
            return features
        env.step(action)
    raise AssertionError("Route never departs")


def test_timed_teacher_stops_in_the_staging_window_without_overshoot():
    from retroagi.stages.block_smb.piranha_tactics import staging_x

    env = MarioScenarioEnv()
    try:
        for sample in _timed_samples():
            actions = sample.oracle["actions"]
            assert 3 not in actions  # No overshoot-and-retreat approach.
            env.reset(scenario=sample.scenario)
            target = staging_x(env)
            for action in actions:
                if action == 2 and env.mario["on_ground"]:
                    break
                if env.mario["on_ground"] and action == 0 and env.mario["vx"] == 0:
                    assert target - 2 <= env.mario["x"] <= target + 4
                env.step(action)
    finally:
        env.close()


def test_certified_departure_window_opens_on_observed_descent():
    import numpy as np

    from retroagi.stages.block_smb.piranha_tactics import (
        MIN_HIDDEN_FRAMES,
        TIMED_PLANT_HEIGHT,
        TIMED_RISE_FRAMES,
        timed_run_on,
    )

    sample = next(s for s in _timed_samples() if s.oracle["actions"].count(0) > 20)
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=sample.scenario)
        for action in sample.oracle["actions"]:
            if action == 0 and env.mario["vx"] == 0 and env.mario["on_ground"]:
                break
            env.step(action)
        saved = snapshot_env_state(env)

        def hidden(age):
            return np.array([0, 0, 0, -float(age == 1), age / 64, age / 64], dtype=np.float32)

        def seen(height, drop):
            restore_env_state(env, saved)
            plant = env.enemies[0]
            plant.update(h=height, y=plant["pipe_top"] - height)
            return np.array([1, drop / 8, 1, 0, 0, 0], dtype=np.float32)

        assert not fresh_retraction(hidden(MIN_HIDDEN_FRAMES))
        # A standing NES jump cannot clear the timed pipe, even just after it empties.
        assert not any(timed_safe_holds(env, hidden(age)) for age in range(1, MIN_HIDDEN_FRAMES))
        # A raised plant that is still or rising bounds no retraction.
        assert not timed_run_on(env, seen(TIMED_PLANT_HEIGHT, 0))
        assert not timed_run_on(env, seen(TIMED_PLANT_HEIGHT, -TIMED_PLANT_HEIGHT / 12))
        # An observed descent certifies a run-up for every rise duration in the family.
        for rise in TIMED_RISE_FRAMES:
            for height in range(TIMED_PLANT_HEIGHT, 0, -20):
                assert timed_run_on(env, seen(height, TIMED_PLANT_HEIGHT / rise))
        # Less remaining hidden time can only remove certified run-ups.
        restore_env_state(env, saved)
        run_on = [timed_run_on(env, hidden(age)) for age in range(1, MIN_HIDDEN_FRAMES, 2)]
        assert run_on == sorted(run_on, reverse=True) and not run_on[-1]
    finally:
        env.close()


def test_running_departures_are_certified():
    # Timed plants are raised whenever a runner arrives, so teacher routes
    # depart from a stop. Certification must still admit a departure made
    # at speed: retime a layout so the plant retracts as a runner approaches.
    from retroagi.stages.block_smb.piranha import runner_crossing_frame

    sample = timed_sample("medium")
    env = MarioScenarioEnv()
    try:
        for retracted_at in range(4, 40, 3):
            scenario = deepcopy(sample.scenario)
            plant = scenario["enemies"][0]
            rise, exposed = plant["rise_frames"], plant["exposed_frames"]
            period = 2 * rise + exposed + plant["hidden_frames"]
            plant["phase"] = (2 * rise + exposed - retracted_at) % period
            env.reset(scenario=scenario)
            history = EnemyObservationHistory()
            for _ in range(runner_crossing_frame(scenario, plant)):
                features = history.observe(env, env.steps)
                if env.mario["vx"] > 0.5 and fresh_retraction(features):
                    if timed_safe_holds(env, features):
                        return
                env.step(1)
        raise AssertionError("No certified running departure in the timed family")
    finally:
        env.close()
