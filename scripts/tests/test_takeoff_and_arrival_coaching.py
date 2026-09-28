"""Takeoff-timing labels, learner-arrival demonstrations, and piranha layout guarantees."""

import copy
from types import SimpleNamespace

from retroagi.stages.block_smb.env import MarioScenarioEnv
from retroagi.stages.block_smb.local_traversal import takeoff_timing_actions
from retroagi.stages.block_smb.monte_carlo import sample_block_smb_monte_carlo_scenario
from retroagi.stages.block_smb.piranha import runner_crossing_frame
from retroagi.stages.block_smb.piranha_tactics import (
    ARRIVAL_OVERRUN_MARGINS,
    arrival_demonstrations,
    staging_x,
)
from retroagi.stages.block_smb.primitive_execution import JumpReleaseState

SEED = 20260908


def piranha(mode, difficulty="medium", start=0):
    for index in range(start, start + 60):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            seed=SEED,
            split="train",
            sample_index=index,
            difficulty=difficulty,
        )
        if sample.parameters["crossing_mode"] == mode:
            return sample
    raise AssertionError(f"no {mode} layout sampled")


def test_release_state_charges_landing_frames_only_to_held_jumps():
    info = {"reward_terms": {"enemy_stomp": 0}}

    def remaining(frames):
        state = JumpReleaseState()
        for action, grounded in frames:
            state.observe(SimpleNamespace(mario={"on_ground": grounded}), action, info)
        return state.remaining

    # Released mid-air: the first grounded frame is a fresh decision.
    assert remaining([(2, False), (2, False), (1, False), (1, True)]) == 0
    # Still held on the landing step: release, then one suppressed frame.
    assert remaining([(2, False), (2, False), (2, True)]) == 2


def stomp_states(index, difficulty):
    """Takeoff labels at spawn, at the teacher's takeoff, and while running on."""
    sample = sample_block_smb_monte_carlo_scenario(
        family="enemy_stomp", seed=SEED, split="train", sample_index=index, difficulty=difficulty
    )
    env = MarioScenarioEnv()
    try:
        env.reset(scenario=copy.deepcopy(sample.scenario))
        env.render = lambda: None
        at_spawn = takeoff_timing_actions(env)
        actions = list(sample.oracle["actions"])
        takeoff = actions.index(2)
        for action in actions[:takeoff]:
            env.step(action)
        at_teacher = takeoff_timing_actions(env)
        # Keep running through the rest of the window.
        closing = []
        while True:
            label = takeoff_timing_actions(env)
            if label is None or not label[2]:
                break
            closing.append(label)
            _, _, done, _, info = env.step(1)
            if done or info["death"]:
                break
        return at_spawn, at_teacher, closing
    finally:
        env.close()


def test_stomp_takeoff_labels_forbid_the_thin_spawn_hop_and_allow_the_teacher_takeoff():
    run_only = [False, True, False, False, False, False]
    # Medium: from spawn only a few long holds reach and running widens the
    # window, so the spawn hop is labelled "keep running".
    at_spawn, at_teacher, closing = stomp_states(1, "medium")
    assert at_spawn == run_only
    assert at_teacher is not None and at_teacher[2]
    assert not any(at_teacher[i] for i in (0, 3, 4, 5))
    # The jump stays allowed while the window closes, and its last frame
    # requires it.
    assert len(closing) > 5
    assert closing[-1] == [False, False, True, False, False, False]

    # Easy: the stationary enemy is in a wide window from spawn, as the
    # teacher's own spawn takeoff shows.
    at_spawn, at_teacher, _ = stomp_states(0, "easy")
    assert at_spawn[2] and at_teacher[2]


def test_timed_plants_are_fully_raised_whenever_a_runner_would_cross():
    for start in (0, 20, 40):
        sample = piranha("timed", start=start)
        plant = sample.scenario["enemies"][0]
        crossing = runner_crossing_frame(sample.scenario, plant)
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=copy.deepcopy(sample.scenario))
            env.render = lambda: None
            heights = []
            for frame in range(crossing + 17):
                if frame >= crossing - 8:
                    heights.append(env.enemies[0]["h"])
                env.step(0)
        finally:
            env.close()
        assert heights and all(h == plant["plant_height"] for h in heights)


def test_hard_clearance_leaves_a_margin_over_the_raised_plant():
    for index in range(0, 30):
        sample = sample_block_smb_monte_carlo_scenario(
            family="piranha_avoidance",
            seed=SEED,
            split="train",
            sample_index=index,
            difficulty="hard",
        )
        parameters = sample.parameters
        if parameters["crossing_mode"] == "clearance":
            assert parameters["pipe_height"] + parameters["plant_height"] <= 56


def test_arrival_demonstrations_supervise_corrections_from_learner_arrivals():
    sample = piranha("timed")
    routes = arrival_demonstrations(sample, index=0)
    # Wall, spawn hop, and every overrun margin.
    assert len(routes) == 2 + len(ARRIVAL_OVERRUN_MARGINS)
    assert any(route.oracle["actions"][0] == 2 for route in routes)
    fast_overruns = 0
    for route in routes:
        start = route.oracle["supervision_start_frame"]
        actions = route.oracle["actions"]
        assert 0 < start < len(actions)
        env = MarioScenarioEnv()
        try:
            env.reset(scenario=copy.deepcopy(route.scenario))
            env.render = lambda: None
            for action in actions[:start]:
                env.step(action)
            if actions[0] == 1 and env.mario["vx"] > 2.5 and env.mario["x"] > staging_x(env) - 1:
                fast_overruns += 1
            for action in actions[start:]:
                _, _, done, truncated, info = env.step(action)
                assert not info["death"]
                if done or truncated:
                    break
            assert env._goal_credited
        finally:
            env.close()
    assert fast_overruns >= len(ARRIVAL_OVERRUN_MARGINS)

    clearance = arrival_demonstrations(piranha("clearance"), index=1)
    assert clearance and all(r.oracle["supervision_start_frame"] > 0 for r in clearance)


def test_arrival_hops_cover_every_duration_at_each_difficulty(monkeypatch):
    from retroagi.stages.block_smb import piranha_tactics

    sample = SimpleNamespace(scenario={"enemies": []})
    monkeypatch.setattr(piranha_tactics, "_spawn_hop_prefix", lambda hold: hold)
    monkeypatch.setattr(piranha_tactics, "_arrival_route", lambda sample, prefix: prefix)
    for tier in range(3):
        hops = [arrival_demonstrations(sample, index)[1] for index in range(tier, 9, 3)]
        assert hops == list(piranha_tactics.ARRIVAL_HOP_HOLDS)


def test_training_rollouts_narrow_the_spawn_stance_to_running():
    import torch

    from retroagi.stages.block_smb.train import (
        BlockSMBStage,
        block_smb_policy_scenario,
        collect_trajectory,
        make_block_smb_model,
    )
    from retroagi.stages.block_smb.vision import BlockVisionTransformer
    from scripts.tests.test_block_smb_training import tiny_config

    model = make_block_smb_model(tiny_config())
    for family, index in (("enemy_stomp", 1), ("piranha_avoidance", 0)):
        sample = sample_block_smb_monte_carlo_scenario(
            family=family, seed=SEED, split="train", sample_index=index, difficulty="medium"
        )
        stage = BlockSMBStage(
            env=MarioScenarioEnv(),
            scenario=block_smb_policy_scenario(copy.deepcopy(sample.scenario), True),
            vision=BlockVisionTransformer(),
        )
        try:
            trajectory = collect_trajectory(
                model,
                stage,
                sample.scenario_id,
                rollout_steps=2,
                seed=0,
                deterministic=False,
                device=torch.device("cpu"),
            )
        finally:
            stage.env.close()
        assert list(trajectory.transitions[0].tactic_actions) == [
            False,
            True,
            False,
            False,
            False,
            False,
        ], family
